// tests/nvfp4_cpu_probe.cpp - P5 probe: correctness (vs fp64 host ref) + throughput of the NEON
// CPU cold-expert kernel on real pack blobs.  Standalone: g++ -O3 -march=<best> -I include.
//
// usage: nvfp4_cpu_probe <pack_dir> [layer_index] [entries]
// Prints machine-parseable lines:  REL_L2 blob=<i> val=<x>  /  THREADS <t> us_per_expert <x>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

#include "strata/kernels/nvfp4_cpu.h"

using namespace strata::kernels;

static double rel_l2(const float* a, const float* b, int64_t n) {
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) b[i] * (double) b[i];
    }
    return std::sqrt(num / (den + 1e-30));
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <pack_dir> [layer] [entries]\n", argv[0]);
        return 2;
    }
    const int layer = argc > 2 ? std::atoi(argv[2]) : 0;
    const int entries = argc > 3 ? std::atoi(argv[3]) : 20;
    char path[1024];
    std::snprintf(path, sizeof path, "%s/layer-%05d-cold-nvfp4.bin", argv[1], layer);
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "open failed: %s\n", path);
        return 2;
    }
    std::fseek(f, 0, SEEK_END);
    const int64_t fsize = std::ftell(f);
    const int nblobs = (int) (fsize / NVFP4_COLD_BLOB_BYTES);
    std::fprintf(stderr, "file %s: %lld bytes, %d blobs\n", path, (long long) fsize, nblobs);
    const int NB = nblobs < 8 ? nblobs : 8;
    std::vector<uint8_t> blobs((size_t) NB * NVFP4_COLD_BLOB_BYTES);
    for (int i = 0; i < NB; ++i) {
        std::fseek(f, (int64_t) i * NVFP4_COLD_BLOB_BYTES, SEEK_SET);
        if (std::fread(blobs.data() + (size_t) i * NVFP4_COLD_BLOB_BYTES, NVFP4_COLD_BLOB_BYTES, 1,
                       f) != 1) {
            std::fprintf(stderr, "read failed blob %d\n", i);
            return 2;
        }
    }
    std::fclose(f);

    float e4m3_lut[256];
    nvfp4_cpu_e4m3_table(e4m3_lut);

    // fixed pseudo-random activation (LCG), in [-1, 1)
    std::vector<float> x(NVFP4_H);
    uint32_t st = 0x1234567u;
    for (int i = 0; i < NVFP4_H; ++i) {
        st = st * 1664525u + 1013904223u;
        x[i] = ((float) (st >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
    }

    // correctness: NEON/scalar kernel vs fp64 host ref on NB blobs
    int pass = 0;
    for (int i = 0; i < NB; ++i) {
        std::vector<float> y_ref(NVFP4_H), y_cpu(NVFP4_H), h(NVFP4_FF);
        nvfp4_cold_expert_host_ref(blobs.data() + (size_t) i * NVFP4_COLD_BLOB_BYTES, x.data(),
                                   y_ref.data());
        nvfp4_cpu_expert(blobs.data() + (size_t) i * NVFP4_COLD_BLOB_BYTES, x.data(), y_cpu.data(),
                         h.data(), e4m3_lut);
        const double rel = rel_l2(y_cpu.data(), y_ref.data(), NVFP4_H);
        std::printf("REL_L2 blob=%d val=%.3e %s\n", i, rel, rel < 1e-3 ? "PASS" : "FAIL");
        if (rel < 1e-3) ++pass;
    }
    std::printf("CORRECTNESS %d/%d %s\n", pass, NB, pass == NB ? "PASS" : "FAIL");

    // throughput: `entries` expert-passes over blobs[i % NB] at several thread counts
    const int thread_counts[] = {1, 4, 8, 16, 20};
    for (int tc : thread_counts) {
        if (tc > entries) continue;
        std::vector<float> out((size_t) entries * NVFP4_H);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < tc; ++t) {
            pool.emplace_back([&, t] {
                std::vector<float> h(NVFP4_FF);
                for (int e = t; e < entries; e += tc)
                    nvfp4_cpu_expert(blobs.data() + (size_t) (e % NB) * NVFP4_COLD_BLOB_BYTES,
                                     x.data(), out.data() + (size_t) e * NVFP4_H, h.data(),
                                     e4m3_lut);
            });
        }
        for (auto& th : pool) th.join();
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::printf("THREADS %d entries %d wall_ms %.2f us_per_expert %.1f experts_per_s %.1f\n", tc,
                    entries, us / 1000.0, us / entries, entries / (us / 1e6));
    }

    // gang mode (row-striped, mirrors the worker's CpuPool): all `tc` threads + main cooperate on
    // ONE expert at a time, pure-spin barriers - the production latency shape.
    for (int tc : {4, 8, 12, 16}) {
        const int np = tc + 1;
        std::atomic<int> bar_count{0};
        std::atomic<uint32_t> bar_phase{0};
        std::atomic<uint64_t> gen{0};
        std::atomic<bool> stop{false};
        const uint8_t* cur = nullptr;
        float* outp = nullptr;
        std::vector<float> h(NVFP4_FF);
        auto barrier = [&] {
            const uint32_t p = bar_phase.load(std::memory_order_acquire);
            if (bar_count.fetch_add(1, std::memory_order_acq_rel) + 1 == np) {
                bar_count.store(0, std::memory_order_relaxed);
                bar_phase.store(p + 1, std::memory_order_release);
            } else {
                while (bar_phase.load(std::memory_order_acquire) == p) asm volatile("yield" ::: "memory");
            }
        };
        auto stripes = [&](int part) {
            const uint8_t* gate_w = cur + NVFP4_OFF_GATE_W;
            const uint8_t* up_w = cur + NVFP4_OFF_UP_W;
            const uint8_t* down_w = cur + NVFP4_OFF_DOWN_W;
            const uint8_t* gate_s = cur + NVFP4_OFF_GATE_S;
            const uint8_t* up_s = cur + NVFP4_OFF_UP_S;
            const uint8_t* down_s = cur + NVFP4_OFF_DOWN_S;
            float ws2[6];
            std::memcpy(ws2, cur + NVFP4_OFF_GLOBAL, sizeof ws2);
            const int64_t g0 = NVFP4_FF * part / np, g1 = NVFP4_FF * (part + 1) / np;
            for (int64_t i = g0; i < g1; ++i) {
                const float g = NVFP4_CPU_ROW_DOT(gate_w + i * (NVFP4_H / 2),
                                                  gate_s + i * (NVFP4_H / NVFP4_GROUP),
                                                  0.5f * ws2[0], x.data(), NVFP4_H, e4m3_lut);
                const float u = NVFP4_CPU_ROW_DOT(up_w + i * (NVFP4_H / 2),
                                                  up_s + i * (NVFP4_H / NVFP4_GROUP), 0.5f * ws2[2],
                                                  x.data(), NVFP4_H, e4m3_lut);
                h[i] = g / (1.0f + std::exp(-g)) * u;
            }
            barrier();
            const int64_t d0 = NVFP4_H * part / np, d1 = NVFP4_H * (part + 1) / np;
            for (int64_t r = d0; r < d1; ++r)
                outp[r] = NVFP4_CPU_ROW_DOT(down_w + r * (NVFP4_FF / 2),
                                            down_s + r * (NVFP4_FF / NVFP4_GROUP), 0.5f * ws2[4],
                                            h.data(), NVFP4_FF, e4m3_lut);
            barrier();
        };
        std::vector<std::thread> pool;
        for (int t = 0; t < tc; ++t)
            pool.emplace_back([&, t] {
                uint64_t seen = 0;
                for (;;) {
                    while (gen.load(std::memory_order_acquire) == seen)
                        if (stop.load(std::memory_order_relaxed)) return;
                    seen = gen.load(std::memory_order_acquire);
                    stripes(t);
                }
            });
        std::vector<float> out((size_t) entries * NVFP4_H);
        // warmup one entry
        cur = blobs.data(); outp = out.data();
        gen.fetch_add(1); stripes(tc);
        const auto t0 = std::chrono::steady_clock::now();
        for (int e = 0; e < entries; ++e) {
            cur = blobs.data() + (size_t) (e % NB) * NVFP4_COLD_BLOB_BYTES;
            outp = out.data() + (size_t) e * NVFP4_H;
            gen.fetch_add(1, std::memory_order_release);
            stripes(tc);
        }
        const auto t1 = std::chrono::steady_clock::now();
        stop.store(true);
        gen.fetch_add(1);
        for (auto& th : pool) th.join();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::printf("GANG threads=%d entries=%d us_per_expert=%.1f\n", tc, entries, us / entries);
    }
    return pass == NB ? 0 : 1;
}

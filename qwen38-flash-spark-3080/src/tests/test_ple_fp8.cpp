// tests/test_ple_fp8.cpp - P4b fixture test: the PLE table's FP8 E4M3 row format (NVFP4 stack).
//
// Self-contained (no model, no RDMA): a synthetic 64-row table served through a loopback PleRemoteRows,
// opened via PleTable in Rdma mode exactly the way generate.cpp does it after fetch_ple_info. Covers:
//   A. fp8e4m3_dequant_row against an independent scalar e4m3 reference (all 256 codes, NaN included)
//   B. the full FP8 path: GGUF shape oracle + remote rows + row cache + gather/gather_batch/read_row
//   C. the IQ4_NL arm through the SAME plumbing (regression: the Q2 stack's path is untouched)
//   D. out-of-range rows decode as zeros (the mmap path's documented behaviour)
#include "strata/kernels/ngram.hpp"
#include "strata/ngram/ple_reader.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strata;

namespace {

constexpr uint64_t N_ROWS = 64;
constexpr float SCALE = 0.00019931793212890625f;

// Independent scalar e4m3 reference (double math, deliberately NOT the engine's table).
double ref_e4m3(uint8_t b) {
    const int s = (b & 0x80) ? -1 : 1;
    const int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 15 && m == 7) return std::nan("");
    double v;
    if (e == 0) v = (m / 8.0) * 0.015625;
    else v = (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return s * v;
}

struct FakeRows final : ngram::PleRemoteRows {
    const uint8_t* table = nullptr;
    uint32_t rb = 0;
    bool read_rows(const uint32_t* rows, size_t n, uint8_t* dst, std::string& err) override {
        for (size_t i = 0; i < n; ++i) std::memcpy(dst + i * rb, table + (size_t) rows[i] * rb, rb);
        return true;
    }
};

// Minimal GGUF v3 writer for the shape-oracle shard: one IQ4_NL tensor [160, N_ROWS].
bool write_fixture_gguf(const std::string& path, const std::vector<uint8_t>& rows90, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "cannot write " + path; return false; }
    auto u32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    auto u64 = [&](uint64_t v) { std::fwrite(&v, 8, 1, f); };
    auto str = [&](const char* s) { u64(std::strlen(s)); std::fwrite(s, 1, std::strlen(s), f); };
    u32(0x46554747u);                       // "GGUF"
    u32(3);                                 // v3
    u64(1);                                 // n_tensors
    u64(1);                                 // n_kv
    str("general.architecture");
    u32(8);                                 // string
    str("qwen4exp");
    str("per_layer_token_embd.weight");
    u32(2);                                 // n_dims
    u64(160); u64(N_ROWS);
    u32(20);                                // IQ4_NL
    u64(0);                                 // offset in data section
    const long pos = std::ftell(f);
    const long pad = (32 - pos % 32) % 32;
    for (long i = 0; i < pad; ++i) std::fputc(0, f);
    std::fwrite(rows90.data(), 1, rows90.size(), f);
    std::fclose(f);
    return true;
}

uint16_t f16_bits(float v) {
    // round-to-nearest-even fp32->fp16, enough for the fixture's d values (0.5, -0.25, 2.0)
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int e = (int) ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e <= 0) return (uint16_t) sign;
    if (e >= 31) return (uint16_t) (sign | 0x7BFF);
    m += 0x1000;
    if (m & 0x800000) { m = 0; e += 1; }
    return (uint16_t) (sign | ((uint32_t) e << 10) | (m >> 13));
}

int failures = 0;
void check(bool ok, const char* what) {
    std::fprintf(stderr, "%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

bool rows_close(const float* a, const float* b, int n, double tol) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) {
        if (std::isnan(a[i]) && std::isnan(b[i])) continue;
        num += (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den == 0 ? num == 0 : std::sqrt(num / den) <= tol;
}

}  // namespace

int main() {
    // ---- A. exhaustive e4m3 decode vs scalar reference
    {
        uint8_t row[kernels::PLE_HEAD_DIM];
        float out[kernels::PLE_HEAD_DIM];
        bool ok = true;
        for (int b = 0; b < 256; ++b) {
            std::memset(row, b, sizeof row);
            kernels::fp8e4m3_dequant_row(row, 2.0f, out);
            const double want = ref_e4m3((uint8_t) b) * 2.0;
            if (std::isnan(want)) { if (!std::isnan(out[0])) ok = false; }
            else if (std::fabs(out[0] - want) > std::fabs(want) * 1e-6 + 1e-12) ok = false;
        }
        check(ok, "A: fp8e4m3_dequant_row matches the scalar reference on all 256 codes");
    }

    // ---- fixture tables
    std::vector<uint8_t> fp8_table(N_ROWS * 160);
    for (size_t i = 0; i < fp8_table.size(); ++i) {
        uint8_t b = (uint8_t) ((i * 37 + (i / 160) * 11) & 0xFF);
        if ((b & 0x7F) == 0x7F) b ^= 0x01;      // keep NaN codes out of the table
        fp8_table[i] = b;
    }
    std::vector<uint8_t> iq_table(N_ROWS * 90);
    for (uint64_t r = 0; r < N_ROWS; ++r)
        for (int blk = 0; blk < 5; ++blk) {
            uint8_t* p = iq_table.data() + r * 90 + blk * 18;
            const uint16_t d = f16_bits((r % 3 == 0 ? 0.5f : r % 3 == 1 ? -0.25f : 2.0f));
            std::memcpy(p, &d, 2);
            for (int j = 0; j < 16; ++j) p[2 + j] = (uint8_t) (r * 13 + blk * 7 + j * 3);
        }

    const std::string gguf_path = "test_ple_fp8_fixture.gguf";
    std::string err;
    if (!write_fixture_gguf(gguf_path, iq_table, err)) { std::fprintf(stderr, "FAIL fixture: %s\n", err.c_str()); return 1; }

    FakeRows fake;
    // ---- B. the FP8 path end to end
    {
        fake.table = fp8_table.data();
        fake.rb = 160;
        kernels::PleTable t;
        kernels::PleIoOptions io;
        io.mode = kernels::PleIo::Rdma;
        io.remote = &fake;
        io.format = kernels::PleRowFormat::Fp8E4m3;
        io.fp8_scale = SCALE;
        io.cache_rows = 16;
        check(t.open(gguf_path, err, io), "B0: PleTable opens (FP8 remote + GGUF shape oracle)");

        uint32_t rows16[kernels::PLE_N_HEADS];
        for (int h = 0; h < kernels::PLE_N_HEADS; ++h) rows16[h] = (uint32_t) (h * 3 + 1);
        float out[2560];
        check(t.issue(rows16) && t.collect(out, err), "B1: issue+collect (FP8)");
        bool ok = true;
        for (int h = 0; h < kernels::PLE_N_HEADS; ++h)
            for (int i = 0; i < 160; ++i) {
                const float want = (float) ref_e4m3(fp8_table[(size_t) rows16[h] * 160 + i]) * SCALE;
                if (std::fabs(out[h * 160 + i] - want) > std::fabs(want) * 1e-6 + 1e-12) ok = false;
            }
        check(ok, "B2: gather decodes FP8 rows head-slowest with the global scale");

        // row-cache arm: the same gather again must serve identical bytes from the cache
        float out2[2560];
        check(t.gather_batch(rows16, 1, out2, err) && std::memcmp(out, out2, sizeof out) == 0,
              "B3: gather_batch agrees with issue/collect (cache arm)");

        // one direct read_row + an out-of-range row
        float r160[160];
        t.read_row(7, r160);
        bool ok7 = true;
        for (int i = 0; i < 160; ++i) {
            const float want = (float) ref_e4m3(fp8_table[(size_t) 7 * 160 + i]) * SCALE;
            if (std::fabs(r160[i] - want) > std::fabs(want) * 1e-6 + 1e-12) ok7 = false;
        }
        check(ok7, "B4: read_row (FP8)");
        t.read_row((uint32_t) N_ROWS + 5, r160);
        bool zero = true;
        for (int i = 0; i < 160; ++i) zero &= r160[i] == 0.0f;
        check(zero, "B5: out-of-range row decodes as zeros");
        t.close();
    }

    // ---- C. the IQ4_NL arm through the same plumbing (Q2 regression)
    {
        fake.table = iq_table.data();
        fake.rb = 90;
        kernels::PleTable t;
        kernels::PleIoOptions io;
        io.mode = kernels::PleIo::Rdma;
        io.remote = &fake;
        io.cache_rows = 16;
        check(t.open(gguf_path, err, io), "C0: PleTable opens (IQ4_NL remote)");
        uint32_t rows16[kernels::PLE_N_HEADS];
        for (int h = 0; h < kernels::PLE_N_HEADS; ++h) rows16[h] = (uint32_t) (h * 3 + 1);
        float out[2560];
        check(t.issue(rows16) && t.collect(out, err), "C1: issue+collect (IQ4_NL)");
        bool ok = true;
        for (int h = 0; h < kernels::PLE_N_HEADS; ++h) {
            float want[160];
            kernels::iq4nl_dequant_row(iq_table.data() + (size_t) rows16[h] * 90, want);
            if (!rows_close(out + h * 160, want, 160, 0.0)) ok = false;
        }
        check(ok, "C2: gather decodes IQ4_NL rows exactly as before");
        t.close();
    }

    // ---- D. FP8 without a scale is fail-closed
    {
        kernels::PleTable t;
        kernels::PleIoOptions io;
        io.mode = kernels::PleIo::Rdma;
        io.remote = &fake;
        io.format = kernels::PleRowFormat::Fp8E4m3;
        io.fp8_scale = 0.0f;
        check(!t.open(gguf_path, err, io), "D: FP8 without a scale is refused");
    }

    std::remove(gguf_path.c_str());
    std::fprintf(stderr, "%s\n", failures == 0 ? "PLE_FP8_TEST_PASS" : "PLE_FP8_TEST_FAIL");
    return failures == 0 ? 0 : 1;
}

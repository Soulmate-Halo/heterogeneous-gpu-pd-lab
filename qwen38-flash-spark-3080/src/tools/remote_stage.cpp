#include "strata/core/remote_stage.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using strata::core::RemoteStage;
using strata::core::RemoteStageAck;
using strata::core::RemoteStageConfig;
using strata::core::RemoteStageRole;
using strata::core::RemoteStageWindow;

namespace {

void usage() {
    std::cerr << "usage: strata-remote-stage --role sender|receiver [--peer HOST] [--bind HOST] "
                 "[--port N] [--split-layer N] [--tokens N] [--hidden N] [--windows N] [--device HCA]\n"
                 "                        [--slots N] [--slot-bytes B] [--mode window|selftest] [--timeout-ms N]\n";
}

bool value(const std::vector<std::string>& args, size_t& i, const char* name, std::string& out) {
    if (args[i] != name || i + 1 >= args.size()) return false;
    out = args[++i];
    return true;
}

int number(const std::string& s, int fallback) {
    try { return std::stoi(s); } catch (...) { return fallback; }
}

long long number_ll(const std::string& s, long long fallback) {
    try { return std::stoll(s); } catch (...) { return fallback; }
}

uint32_t crc32(const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= p[i];
        for (int b = 0; b < 8; ++b) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

struct Latency {
    std::vector<double> us;
    void add(double v) { us.push_back(v); }
    void sort() { std::sort(us.begin(), us.end()); }
    double mean() const {
        if (us.empty()) return 0.0;
        double s = 0; for (double v : us) s += v; return s / us.size();
    }
    double percentile(double q) const {
        if (us.empty()) return 0.0;
        size_t i = static_cast<size_t>(q * static_cast<double>(us.size() - 1));
        return us[i];
    }
};

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--selftest") {
        std::string error;
        const bool ok = RemoteStage::selftest(error);
        if (ok) {
            std::cout << "remote_stage_selftest PASS\n";
            return 0;
        }
        std::cerr << "remote_stage_selftest FAIL: " << error << "\n";
        return 1;
    }

    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    RemoteStageConfig config;
    int tokens = 1;
    int hidden = 2560;
    int windows = 1;
    int timeout_ms = config.timeout_ms;
    std::string mode = "window";
    for (size_t i = 0; i < args.size(); ++i) {
        std::string v;
        if (value(args, i, "--role", v)) {
            if (v == "sender") config.role = RemoteStageRole::Sender;
            else if (v == "receiver") config.role = RemoteStageRole::Receiver;
            else { usage(); return 2; }
        } else if (value(args, i, "--peer", config.peer_host)) {
        } else if (value(args, i, "--bind", config.bind_host)) {
        } else if (value(args, i, "--device", config.device)) {
        } else if (value(args, i, "--port", v)) {
            config.port = static_cast<uint16_t>(number(v, config.port));
        } else if (value(args, i, "--split-layer", v)) {
            config.split_layer = number(v, config.split_layer);
        } else if (value(args, i, "--tokens", v)) {
            tokens = number(v, tokens);
        } else if (value(args, i, "--hidden", v)) {
            hidden = number(v, hidden);
        } else if (value(args, i, "--windows", v)) {
            windows = number(v, windows);
        } else if (value(args, i, "--slots", v)) {
            config.slots = static_cast<uint32_t>(number(v, static_cast<int>(config.slots)));
        } else if (value(args, i, "--slot-bytes", v)) {
            config.slot_bytes = static_cast<size_t>(number_ll(v, static_cast<long long>(config.slot_bytes)));
        } else if (value(args, i, "--mode", v)) {
            mode = v;
        } else if (value(args, i, "--timeout-ms", v)) {
            timeout_ms = number(v, timeout_ms);
        } else if (args[i] == "--no-rdma") {
            config.require_rdma = false;
        } else {
            usage();
            return 2;
        }
    }
    if (config.role == RemoteStageRole::Sender && config.peer_host.empty()) {
        usage();
        return 2;
    }
    if (tokens <= 0 || hidden <= 0 || windows <= 0 || config.slots == 0 || config.slot_bytes < 16) {
        std::cerr << "invalid geometry\n";
        return 2;
    }
    if (mode != "window" && mode != "selftest") {
        usage();
        return 2;
    }
    config.timeout_ms = timeout_ms;

    RemoteStage stage(config);
    std::string error;
    if (!stage.open(error)) {
        std::cerr << "remote_stage OPEN FAIL: " << error << "\n";
        return 1;
    }
    std::cout << "remote_stage OPEN PASS rdma=" << (stage.is_rdma() ? "true" : "false") << "\n";

    if (mode == "selftest") {
        // Full one-sided flag-ring loop: the sender posts an activation plus a
        // doorbell sequence, the receiver host-polls the doorbell, transforms
        // the payload (byte-wise XOR 0x5A) into a ring slot whose 8-byte header
        // is the new sequence (release order), and the sender RDMA-reads the
        // whole slot back and verifies transform plus CRC.  M windows rotate
        // over N slots; each side reports its per-hop latency in microseconds.
        const size_t payload_bytes = config.slot_bytes - 8;
        std::vector<uint8_t> payload(payload_bytes);
        for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);
        std::vector<uint8_t> expected(payload_bytes);
        for (size_t i = 0; i < payload.size(); ++i) expected[i] = static_cast<uint8_t>(payload[i] ^ 0x5Au);
        const uint32_t expected_crc = crc32(expected.data(), expected.size());
        Latency lat;
        bool ok = true;
        const auto t_start = std::chrono::steady_clock::now();
        const auto deadline = t_start + std::chrono::milliseconds(static_cast<long long>(config.timeout_ms) * 8LL);
        for (int w = 0; w < windows && ok; ++w) {
            const uint64_t seq = static_cast<uint64_t>(w + 1);
            const uint32_t slot = static_cast<uint32_t>(w) % config.slots;
            if (config.role == RemoteStageRole::Sender) {
                const auto t0 = std::chrono::steady_clock::now();
                if (!stage.rdma_write_doorbell(slot, payload.data(), payload.size(), seq, error)) {
                    std::cerr << "remote_stage RING SEND FAIL w=" << w << ": " << error << "\n";
                    ok = false; break;
                }
                // poll the ring slot header until it carries this sequence
                std::vector<uint8_t> slotbuf(config.slot_bytes, 0);
                uint64_t hdr = 0;
                while (true) {
                    if (!stage.rdma_read_slot(slot, slotbuf.data(), slotbuf.size(), error)) {
                        std::cerr << "remote_stage RING READ FAIL w=" << w << ": " << error << "\n";
                        ok = false; break;
                    }
                    std::memcpy(&hdr, slotbuf.data(), 8);
                    if (hdr == seq) break;
                    if (std::chrono::steady_clock::now() > deadline) {
                        std::cerr << "remote_stage RING POLL TIMEOUT w=" << w << " hdr=" << hdr << "\n";
                        ok = false; break;
                    }
                }
                if (!ok) break;
                const uint32_t got_crc = crc32(slotbuf.data() + 8, payload_bytes);
                if (got_crc != expected_crc) {
                    std::cerr << "remote_stage RING CRC FAIL w=" << w << " got=" << got_crc
                              << " want=" << expected_crc << "\n";
                    ok = false; break;
                }
                const auto t1 = std::chrono::steady_clock::now();
                lat.add(std::chrono::duration<double, std::micro>(t1 - t0).count());
            } else {
                // receiver: host-poll doorbell, transform, publish ring slot
                while (stage.load_local_doorbell() != seq) {
                    if (std::chrono::steady_clock::now() > deadline) {
                        std::cerr << "remote_stage RING DOORBELL TIMEOUT w=" << w
                                  << " seen=" << stage.load_local_doorbell() << "\n";
                        ok = false; break;
                    }
                }
                if (!ok) break;
                const auto t0 = std::chrono::steady_clock::now();
                const uint8_t* inbox = stage.local_inbox();
                uint8_t* slotp = stage.local_ring_slot(slot);
                if (!inbox || !slotp) { std::cerr << "remote_stage RING LOCAL FAIL\n"; ok = false; break; }
                for (size_t i = 0; i < payload_bytes; ++i) slotp[8 + i] = static_cast<uint8_t>(inbox[i] ^ 0x5Au);
                std::atomic_thread_fence(std::memory_order_release);
                uint64_t publish = seq;
                std::memcpy(slotp, &publish, 8);  // seq header last, release order
                const auto t1 = std::chrono::steady_clock::now();
                lat.add(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
        }
        if (ok) {
            lat.sort();
            std::cout << "remote_stage RING PASS windows=" << windows
                      << " slots=" << config.slots << " slot_bytes=" << config.slot_bytes
                      << " mean_us=" << lat.mean() << " p50_us=" << lat.percentile(0.50)
                      << " p99_us=" << lat.percentile(0.99) << "\n";
            stage.close();
            return 0;
        }
        stage.close();
        return 1;
    }

    const size_t payload_bytes = static_cast<size_t>(tokens) * static_cast<size_t>(hidden) * sizeof(float);
    std::vector<uint8_t> payload(payload_bytes);
    std::vector<uint8_t> received(payload_bytes);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);

    for (int i = 0; i < windows; ++i) {
        const uint64_t sequence = static_cast<uint64_t>(i + 1);
        if (config.role == RemoteStageRole::Sender) {
            RemoteStageWindow window{sequence, static_cast<uint32_t>(tokens), static_cast<uint32_t>(hidden), config.split_layer};
            RemoteStageAck ack{};
            if (!stage.send_window(window, payload.data(), payload.size(), ack, error)) {
                std::cerr << "remote_stage SEND FAIL: " << error << "\n";
                stage.close();
                return 1;
            }
            std::cout << "window=" << sequence << " ack=" << ack.accepted << " status=" << ack.status << "\n";
        } else {
            RemoteStageWindow window{};
            if (!stage.recv_window(window, received.data(), received.size(), error)) {
                std::cerr << "remote_stage RECV FAIL: " << error << "\n";
                stage.close();
                return 1;
            }
            RemoteStageAck ack{window.sequence, window.tokens, 0};
            if (!stage.send_ack(ack, error)) {
                std::cerr << "remote_stage ACK FAIL: " << error << "\n";
                stage.close();
                return 1;
            }
            std::cout << "window=" << window.sequence << " tokens=" << window.tokens
                      << " hidden=" << window.hidden << " split_layer=" << window.split_layer << "\n";
        }
    }
    std::cout << "remote_stage PASS windows=" << stage.windows()
              << " bytes_sent=" << stage.bytes_sent()
              << " bytes_received=" << stage.bytes_received() << "\n";
    stage.close();
    return 0;
}

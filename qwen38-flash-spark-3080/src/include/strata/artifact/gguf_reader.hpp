// include/strata/artifact/gguf_reader.hpp - generated from src/artifact/gguf_reader.cpp by
// scripts/split_artifact.py.  Header-only on purpose: the reader is one translation unit's worth of
// code with no state to hide, and a header-only split cannot introduce a duplicate-symbol or
// missing-declaration bug in code that is already validated.
#pragma once
// src/artifact/gguf_reader.cpp - P1.S2: GGUF v3 reader, mmap, no ggml dependency.
//
// The C++ counterpart of tools/gguf_reader.py, which has been the reference since P0.S6 (written
// because gguf-py cannot represent type 42 / Q2_0). Per P1.S2 it must:
//   * mmap the file and parse header, metadata KV (ALL value types incl. arrays), tensor directory
//   * be multi-shard aware (split.* keys)
//   * carry an architecture guard: general.architecture == "qwen4exp" and the compiled-in constants
//     must match, refusing with a precise error otherwise
//   * have no ggml dependency
//
// This file is the reader plus a `--check` mode that validates it the way the Python one is
// validated: parse the tiny model AND both real shards, and report counts that other harnesses have
// already established independently (1,223 / 1 tensors; 202 Q2_0 in shard 1). Agreement with those
// numbers is the test.
//
// Build: scripts/build_artifact.bat        Run: gguf_reader.exe <file.gguf> [--check]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stdexcept>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata {

// ---- ggml type ids we care about. 42 = Q2_0, the PrismML ternary 2-bit encoding this engine targets.
inline const char* ggml_type_name(uint32_t t) {
    switch (t) {
    case 0:
        return "F32";
    case 1:
        return "F16";
    case 2:
        return "Q4_0";
    case 3:
        return "Q4_1";
    case 6:
        return "Q5_0";
    case 7:
        return "Q5_1";
    case 8:
        return "Q8_0";
    case 9:
        return "Q8_1";
    case 10:
        return "Q2_K";
    case 11:
        return "Q3_K";
    case 12:
        return "Q4_K";
    case 13:
        return "Q5_K";
    case 14:
        return "Q6_K";
    case 15:
        return "Q8_K";
    case 16:
        return "IQ2_XXS";
    case 17:
        return "IQ2_XS";
    case 18:
        return "IQ3_XXS";
    case 19:
        return "IQ1_S";
    case 20:
        return "IQ4_NL";
    case 21:
        return "IQ3_S";
    case 22:
        return "IQ2_S";
    case 23:
        return "IQ4_XS";
    case 30:
        return "BF16";
    case 34:
        return "TQ1_0";
    case 35:
        return "TQ2_0";
    case 39:
        return "MXFP4";
    case 40:
        return "NVFP4";
    case 41:
        return "Q1_0";
    case 42:
        return "Q2_0";
    default:
        return "?";
    }
}

// Block geometry: (elements per block, bytes per block). Q2_0 is 64/18 - proven from this artifact's
// own offset brackets in P0.S6 and recorded in docs/q2_0-contract.md.
inline bool block_geometry(uint32_t t, int& elems, int& bytes) {
    switch (t) {
    case 0:
        elems = 1;
        bytes = 4;
        return true;
    case 1:
    case 30:
        elems = 1;
        bytes = 2;
        return true;
    case 2:
        elems = 32;
        bytes = 18;
        return true;
    case 3:
        elems = 32;
        bytes = 20;
        return true;
    case 6:
        elems = 32;
        bytes = 22;
        return true;
    case 7:
        elems = 32;
        bytes = 24;
        return true;
    case 8:
        elems = 32;
        bytes = 34;
        return true;
    case 9:
        elems = 32;
        bytes = 36;
        return true;
    case 10:
        elems = 256;
        bytes = 84;
        return true;
    case 11:
        elems = 256;
        bytes = 110;
        return true;
    case 12:
        elems = 256;
        bytes = 144;
        return true;
    case 13:
        elems = 256;
        bytes = 176;
        return true;
    case 14:
        elems = 256;
        bytes = 210;
        return true;
    case 16:
        elems = 256;
        bytes = 66;
        return true;
    case 17:
        elems = 256;
        bytes = 74;
        return true;
    case 18:
        elems = 256;
        bytes = 98;
        return true;
    case 20:
        elems = 32;
        bytes = 18;
        return true;
    case 21:   // IQ3_S
        elems = 256;
        bytes = 110;
        return true;
    case 22:   // IQ2_S
        elems = 256;
        bytes = 82;
        return true;
    case 23:
        elems = 256;
        bytes = 136;
        return true;
    case 29:   // IQ1_M
        elems = 256;
        bytes = 56;
        return true;
    case 42:
        elems = 64;
        bytes = 18;
        return true;
    default:
        return false;
    }
}

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> shape; // GGUF order: dim 0 varies fastest
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t elements() const {
        uint64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    const char* type_name() const { return ggml_type_name(type); }
};

// A bounds-checked cursor over the mmapped header region. Every read is checked, so a truncated or
// corrupt file produces a precise error rather than a segfault.
class Cursor {
public:
    Cursor(const uint8_t* base, size_t size) : base_(base), size_(size) {}
    void need(size_t n) const {
        if (pos_ + n > size_) throw std::runtime_error("GGUF: unexpected end of file in header");
    }
    template <class T> T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = read<uint64_t>();
        need((size_t)n);
        std::string s(reinterpret_cast<const char*>(base_ + pos_), (size_t)n);
        pos_ += (size_t)n;
        return s;
    }
    size_t pos() const { return pos_; }

private:
    const uint8_t* base_;
    size_t size_;
    size_t pos_ = 0;
};

enum class MetaType : uint32_t { U8 = 0, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 };

struct MetaValue {
    MetaType type = MetaType::U32;
    uint64_t u = 0;                // integer payloads
    double f = 0;                  // float payloads
    std::string s;                 // string payloads
    MetaType elem = MetaType::U32; // arrays
    uint64_t count = 0;
    std::vector<MetaValue> items;
    bool is_num() const { return type != MetaType::STRING && type != MetaType::ARRAY; }
    double num() const { return (type == MetaType::F32 || type == MetaType::F64) ? f : (double)u; }
};

inline MetaValue read_value(Cursor& c, MetaType t, int depth = 0) {
    if (depth > 2) throw std::runtime_error("GGUF: array nesting too deep");
    MetaValue v;
    v.type = t;
    switch (t) {
    case MetaType::U8:
        v.u = c.read<uint8_t>();
        break;
    case MetaType::I8:
        v.u = (uint64_t)(int64_t)c.read<int8_t>();
        break;
    case MetaType::U16:
        v.u = c.read<uint16_t>();
        break;
    case MetaType::I16:
        v.u = (uint64_t)(int64_t)c.read<int16_t>();
        break;
    case MetaType::U32:
        v.u = c.read<uint32_t>();
        break;
    case MetaType::I32:
        v.u = (uint64_t)(int64_t)c.read<int32_t>();
        break;
    case MetaType::F32: {
        float x = c.read<float>();
        v.f = x;
        v.u = 0;
        break;
    }
    case MetaType::BOOL:
        v.u = c.read<uint8_t>() ? 1 : 0;
        break;
    case MetaType::STRING:
        v.s = c.str();
        break;
    case MetaType::U64:
        v.u = c.read<uint64_t>();
        break;
    case MetaType::I64:
        v.u = (uint64_t)c.read<int64_t>();
        break;
    case MetaType::F64:
        v.f = c.read<double>();
        break;
    case MetaType::ARRAY: {
        v.elem = (MetaType)c.read<uint32_t>();
        v.count = c.read<uint64_t>();
        if (v.count > (1u << 24)) throw std::runtime_error("GGUF: implausible array length");
        v.items.reserve((size_t)std::min<uint64_t>(v.count, 64));
        for (uint64_t i = 0; i < v.count; ++i) {
            MetaValue e = read_value(c, v.elem, depth + 1);
            if (i < 64) v.items.push_back(std::move(e)); // keep a sample; the count is what matters
        }
        break;
    }
    default:
        throw std::runtime_error("GGUF: unknown metadata value type");
    }
    return v;
}

class GgufFile {
public:
    explicit GgufFile(const std::string& path) : path_(path) {
        try { open(); }
        catch (...) { close(); throw; }
    }
    ~GgufFile() { close(); }
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string, MetaValue>& metadata() const { return meta_; }
    uint32_t version() const { return version_; }
    uint64_t data_start() const { return data_start_; }
    uint64_t file_size() const { return size_; }

    const TensorInfo* find(const std::string& name) const {
        for (const auto& t : tensors_)
            if (t.name == name) return &t;
        return nullptr;
    }
    uint64_t count_type(const char* tn) const {
        uint64_t n = 0;
        for (const auto& t : tensors_)
            if (std::strcmp(t.type_name(), tn) == 0) ++n;
        return n;
    }
    const MetaValue* get(const std::string& key) const {
        auto it = meta_.find(key);
        return it == meta_.end() ? nullptr : &it->second;
    }
    const uint8_t* tensor_data(const TensorInfo& t) const { return base_ + data_start_ + t.offset; }

private:
    void open() {
#ifdef _WIN32
        HANDLE h = CreateFileA(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path_);
        LARGE_INTEGER li{};
        GetFileSizeEx(h, &li);
        size_ = (uint64_t)li.QuadPart;
        HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!m) {
            CloseHandle(h);
            throw std::runtime_error("CreateFileMapping failed");
        }
        base_ = (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        if (!base_) {
            CloseHandle(m);
            CloseHandle(h);
            throw std::runtime_error("MapViewOfFile failed");
        }
        map_ = m;
        file_ = h;
#else
        int fd = ::open(path_.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path_);
        fd_ = fd;
        struct stat st{};
        if (fstat(fd, &st) != 0) throw std::runtime_error("fstat failed");
        size_ = (uint64_t)st.st_size;
        void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        base_ = (const uint8_t*)p;
#endif
        parse();
    }
    void close() {
#ifdef _WIN32
        if (base_) UnmapViewOfFile(base_);
        if (map_) CloseHandle((HANDLE)map_);
        if (file_) CloseHandle((HANDLE)file_);
#else
        if (base_) munmap((void*)base_, size_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }

    void parse() {
        Cursor c(base_, size_);
        const uint32_t magic = c.read<uint32_t>();
        if (magic != 0x46554747u) throw std::runtime_error("not a GGUF file (bad magic)");
        version_ = c.read<uint32_t>();
        if (version_ != 3)
            throw std::runtime_error("GGUF v" + std::to_string(version_) + ", this reader handles v3");
        const uint64_t n_tensors = c.read<uint64_t>();
        const uint64_t n_kv = c.read<uint64_t>();

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = c.str();
            MetaType t = (MetaType)c.read<uint32_t>();
            meta_.emplace(std::move(key), read_value(c, t));
        }
        tensors_.reserve((size_t)n_tensors);
        // GGUF has no index to arbitrate between two tensors of one name: find() is first-match, so a
        // duplicate would silently win by position.  Refuse the file at open instead, naming both.
        std::set<std::string> names;
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            t.name = c.str();
            if (!names.insert(t.name).second)
                throw std::runtime_error("GGUF: duplicate tensor name '" + t.name + "' in " + path_);
            const uint32_t nd = c.read<uint32_t>();
            if (nd == 0 || nd > 4) throw std::runtime_error("GGUF: bad n_dims for " + t.name);
            t.shape.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) t.shape[d] = c.read<uint64_t>();
            t.type = c.read<uint32_t>();
            t.offset = c.read<uint64_t>();
            tensors_.push_back(std::move(t));
        }
        uint64_t align = 32;
        if (const MetaValue* a = get("general.alignment"))
            if (a->u) align = a->u;
        alignment_ = align;
        data_start_ = (c.pos() + align - 1) / align * align;
        if (data_start_ > size_) throw std::runtime_error("GGUF: data section starts past EOF");
    }

    std::string path_;
    const uint8_t* base_ = nullptr;
    uint64_t size_ = 0, data_start_ = 0, alignment_ = 32;
    uint32_t version_ = 0;
#ifdef _WIN32
    void* file_ = nullptr;
    void* map_ = nullptr;
#else
    int fd_ = -1;
#endif
    std::vector<TensorInfo> tensors_;
    std::map<std::string, MetaValue> meta_;
};

// ---- architecture guard (P1.S2). The engine is specialised to ONE model; anything else must be
// refused with a precise error rather than silently mis-run.
struct Qwen4ExpGuard {
    uint32_t block_count = 48, hidden = 2560, experts = 0, experts_used = 0, head_count = 24,
             head_count_kv = 2;   // 0 = presence-only: pruned variants (GSQ-RCO Coder) legitimately ship
                                  // fewer experts than the canonical 512; the graph reads the true value
};

inline std::string check_architecture(const GgufFile& g, const Qwen4ExpGuard& want = {}) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "qwen4exp") return "architecture is '" + arch->s + "', this engine requires 'qwen4exp'";
    struct Req {
        const char* key;
        uint64_t want;
    };
    const Req reqs[] = {
        {"qwen4exp.block_count", want.block_count},
        {"qwen4exp.embedding_length", want.hidden},
        {"qwen4exp.expert_count", want.experts},
        {"qwen4exp.expert_used_count", want.experts_used},
        {"qwen4exp.attention.head_count", want.head_count},
        {"qwen4exp.attention.head_count_kv", want.head_count_kv},
    };
    for (const auto& r : reqs) {
        const MetaValue* v = g.get(r.key);
        if (!v) return std::string("missing ") + r.key;
        if (r.want && v->u != r.want)
            return std::string(r.key) + " = " + std::to_string(v->u) + ", expected " + std::to_string(r.want);
    }
    return {}; // empty == ok
}

} // namespace strata

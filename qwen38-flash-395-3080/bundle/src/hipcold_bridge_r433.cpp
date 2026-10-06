// hipcold_bridge.cpp —— 8060S 冷专家桥接实现（dlopen + worker 线程）
#include "strata/prefill/hipcold_bridge.hpp"

#include "strata/core/expert_source.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace strata::prefill {

HipCold& hipcold() {
  static HipCold hc;
  return hc;
}

namespace {
constexpr int64_t kH = 2560;
constexpr size_t kX80Row = 40 * 80;  // 一行激活的 80B 块字节数
// r431: 库的 per-call 行上限是 8192（hipcold_prep_xq80 / hipcold_fetch_out）。引擎的 prefill chunk
// (prompt chunk auto) 可以到 23040 tokens，所以激活必须按 8192 行切片走库：每片 prep_xq80 ->
// 逐专家 sparse（token 下标改成片内相对）-> fetch_out 落到 out 的对应行。数值与一次性 8192 行一致。
constexpr int64_t kRowsPerSlice = 8192;

// r431: 探测失败时也把原因打到引擎 stderr（原来只记在 why 里，线上看不出为什么静默回退）。
void say_why(const char* what) {
  std::fprintf(stderr, "strata hipcold: %s（回退原 PCIe 路径）\n", what);
  std::fflush(stderr);
}
// r434d：worker 分段埋点（引擎 stderr 现已落盘，排 issue#29 式挂死用；STRATA_HIPCOLD_DEBUG=0 关）
void dbg(const char* what) {
  static const bool on =
      !std::getenv("STRATA_HIPCOLD_DEBUG") || std::atoi(std::getenv("STRATA_HIPCOLD_DEBUG")) != 0;
  if (!on) return;
  std::fprintf(stderr, "strata hipcold dbg: %s\n", what);
  std::fflush(stderr);
}
// r434j：worker 分段计时
double dbg_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void dbg_t(const char* what, double ms) {
  static const bool on =
      !std::getenv("STRATA_HIPCOLD_DEBUG") || std::atoi(std::getenv("STRATA_HIPCOLD_DEBUG")) != 0;
  if (!on) return;
  std::fprintf(stderr, "strata hipcold dbg: %s %.1f ms\n", what, ms);
  std::fflush(stderr);
}
}  // namespace

void HipCold::probe(core::ExpertSource* src) {
  probed = true;
  if (std::getenv("STRATA_HIPCOLD") && std::atoi(std::getenv("STRATA_HIPCOLD")) == 0) {
    why = "STRATA_HIPCOLD=0";
    return;
  }
  const char* path = std::getenv("STRATA_HIPCOLD_LIB");
  if (!path) path = "/home/hfy/strata/r429_hipcold/libstrata_hipcold.so";
  so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!so) {
    why = std::string("dlopen: ") + dlerror();
    say_why(why.c_str());
    return;
  }
  fn_init = (decltype(fn_init)) dlsym(so, "hipcold_init");
  fn_err = (decltype(fn_err)) dlsym(so, "hipcold_last_error");
  fn_map_table = (decltype(fn_map_table)) dlsym(so, "hipcold_map_arena_table");
  fn_set_table = (decltype(fn_set_table)) dlsym(so, "hipcold_set_arena_table");
  fn_map_blob = (decltype(fn_map_blob)) dlsym(so, "hipcold_map_expert_blob");
  fn_unmap_blob = (decltype(fn_unmap_blob)) dlsym(so, "hipcold_unmap_expert_blob");
  fn_prep = (decltype(fn_prep)) dlsym(so, "hipcold_prep_xq80");
  fn_sparse = (decltype(fn_sparse)) dlsym(so, "hipcold_prefill_expert_sparse");
  fn_fetch = (decltype(fn_fetch)) dlsym(so, "hipcold_fetch_out");
  if (!fn_init || !fn_err || !fn_set_table || !fn_map_blob || !fn_unmap_blob || !fn_prep || !fn_sparse || !fn_fetch) {
    why = "dlsym 不全";
    say_why(why.c_str());
    return;
  }
  if (fn_init(-1) != 0) {
    why = std::string("hipcold_init: ") + fn_err();
    say_why(why.c_str());
    return;
  }
  size_t bytes = 0;
  const void* base = src ? src->hipcold_arena(&bytes, &offsets, &offsets_count) : nullptr;
  arena_base = (const uint8_t*) base;
  arena_bytes = bytes;
  if (!base || bytes == 0 || !offsets || offsets_count <= 0) {
    why = "引擎无可用页锁补集（hipcold_arena 空）";
    say_why(why.c_str());
    return;
  }
  // r432: publish metadata only; map each selected expert blob on demand.
  const int rc = fn_set_table((void*) base, bytes, offsets, offsets_count);
  if (rc != 0) {
    why = std::string("map_arena_table: ") + fn_err();
    say_why(why.c_str());
    return;
  }
  if (cudaEventCreateWithFlags(&ev_x, cudaEventDisableTiming) != cudaSuccess) {
    why = "cudaEventCreate 失败";
    say_why(why.c_str());
    return;
  }
  enabled = true;
  std::fprintf(stderr, "strata hipcold: 8060S 冷专家路径已启用（arena %.1f GiB，偏移表 %lld 项）\n",
               bytes / 1073741824.0, (long long) offsets_count);
}

bool HipCold::begin(int64_t layer, int64_t T_, int64_t K_, const int32_t* ids_h, const int32_t* slot_h,
                    const void* Xq, const float* w_dev, cudaStream_t cs) {
  T = T_;
  K = K_;
  active = false;
  err.clear();
  if (!enabled || experts.empty() || T <= 0 || K <= 0) return true;
  // Dm 置零行号 + 每专家 (token, 权重下标) 列表（遍历 ids_h 一次）
  rows_idx.clear();
  std::vector<int> slot_of(512, -1);
  for (size_t i = 0; i < experts.size(); ++i) slot_of[(size_t) experts[i]] = (int) i;
  std::vector<std::vector<int32_t>> toks(experts.size());
  std::vector<std::vector<int32_t>> widx(experts.size());
  for (int64_t i = 0; i < T * K; ++i) {
    const int s = slot_of[(size_t) ids_h[i]];
    if (s < 0) continue;
    rows_idx.push_back(slot_h[i]);
    toks[(size_t) s].push_back((int32_t) (i / K));
    widx[(size_t) s].push_back((int32_t) i);
  }

  // 缓冲与 CUDA 注册（out 要被 3080 的 add kernel 直读，必须 mapped）
  const size_t xb = (size_t) kRowsPerSlice * kX80Row, ob = (size_t) T * kH * sizeof(float);
  if (xq80.size() < xb) xq80.resize(xb);
  if (w_host.size() < (size_t) T * K) w_host.resize((size_t) T * K);
  // r434c：要扩容时必须先反注册旧指针再 resize——旧代码先 resize 后对「新」指针
  // cudaHostUnregister，留下 sticky 的 "pointer does not correspond to a registered
  // memory region"，引擎 mmq quantize 的 cudaGetLastError 捡到后 fatal exit(1)
  //（1K 能过、4K 必死的真正根因，r432/r433/r434 三连复现）。
  if (out_reg && out.size() < (size_t) T * kH) {
    cudaHostUnregister(out.data());
    (void) cudaGetLastError();   // 清残留错误：引擎把 sticky error 当致命
    out_reg = 0;
    out_alias = nullptr;
  }
  if (out.size() < (size_t) T * kH) out.resize((size_t) T * kH);
  if (out_reg != ob) {
    if (out_reg) {
      cudaHostUnregister(out.data());
      (void) cudaGetLastError();
      out_reg = 0;
      out_alias = nullptr;
    }
    if (cudaHostRegister(out.data(), ob, cudaHostRegisterMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &out_alias, out.data(), 0) != cudaSuccess) {
      err = std::string("cudaHostRegister(out ") + std::to_string(ob >> 20) + "MiB) 失败: " +
            cudaGetErrorString(cudaGetLastError());
      return false;
    }
    out_reg = ob;
  }
  if (rows_cap < rows_idx.size()) {
    if (rows_dev) cudaFree(rows_dev);
    rows_cap = rows_idx.size() + 4096;
    if (cudaMalloc((void**) &rows_dev, rows_cap * sizeof(int32_t)) != cudaSuccess) {
      err = std::string("cudaMalloc(rows_idx) 失败: ") + cudaGetErrorString(cudaGetLastError());
      rows_dev = nullptr;
      rows_cap = 0;
      return false;
    }
  }
  if (!rows_idx.empty() &&
      cudaMemcpyAsync(rows_dev, rows_idx.data(), rows_idx.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
          cudaSuccess) {
    err = "rows_idx H2D 失败";
    return false;
  }
  // Xq 就绪事件（quantize_act 已在 cs 上写完 Xq），worker 等它再 D2H
  if (cudaEventRecord(ev_x, cs) != cudaSuccess) { err = "cudaEventRecord 失败"; return false; }

  std::vector<std::vector<int32_t>>* toks_p = new std::vector<std::vector<int32_t>>(std::move(toks));
  std::vector<std::vector<int32_t>>* widx_p = new std::vector<std::vector<int32_t>>(std::move(widx));
  job = std::async(std::launch::async, [this, layer, xq_dev = Xq, w_dev, nT = T, nK = K, toks_p, widx_p]() -> bool {
    std::vector<std::vector<int32_t>> tk(std::move(*toks_p));
    std::vector<std::vector<int32_t>> wi(std::move(*widx_p));
    delete toks_p;
    delete widx_p;
    dbg("worker start");
    double tw0 = dbg_ms();
    if (cudaEventSynchronize(ev_x) != cudaSuccess) { err = "eventSync(Xq) 失败"; return false; }
    dbg_t("ev_x synced", dbg_ms() - tw0);
    // worker 线程内同步 D2H（离开主流；w 0.6MB 级别；Xq 分片后在片循环里拷）
    if (cudaMemcpy(w_host.data(), w_dev, (size_t) nT * nK * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
      err = "w D2H 失败";
      return false;
    }
    dbg("w D2H ok");
    double ms_xq = 0, ms_prep = 0, ms_loop = 0, ms_fetch = 0;
    // r431: 激活按 kRowsPerSlice 行切片过库（库的 per-call 上限 8192 行）
    std::vector<float> wts;
    std::vector<int32_t> rel;
    for (int64_t s0 = 0; s0 < nT; s0 += kRowsPerSlice) {
      const int64_t S = std::min<int64_t>(kRowsPerSlice, nT - s0);
      double ts0 = dbg_ms();
      if (cudaMemcpy(xq80.data(), (const uint8_t*) xq_dev + (size_t) s0 * kX80Row, (size_t) S * kX80Row,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "Xq D2H(slice) 失败";
        return false;
      }
      ms_xq += dbg_ms() - ts0;
      dbg("slice xq D2H ok");
      ts0 = dbg_ms();
      if (fn_prep(xq80.data(), S) != 0) {
        err = std::string("prep_xq80(rows=") + std::to_string((long long) S) + "): " + fn_err();
        return false;
      }
      ms_prep += dbg_ms() - ts0;
      ts0 = dbg_ms();
      for (size_t i = 0; i < experts.size(); ++i) {
        if (tk[i].empty()) continue;
        rel.clear();
        wts.clear();
        for (size_t j = 0; j < tk[i].size(); ++j) {
          const int32_t t = tk[i][j];
          if (t < s0 || t >= s0 + S) continue;   // 不属于本片
          rel.push_back(t - (int32_t) s0);       // 片内相对行号
          wts.push_back(w_host[(size_t) wi[i][j]]);
        }
        if (rel.empty()) continue;
        const size_t oi = (size_t) layer * 512 + (size_t) experts[i];
        if (oi >= (size_t) offsets_count || offsets[oi] == ~(uint64_t) 0 ||
            offsets[oi] + 2764816ull > arena_bytes) {
          err = "cold expert offset invalid";
          return false;
        }
        if (fn_map_blob(layer, experts[i], (void*) (arena_base + offsets[oi])) != 0) {
          err = std::string("map_blob(e=") + std::to_string(experts[i]) + "): " + fn_err();
          return false;
        }
        if (fn_sparse(layer, experts[i], rel.data(), (int) rel.size(), wts.data()) != 0) {
          // r434e：先取错误再 unmap——no-op unmap 会 clear_err 把真实错误洗成 "ok"
          err = std::string("sparse(e=") + std::to_string(experts[i]) + ", ntok=" +
                std::to_string((long long) rel.size()) + "): " + fn_err();
          fn_unmap_blob(layer, experts[i]);
          return false;
        }
        if (fn_unmap_blob(layer, experts[i]) != 0) {
          err = std::string("unmap_blob(e=") + std::to_string(experts[i]) + "): " + fn_err();
          return false;
        }
      }
      ms_loop += dbg_ms() - ts0;
      dbg("slice experts ok");
      ts0 = dbg_ms();
      if (fn_fetch(out.data() + (size_t) s0 * kH, S) != 0) {
        err = std::string("fetch_out(rows=") + std::to_string((long long) S) + "): " + fn_err();
        return false;
      }
      ms_fetch += dbg_ms() - ts0;
    }
    dbg_t("worker xq_total", ms_xq);
    dbg_t("worker prep_total", ms_prep);
    dbg_t("worker loop_total", ms_loop);
    dbg_t("worker fetch_total", ms_fetch);
    dbg("worker done");
    return true;
  });
  active = true;
  return true;
}

bool HipCold::finish(float* bo, cudaStream_t cs, std::string& err_out) {
  if (!active) return true;
  active = false;
  dbg("finish wait");
  const bool ok = job.valid() ? job.get() : false;
  dbg(ok ? "finish got ok" : "finish got FAIL");
  if (!ok) {
    err_out = std::string("hipcold worker: ") + err;
    return false;
  }
  hipcold_add_out(bo, out_alias, T * kH, cs);
  return true;
}

}  // namespace strata::prefill

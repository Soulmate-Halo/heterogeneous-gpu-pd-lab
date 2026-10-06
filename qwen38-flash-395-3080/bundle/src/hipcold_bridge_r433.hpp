// hipcold_bridge.hpp —— 8060S(gfx1151) 冷专家桥接：把「路由选中、VRAM 未命中、但在页锁补集里」
// 的专家从 3080 的 PCIe 流式路径摘除，交给 libstrata_hipcold.so（8060S 零拷贝直读页锁权重），
// 结果在 moe_combine 之后并入 m.bo。任何一步失败：探测期 = 静默回退原路径；运行期 = 上抛错误
// （绝不静默产出错误结果）。
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <future>

#include <cuda_runtime.h>

namespace strata { namespace core { class ExpertSource; } }

namespace strata::prefill {

struct HipCold {
  bool probed = false;
  bool enabled = false;    // 探测通过（dlopen+符号+init+偏移表 arena 映射全过）
  bool active = false;     // 本层已摘除专家（finish/zero_w 据此工作）
  std::string why;         // 禁用原因（stderr 日志一次）

  void* so = nullptr;
  int (*fn_init)(int) = nullptr;
  const char* (*fn_err)(void) = nullptr;
  int (*fn_map_table)(void*, size_t, const uint64_t*, int64_t) = nullptr;
  int (*fn_set_table)(void*, size_t, const uint64_t*, int64_t) = nullptr;
  int (*fn_map_blob)(int64_t, int32_t, void*) = nullptr;
  int (*fn_unmap_blob)(int64_t, int32_t) = nullptr;
  int (*fn_prep)(const uint8_t*, int64_t) = nullptr;
  int (*fn_sparse)(int64_t, int32_t, const int32_t*, int, const float*) = nullptr;
  int (*fn_fetch)(float*, int64_t) = nullptr;

  const uint64_t* offsets = nullptr;   // 引擎 ExpertSource 的页锁补集偏移表（它的内存，只读）
  const uint8_t* arena_base = nullptr;
  size_t arena_bytes = 0;
  int64_t offsets_count = 0;

  // 本层状态（begin 填充，finish 消费；prefill 单线程，无需锁）
  int64_t T = 0, K = 0;
  std::vector<int32_t> experts;     // 摘除的专家 id（升序）
  std::vector<int32_t> rows_idx;    // m.Dm 里这些行号置零（= slot_host[摘除槽位]）
  std::vector<uint8_t> xq80;        // T × 3200（80B 块激活的 D2H 副本）
  std::vector<float> w_host;        // T × K（路由权重 D2H 副本）
  std::vector<float> out;           // T × 2560（8060S 累加输出，CUDA mapped 注册）
  float* out_alias = nullptr;       // out 的 CUDA device alias
  size_t out_reg = 0;
  int32_t* rows_dev = nullptr;
  size_t rows_cap = 0;
  cudaEvent_t ev_x = nullptr;
  std::future<bool> job;
  std::string err;

  bool in_arena(int64_t layer, int32_t e) const {
    if (!offsets) return false;
    const size_t i = (size_t) layer * 512 + (size_t) e;
    return i < (size_t) offsets_count && offsets[i] != ~(uint64_t) 0;
  }

  void probe(core::ExpertSource* src);   // 一次性探测；失败只记 why
  // begin：记录 Xq 就绪事件并起 worker（D2H→prep→逐专家稀疏→fetch）。失败返回 false（调用方上抛）。
  bool begin(int64_t layer, int64_t T_, int64_t K_, const int32_t* ids_h, const int32_t* slot_h,
             const void* Xq, const float* w_dev, cudaStream_t cs);
  // finish：join worker，把 out 并入 bo（add kernel 在 cs 上）。
  bool finish(float* bo, cudaStream_t cs, std::string& err_out);
};

HipCold& hipcold();

void hipcold_zero_rows(float* dm, const int32_t* rows, int n, int64_t N, cudaStream_t s);
void hipcold_add_out(float* bo, const float* o, int64_t n, cudaStream_t s);

}  // namespace strata::prefill

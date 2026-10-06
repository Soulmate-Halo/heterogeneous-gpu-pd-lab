// libstrata_hipcold.so —— 8060S(gfx1151/HIP) 冷专家零拷贝库的接口层
//
// 与 hipcold_api.h 逐字对应。主 CUDA 二进制 dlopen 本库、按名取符号，
// 于是 include/strata/hip_compat/cuda_runtime.h 那种「CUDA/HIP 编译期二选一」
// 的宏映射不会传染到主树：HIP 只活在这一个独立 .so 里。
//
// 本库只操作自己进程内的内存与 8060S，不碰任何别的服务。
// 内存纪律：默认单次分配上限 1GiB，且分配前查 /proc/meminfo 余量自动缩档。

#include "hipcold_api.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdarg>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace {

char g_err[512] = "ok";
int g_inited = 0;

struct Mapping {
  void* host;
  size_t bytes;
};
std::vector<Mapping> g_maps;
struct BlobMapping {
  int64_t layer;
  int32_t expert;
  void* host;
  void* dev;
  int slot = -1;  // r434m：暂存槽下标（sparse 完事后往这个槽记事件）
};
std::vector<BlobMapping> g_blob_maps;

BlobMapping* find_blob_map(int64_t layer, int32_t expert) {
  for (auto& m : g_blob_maps) {
    if (m.layer == layer && m.expert == expert) return &m;
  }
  return nullptr;
}

void set_err(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof(g_err), fmt, ap);
  va_end(ap);
}

void set_err_hip(const char* what, hipError_t e) {
  set_err("%s: %s", what, hipGetErrorString(e));
}

void clear_err() { std::snprintf(g_err, sizeof(g_err), "ok"); }

// r434b：可选跟踪日志（引擎 stderr 被 serve 丢到 /dev/null，排障时写文件）。
void trace(const char* fmt, ...) {
  const char* path = std::getenv("HIPCOLD_TRACE");
  if (!path) return;
  FILE* f = std::fopen(path, "a");
  if (!f) return;
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(f, fmt, ap);
  va_end(ap);
  std::fputc('\n', f);
  std::fclose(f);
}

// r434b：暂存槽架构见下方 kNvBlobBytes 定义之后（常量依赖）。

Mapping* find_map(void* host) {
  for (size_t i = 0; i < g_maps.size(); ++i) {
    if (g_maps[i].host == host) return &g_maps[i];
  }
  return nullptr;
}

// 失败时返回 false，错误信息写入 g_err。
bool get_dev_ptr(void* host, void** out) {
  if (!host) { set_err("host_ptr 为空"); return false; }
  Mapping* m = find_map(host);
  if (!m) { set_err("host_ptr 未注册（先调 hipcold_host_register）"); return false; }
  hipError_t e = hipHostGetDevicePointer(out, host, 0);
  if (e != hipSuccess) { set_err_hip("hipHostGetDevicePointer", e); return false; }
  return true;
}

// 读 /proc/meminfo 的 MemAvailable（字节）。读不到返回 0。
unsigned long long mem_available_bytes() {
  FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  unsigned long long kb = 0;
  while (std::fgets(line, sizeof(line), f)) {
    if (std::strncmp(line, "MemAvailable:", 13) == 0) {
      std::sscanf(line + 13, "%llu", &kb);
      break;
    }
  }
  std::fclose(f);
  return kb * 1024ull;
}

const size_t kDefaultCap = 1ull << 30;  // 单次分配硬上限 1GiB

// 余量不足时把请求量缩档，保住 8095 引擎（占 48GiB 页锁）不被挤爆。
// 留 2GiB 系统余量，不足则退到 512MiB / 256MiB / 64MiB。
size_t clamp_to_budget(size_t want) {
  if (want > kDefaultCap) want = kDefaultCap;
  unsigned long long avail = mem_available_bytes();
  if (avail == 0) return want;
  const unsigned long long reserve = 2ull << 30;
  unsigned long long budget = (avail > reserve) ? (avail - reserve) : 0;
  const size_t cands[4] = {1024ull << 20, 512ull << 20, 256ull << 20, 64ull << 20};
  for (int i = 0; i < 4; ++i) {
    if (want >= cands[i] && (unsigned long long)cands[i] <= budget) return cands[i];
  }
  if ((unsigned long long)want <= budget) return want;
  return 0;
}

double now_s() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

__global__ void add1_kernel(float* p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s) p[i] += 1.0f;
}

__global__ void fill_kernel(float* p, size_t n, float v) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s) p[i] = v;
}

__global__ void read_kernel(const float* __restrict__ p, size_t n, float* sink) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  float acc = 0.0f;
  for (; i < n; i += s) acc += p[i];
  if (acc == -12345.678f) *sink = acc;  // 防优化掉整段读
}

__global__ void pattern_kernel(float* p, size_t n, int magic) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s) p[i] = (float)(((int)(i % 251u) * magic) % 1021) * 0.5f;
}

inline int blocks_for(size_t n, int threads = 256) {
  size_t b = n / threads / 8;
  if (b < 1) b = 1;
  if (b > 65535) b = 65535;
  return (int)b;
}

// ==================== r429：冷专家 NVFP4 反量化 + MMQ（8060S 内核） ====================
//
// 算术与 ggml 的 NVFP4 路径逐项一致（third_party/llama.cpp：convert.cu 的
// dequantize_block_nvfp4 + ggml-impl.h 的 ggml_ue4m3_to_fp32 + ggml-common.h 的
// kvalues_fp4，激活的 vec_dot_type 是 Q8_0）：
//       w      = ue4m3_to_fp32(d_sub) * kvalues_fp4[code]
//       dot    = Σ_sub ue4m3_to_fp32(d_sub) * (Σ_{16 值} kvalues_fp4[code]*q8)
// 括号里是 16 项的整数精确和（|kvalues|<=12、|q8|<=127 -> 最大 24384，int32 里精确），
// 所以与 ggml 的差别只有浮点加法顺序，不是舍入模型。这也是源码树
// src/kernels/cpu/nvfp4_avx512.cpp 用的同一套化简。
constexpr int kNvH = 2560, kNvFF = 640, kNvExpertsPerLayer = 512;
#define HIPCOLD_TB 32  // v3 内核 token 批大小（激活出 LDS 后只受寄存器限制）
constexpr size_t kNvGuRow = 1440, kNvDRow = 360;                       // NVFP4 行字节
constexpr size_t kNvUpOff = (size_t) kNvFF * kNvGuRow;                 // 921600
constexpr size_t kNvDownOff = 2 * kNvUpOff;                            // 1843200
constexpr size_t kNvTailOff = kNvDownOff + (size_t) kNvH * kNvDRow;    // 2764800
constexpr size_t kNvBlobBytes = kNvTailOff + 16;                       // 2764816
constexpr size_t kNvLayerStride = kNvBlobBytes * kNvExpertsPerLayer;   // 1415585792
constexpr int kNvBlkBytes = 36;                                        // block_nvfp4
constexpr int kQ8Blk = 34;                                             // block_q8_0
constexpr int kNvBlocksPerGuRow = (int) (kNvGuRow / kNvBlkBytes);      // 40
constexpr int kNvBlocksPerDRow = (int) (kNvDRow / kNvBlkBytes);        // 10
constexpr int kQ8BlocksPerH = kNvH / 32;                               // 80
constexpr int kQ8BlocksPerFF = kNvFF / 32;                             // 20

// r434b：暂存槽架构。引擎 arena 上的 userptr 注册在少量 blob 时可用（1K 稳定），
// 但 4K 触发的大量注册仍引发 amdgpu gfxhub page fault、引擎 exit 1（r432/r433/r434 均复现）。
// 改为：库自己用 hipHostAlloc 持有一块页锁暂存区（驱动分配的 pinned，非 userptr），
// 任何 blob 用前先 memcpy 进槽位，GPU 只读暂存区；槽位 LRU 复用、进程生命周期内不释放。
// memcpy 2.76MB 约 0.2ms/专家，相对数秒级 prefill 可忽略；跨请求命中还省掉重复拷贝。
constexpr int64_t kStageDefaultSlots = 256;
struct StageSlot {
  int64_t layer = -1;
  int32_t expert = -1;
  uint64_t lru = 0;
  hipEvent_t evt = nullptr;  // r434m：该槽最近一次被 kernel 使用的完成事件
  int state = 0;             // r435：0=就绪，1=后台暂存进行中（内容归 worker，kernel 不得读）
};
// r435：后台暂存线程池。decode 每层窗口的淘汰 memcpy 若留在池线程（map_blob 调用方）上会全部
// 落在 verify 关键路径（实测 sync 修好后 stage_ms=4.8s/7 请求，decode 反而比基线慢 18%）。
// 改法：map_blob 只在锁内选好槽、标 state=1、把 (slot, host_ptr) 推进队列；worker 在线程上等
// 被淘汰槽的事件、做 memcpy、标 state=0 并广播。sparse 发射 kernel 前若目标槽没就绪才等
// （稳态下整个窗口 ~30ms 早已把 <1 个/层的淘汰拷完，实际几乎不阻塞）。
// HIPCOLD_STAGE_THREADS=0 回退旧的同步路径（A/B 用），默认 3。
struct StageReq {
  int64_t slot;
  int64_t layer;
  int32_t expert;
  const void* host_ptr;
};
std::deque<StageReq> g_stage_q;
std::condition_variable g_stage_cv;
std::condition_variable g_stage_done_cv;
int g_stage_workers = 0;
// r434o：暂存区 = 自有 malloc + hipHostRegister（userptr 钉自有页，实测 620MiB=260ms 不卡；
// hipHostAlloc 在内存压力下会随机卡死、hipMalloc 设备内存的 H2D 只有 0.5GiB/s，都已被否掉）。
// GPU 零拷贝直读（224GiB/s 实测量级），blob 用前 CPU memcpy 进槽（~0.2ms），
// 淘汰覆写前只等被淘汰槽自己的事件（r434m）。
void* g_stage_host = nullptr;
void* g_stage_dev = nullptr;
int64_t g_stage_slots = 0;
std::vector<StageSlot> g_stage;
uint64_t g_stage_tick = 0;
std::mutex g_stage_mtx;
int g_pending = 0;  // r434h：已发射未同步的 sparse 批数（swork 复用/fetch 前需要）

// r434j：分段耗时统计（每次 fetch_out 末尾打一行到 trace）
double g_ms_sync = 0, g_ms_stage = 0, g_ms_fetch = 0;
long long g_n_sparse = 0, g_n_evict = 0;
double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint8_t* stage_slot_host(int64_t slot) { return (uint8_t*) g_stage_host + (size_t) slot * kNvBlobBytes; }
uint8_t* stage_slot_dev(int64_t slot) { return (uint8_t*) g_stage_dev + (size_t) slot * kNvBlobBytes; }

// r435：后台暂存 worker。等被淘汰槽上次的 kernel（事件）再覆写——等待发生在线程上，不占池线程。
void stage_worker() {
  for (;;) {
    StageReq r;
    {
      std::unique_lock<std::mutex> lk(g_stage_mtx);
      g_stage_cv.wait(lk, [] { return !g_stage_q.empty(); });
      r = g_stage_q.front();
      g_stage_q.pop_front();
    }
    StageSlot& sl = g_stage[(size_t) r.slot];
    if (sl.evt) {
      const double t0 = now_ms();
      const hipError_t se = hipEventSynchronize(sl.evt);
      {
        std::lock_guard<std::mutex> lk(g_stage_mtx);
        g_ms_sync += now_ms() - t0;
      }
      if (se != hipSuccess) {
        trace("stage worker evict sync failed slot %lld", (long long) r.slot);
        (void) hipGetLastError();
      }
    }
    const double t1 = now_ms();
    std::memcpy(stage_slot_host(r.slot), r.host_ptr, kNvBlobBytes);
    {
      std::lock_guard<std::mutex> lk(g_stage_mtx);
      g_ms_stage += now_ms() - t1;
      sl.state = 0;
      g_stage_done_cv.notify_all();
    }
  }
}

// sparse 发射 kernel 前的槽就绪等待（稳态下几乎不阻塞；见上方 r435 注释）
void stage_wait_ready(int64_t slot) {
  std::unique_lock<std::mutex> lk(g_stage_mtx);
  g_stage_done_cv.wait(lk, [&] { return g_stage[(size_t) slot].state == 0; });
}

int stage_alloc_locked() {
  if (g_stage_host) return 0;
  int64_t want = kStageDefaultSlots;
  if (const char* s = std::getenv("HIPCOLD_SLOTS")) {
    const long v = std::atol(s);
    if (v >= 16 && v <= 65536) want = v;
  }
  for (; want >= 32; want /= 2) {
    trace("stage alloc try %lld slots (malloc+register)", (long long) want);
    void* h = std::malloc((size_t) want * kNvBlobBytes);
    if (!h) continue;
    std::memset(h, 0, (size_t) want * kNvBlobBytes);  // 先驻留，钉住只走 get_user_pages
    hipError_t e = hipHostRegister(h, (size_t) want * kNvBlobBytes, hipHostRegisterMapped);
    if (e != hipSuccess) { (void) hipGetLastError(); std::free(h); continue; }
    void* d = nullptr;
    e = hipHostGetDevicePointer(&d, h, 0);
    if (e != hipSuccess) {
      (void) hipGetLastError();
      hipHostUnregister(h);
      std::free(h);
      continue;
    }
    g_stage_host = h;
    g_stage_dev = d;
    g_stage_slots = want;
    break;
  }
  if (!g_stage_host) { set_err("暂存区分配/注册失败"); return 1; }
  g_stage.assign((size_t) g_stage_slots, StageSlot{});
  for (int64_t i = 0; i < g_stage_slots; ++i) {
    if (hipEventCreateWithFlags(&g_stage[(size_t) i].evt, hipEventDisableTiming) != hipSuccess) {
      (void) hipGetLastError();
      g_stage[(size_t) i].evt = nullptr;  // 建不起来就退化为淘汰全同步
    }
  }
  trace("stage alloc: %lld slots x %zu B = %.1f MiB (malloc+register)", (long long) g_stage_slots,
        (size_t) kNvBlobBytes, g_stage_slots * (double) kNvBlobBytes / 1048576.0);
  // r435：后台暂存线程（HIPCOLD_STAGE_THREADS，默认 3，0=同步回退）
  int nw = 3;
  if (const char* s = std::getenv("HIPCOLD_STAGE_THREADS")) {
    const long v = std::atol(s);
    if (v >= 0 && v <= 16) nw = (int) v;
  }
  for (int i = 0; i < nw; ++i) std::thread(stage_worker).detach();
  g_stage_workers = nw;
  trace("stage workers: %d", nw);
  return 0;
}

// 保证 (layer, expert) 的 blob 已拷进某个暂存槽（或正在后台拷）；返回槽下标，lk 为已持有的 g_stage_mtx
// （unique_lock：极端情况下淘汰选择要等待在途暂存完成，wait 会临时放锁）。
// 淘汰安全：r435 起覆写由后台 worker 执行，worker 覆写前先等被淘汰槽自己的事件；kernel 侧由
// sparse 在发射前等 state==0（stage_wait_ready），两个方向都不会读到半截内容。
int64_t stage_ensure_locked(int64_t layer, int32_t expert, const void* host_ptr,
                            std::unique_lock<std::mutex>& lk) {
  for (;;) {
    for (int64_t i = 0; i < g_stage_slots; ++i) {
      if (g_stage[(size_t) i].layer == layer && g_stage[(size_t) i].expert == expert) {
        g_stage[(size_t) i].lru = ++g_stage_tick;
        return i;   // 命中（含 state==1 的同专家在途暂存：sparse 会等它就绪）
      }
    }
    int64_t victim = -1;
    for (int64_t i = 0; i < g_stage_slots; ++i) {
      if (g_stage[(size_t) i].state != 0) continue;   // 在途槽不参与淘汰
      if (victim < 0 || g_stage[(size_t) i].lru < g_stage[(size_t) victim].lru) victim = i;
    }
    if (victim >= 0) {
      if (g_stage[(size_t) victim].layer >= 0) {
        trace("stage evict L%lld E%d -> L%lld E%d", (long long) g_stage[(size_t) victim].layer,
              g_stage[(size_t) victim].expert, (long long) layer, expert);
      }
      if (g_stage_workers > 0) {
        // r435：入队即返——标记换主 + state=1，覆写与等事件都在 worker 线程上做
        g_stage[(size_t) victim].layer = layer;
        g_stage[(size_t) victim].expert = expert;
        g_stage[(size_t) victim].lru = ++g_stage_tick;
        g_stage[(size_t) victim].state = 1;
        g_stage_q.push_back(StageReq{victim, layer, expert, host_ptr});
        g_stage_cv.notify_one();
        ++g_n_evict;
        return victim;
      }
      // 同步回退（HIPCOLD_STAGE_THREADS=0）：r434m 语义，只等被淘汰槽自己上次的 kernel
      if (g_stage[(size_t) victim].layer >= 0 && g_pending > 0) {
        const double t0 = now_ms();
        hipError_t se;
        if (g_stage[(size_t) victim].evt) {
          se = hipEventSynchronize(g_stage[(size_t) victim].evt);
        } else {
          se = hipDeviceSynchronize();
        }
        g_ms_sync += now_ms() - t0;
        if (se != hipSuccess) { set_err_hip("stage evict sync", se); return -1; }
      }
      ++g_n_evict;
      const double t1 = now_ms();
      std::memcpy(stage_slot_host(victim), host_ptr, kNvBlobBytes);
      g_ms_stage += now_ms() - t1;
      g_stage[(size_t) victim].layer = layer;
      g_stage[(size_t) victim].expert = expert;
      g_stage[(size_t) victim].lru = ++g_stage_tick;
      return victim;
    }
    // 全部槽都在途（队列积压超过槽数才可能出现）：等任意一个就绪再重选
    g_stage_done_cv.wait(lk, [&] {
      for (int64_t i = 0; i < g_stage_slots; ++i)
        if (g_stage[(size_t) i].state == 0) return true;
      return false;
    });
  }
}

// 对外统一入口：保证 (layer, expert) 可用并返回其 BlobMapping（dev 指向暂存槽）；
// 失败返回 nullptr（已 set_err）。被淘汰专家的旧记录同步清掉。
BlobMapping* blob_ensure(int64_t layer, int32_t expert, const void* host_ptr) {
  std::unique_lock<std::mutex> lk(g_stage_mtx);
  if (stage_alloc_locked() != 0) return nullptr;
  BlobMapping* bm = find_blob_map(layer, expert);
  if (bm) return bm;
  const int64_t slot = stage_ensure_locked(layer, expert, host_ptr, lk);
  if (slot < 0) return nullptr;
  void* slot_dev = stage_slot_dev(slot);
  for (size_t i = 0; i < g_blob_maps.size(); ++i) {
    if (g_blob_maps[i].dev == slot_dev) {  // 该槽旧主人的记录作废
      g_blob_maps.erase(g_blob_maps.begin() + (long) i);
      break;
    }
  }
  g_blob_maps.push_back({layer, expert, const_cast<void*>(host_ptr), slot_dev, (int) slot});
  return &g_blob_maps.back();
}

struct ArenaInfo {
  void* base = nullptr;
  size_t bytes = 0;
  void* dev = nullptr;
  int mapped = 0;
  // r430：引擎的页锁区不是稠密布局（它是按 expert profile 挑出来的紧凑补集），所以除了
  // 稠密公式 base + (layer*512+expert)*blob 之外，还允许挂一张调用方给的偏移表：
  // offsets[layer*512+expert] = 该 blob 相对 base 的字节偏移；~(uint64_t)0 = 不在 arena 里。
  // 指针属于调用方，本库只读、不释放（表的生命周期由调用方保证，arena 反注册前必须有效）。
  const uint64_t* offsets = nullptr;
  int64_t offsets_count = 0;
};
ArenaInfo g_arena;

// 一个 (layer, expert) 的 blob 在 arena 里的字节偏移。表存在时查表，否则用稠密公式。
// 返回 0 成功；非 0 = 失败（已 set_err）。
int arena_blob_offset(int64_t layer, int32_t expert, size_t* off_out) {
  if (g_arena.offsets != nullptr) {
    const size_t idx = (size_t) layer * (size_t) kNvExpertsPerLayer + (size_t) expert;
    if ((int64_t) idx >= g_arena.offsets_count) {
      set_err("偏移表太小：layer %lld expert %d 需下标 %zu，表只有 %lld 项", (long long) layer, expert, idx,
              (long long) g_arena.offsets_count);
      return 1;
    }
    const uint64_t o = g_arena.offsets[idx];
    if (o == ~(uint64_t) 0) {
      set_err("layer %lld expert %d 不在 arena 里（偏移表标记为未驻留）", (long long) layer, expert);
      return 1;
    }
    *off_out = (size_t) o;
    return 0;
  }
  *off_out = (size_t) layer * kNvLayerStride + (size_t) expert * kNvBlobBytes;
  return 0;
}

// 常驻工作区（Q8_0 激活 / hidden / ff / 设备输出副本）。它的作用只是给 8060S 一个
// 可直接读写的地方：避免每一层都做一次页锁，也避免 hipHostRegister/Unregister
// 高频往返。首次用到时按需分配 + 注册，之后按大小复用；释放只在 hipcold_unmap_arena。
struct WorkInfo {
  uint8_t* ptr = nullptr;
  size_t bytes = 0;
  void* dev = nullptr;
};
WorkInfo g_work;

// kernel 间中间结果（ff/hidden-q8/累加输出）专用设备显存：mapped host 内存被 GPU 写后再被
// 后续 kernel 读会命中 non-coherent 的陈旧 L2 行（实测 rel_l2 爆炸），所以这三段必须 device-local。
void* g_dff = nullptr;   // 64 token × 640 float
void* g_dhq = nullptr;   // 64 token × 20 × 34 B
void* g_dout = nullptr;  // 64 token × 2560 float（各专家路由权重累加）
void* g_dxqu = nullptr;  // v3 预处理激活：64 × 2560 u8（q+128）
void* g_dxqs = nullptr;  // 64 × 160 int 子块偏转和
void* g_dxqd = nullptr;  // 64 × 80 float q8 scale（prep 统一转 fp32）
void* g_dhqs = nullptr;  // 64 × 40 int（down 激活子块和）
void* g_dhqd = nullptr;  // 64 × 20 float

int ensure_device_bufs() {
  if (g_dff) return 0;
  hipError_t e = hipMalloc(&g_dff, (size_t) 64 * kNvFF * sizeof(float));
  if (e != hipSuccess) { set_err_hip("hipMalloc(ff)", e); return -(int)e; }
  e = hipMalloc(&g_dhq, (size_t) 64 * kQ8BlocksPerFF * kQ8Blk);
  if (e != hipSuccess) { set_err_hip("hipMalloc(hq)", e); return -(int)e; }
  e = hipMalloc(&g_dout, (size_t) 64 * kNvH * sizeof(float));
  if (e != hipSuccess) { set_err_hip("hipMalloc(out)", e); return -(int)e; }
  e = hipMalloc(&g_dxqu, (size_t) 64 * kNvH);
  if (e != hipSuccess) { set_err_hip("hipMalloc(xqu)", e); return -(int)e; }
  e = hipMalloc(&g_dxqs, (size_t) 64 * 160 * sizeof(int));
  if (e != hipSuccess) { set_err_hip("hipMalloc(xqs)", e); return -(int)e; }
  e = hipMalloc(&g_dxqd, (size_t) 64 * 80 * sizeof(float));
  if (e != hipSuccess) { set_err_hip("hipMalloc(xqd)", e); return -(int)e; }
  e = hipMalloc(&g_dhqs, (size_t) 64 * 40 * sizeof(int));
  if (e != hipSuccess) { set_err_hip("hipMalloc(hqs)", e); return -(int)e; }
  e = hipMalloc(&g_dhqd, (size_t) 64 * 20 * sizeof(float));
  if (e != hipSuccess) { set_err_hip("hipMalloc(hqd)", e); return -(int)e; }
  return 0;
}

// ---- sparse（引擎接入）状态：全 T 行激活 + 单专家 token 列表 ----
int64_t g_xrows = 0;             // 最近 hipcold_prep_xq80 登记的行数
bool g_big_bufs = false;         // 8192 行缓冲是否已分配
void* g_dout_big = nullptr;  // 8192 × 2560 fp32：sparse 累加输出（HIP 私域显存）

struct SparseWork {
  uint8_t* host = nullptr;       // tidx/weights 的 mapped 工作区（64 槽轮转）
  size_t bytes = 0;
  void* dev = nullptr;
};
SparseWork g_swork;
constexpr int kWorkSlots = 64;   // r434h：sparse token/权重工作槽数，满 64 槽才同步一次
int g_swork_idx = 0;
hipEvent_t g_swork_evts[64];     // r434m：每槽完成事件（复用前只等自己）
bool g_swork_evts_ok = false;

int ensure_big_bufs() {
  if (g_big_bufs) return 0;
  const int64_t R = 8192;
  hipError_t e = hipMalloc(&g_dxqu, (size_t) R * kNvH);
  if (e != hipSuccess) { set_err_hip("hipMalloc(xqu big)", e); return -(int)e; }
  e = hipMalloc(&g_dxqs, (size_t) R * 160 * sizeof(int));
  if (e != hipSuccess) { set_err_hip("hipMalloc(xqs big)", e); return -(int)e; }
  e = hipMalloc(&g_dxqd, (size_t) R * 80 * sizeof(float));
  if (e != hipSuccess) { set_err_hip("hipMalloc(xqd big)", e); return -(int)e; }
  e = hipMalloc(&g_dff, (size_t) R * kNvFF * sizeof(float));
  if (e != hipSuccess) { set_err_hip("hipMalloc(ff big)", e); return -(int)e; }
  e = hipMalloc(&g_dhq, (size_t) R * kNvFF);
  if (e != hipSuccess) { set_err_hip("hipMalloc(hqu big)", e); return -(int)e; }
  e = hipMalloc(&g_dhqs, (size_t) R * 40 * sizeof(int));
  if (e != hipSuccess) { set_err_hip("hipMalloc(hqs big)", e); return -(int)e; }
  e = hipMalloc(&g_dhqd, (size_t) R * 20 * sizeof(float));
  if (e != hipSuccess) { set_err_hip("hipMalloc(hqd big)", e); return -(int)e; }
  e = hipMalloc(&g_dout_big, (size_t) R * kNvH * sizeof(float));
  if (e != hipSuccess) { set_err_hip("hipMalloc(dout big)", e); return -(int)e; }
  if (!g_swork.host) {
    const size_t bytes = (size_t) kWorkSlots * (size_t) R * 2 * sizeof(int32_t);  // 64 槽 × (tidx + weights)
    void* p = std::malloc(bytes);
    if (!p) { set_err("malloc(swork) 失败"); return 1; }
    const int rc = hipcold_host_register(p, bytes);
    if (rc != 0) { std::free(p); return rc; }
    void* pd = nullptr;
    if (!get_dev_ptr(p, &pd)) { hipcold_host_unregister(p); std::free(p); return 1; }
    g_swork.host = (uint8_t*) p;
    g_swork.bytes = bytes;
    g_swork.dev = pd;
  }
  if (!g_swork_evts_ok) {
    g_swork_evts_ok = true;
    for (int i = 0; i < kWorkSlots; ++i) {
      if (hipEventCreateWithFlags(&g_swork_evts[i], hipEventDisableTiming) != hipSuccess) {
        (void) hipGetLastError();
        g_swork_evts_ok = false;  // 建不齐就退回满圈全同步
        break;
      }
    }
  }
  g_big_bufs = true;
  return 0;
}

void free_device_bufs() {
  if (g_dff) { hipFree(g_dff); g_dff = nullptr; }
  if (g_dhq) { hipFree(g_dhq); g_dhq = nullptr; }
  if (g_dout) { hipFree(g_dout); g_dout = nullptr; }
  if (g_dxqu) { hipFree(g_dxqu); g_dxqu = nullptr; }
  if (g_dxqs) { hipFree(g_dxqs); g_dxqs = nullptr; }
  if (g_dxqd) { hipFree(g_dxqd); g_dxqd = nullptr; }
  if (g_dhqs) { hipFree(g_dhqs); g_dhqs = nullptr; }
  if (g_dhqd) { hipFree(g_dhqd); g_dhqd = nullptr; }
  if (g_dout_big) { hipFree(g_dout_big); g_dout_big = nullptr; }
  if (g_swork.host) { hipcold_host_unregister(g_swork.host); std::free(g_swork.host);
                      g_swork.host = nullptr; g_swork.bytes = 0; g_swork.dev = nullptr; }
  g_big_bufs = false;
  g_xrows = 0;
}

// fp32 -> fp16（round-to-nearest-even，与 F16C / GGML_FP32_TO_FP16 同舍入）。
uint16_t f32_to_f16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint16_t sign = (uint16_t) ((u >> 16) & 0x8000u);
  const uint32_t xe = (u >> 23) & 0xFFu;
  uint32_t man = u & 0x7FFFFFu;
  if (xe == 0xFFu) return (uint16_t) (sign | 0x7C00u | (man ? 0x200u : 0u));
  int exp = (int) xe - 127 + 15;
  if (exp >= 31) return (uint16_t) (sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10) return sign;
    man |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t m = man >> shift;
    const uint32_t rem = man & ((1u << shift) - 1u);
    const uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (m & 1u))) ++m;
    return (uint16_t) (sign | (uint16_t) m);
  }
  uint32_t m = man >> 13;
  const uint32_t rem = man & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
    ++m;
    if (m == 0x400u) {
      m = 0;
      if (++exp >= 31) return (uint16_t) (sign | 0x7C00u);
    }
  }
  return (uint16_t) (sign | ((uint32_t) exp << 10) | m);
}

// 与 ggml 的 quantize_row_q8_0 同算术（amax/127、roundf 远离零）。
void quantize_q8_0_host(const float* x, uint8_t* dst, int n) {
  const int nb = n / 32;
  for (int i = 0; i < nb; ++i) {
    const float* xb = x + (size_t) i * 32;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
      const float a = std::fabs(xb[j]);
      if (amax < a) amax = a;
    }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint16_t dh = f32_to_f16(d);
    std::memcpy(dst + (size_t) i * kQ8Blk, &dh, 2);
    for (int j = 0; j < 32; ++j) {
      int v = (int) std::lround(xb[j] * id);
      if (v > 127) v = 127;
      if (v < -128) v = -128;
      dst[(size_t) i * kQ8Blk + 2 + j] = (uint8_t) (int8_t) v;
    }
  }
}

// E2M1 的 doubling 整数值（ggml-common.h 的 kvalues_fp4，MXFP4/NVFP4 共用）。
__device__ __constant__ int kAmpDev[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
// v2 专用 biased 版（kAmp+12），配合 u8×u8 dot4 的双偏转数学。
__device__ __constant__ int kAmpUDev[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};

__device__ __forceinline__ float h2f_dev(uint16_t h) {
  const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
  uint32_t u;
  if (e == 0) {
    if (m == 0) {
      u = sign;
    } else {
      uint32_t ex = 127 - 15 + 1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --ex;
      }
      u = sign | (ex << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (e == 31) {
    u = sign | 0x7F800000u | (m << 13);
  } else {
    u = sign | ((e + 127 - 15) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// GGML 的 ggml_ue4m3_to_fp32：0 与 0x7F 读作 0，指数偏置 7，结果再乘 0.5。
__device__ __forceinline__ float ue4m3_dev(uint8_t x) {
  if (x == 0 || x == 0x7F) return 0.0f;
  const int e = (x >> 3) & 0xF, m = x & 0x7;
  const float raw = (e == 0) ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
  return raw * 0.5f;
}

// 一个 64 值 NVFP4 block 与两个 32 值 Q8_0 block 的点积。
__device__ __forceinline__ float nvfp4_block_dot(const uint8_t* __restrict__ wb, const uint8_t* __restrict__ qb) {
  const uint8_t* qs = wb + 4;
  const int8_t* qa = (const int8_t*) (qb + 2);          // 元素 0..31
  const int8_t* qc = (const int8_t*) (qb + kQ8Blk + 2); // 元素 32..63
  uint16_t dh0, dh1;
  __builtin_memcpy(&dh0, qb, 2);
  __builtin_memcpy(&dh1, qb + kQ8Blk, 2);
  const float d0 = h2f_dev(dh0), d1 = h2f_dev(dh1);
  int acc[4] = {0, 0, 0, 0};
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int8_t* qq = (s < 2) ? (qa + (s & 1) * 16) : (qc + ((s - 2) & 1) * 16);
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const uint8_t byte = qs[s * 8 + b];
      acc[s] += kAmpDev[byte & 0xF] * (int) qq[b] + kAmpDev[byte >> 4] * (int) qq[b + 8];
    }
  }
  return (float) acc[0] * ue4m3_dev(wb[0]) * d0 + (float) acc[1] * ue4m3_dev(wb[1]) * d0 +
         (float) acc[2] * ue4m3_dev(wb[2]) * d1 + (float) acc[3] * ue4m3_dev(wb[3]) * d1;
}

__device__ __forceinline__ float warp_sum_dev(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, o);
  return v;
}

// 一个 warp 负责一个 (行, token)：lane 按 64 值 block 分片，warp 内归约。
// gate/up 同行同时算：ff[t][r] = silu(s_gate*g) * (s_up*u)。
__global__ void nvfp4_gu_kernel(const uint8_t* __restrict__ blob, const uint8_t* __restrict__ xq,
                                int q8_row_bytes, int tokens, float s_gate, float s_up, float* __restrict__ ff) {
  const int lane = threadIdx.x & 31;
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int row = wid % kNvFF;
  const int tok = wid / kNvFF;
  if (tok >= tokens) return;
  const uint8_t* grow = blob + (size_t) row * kNvGuRow;
  const uint8_t* urow = blob + kNvUpOff + (size_t) row * kNvGuRow;
  const uint8_t* qrow = xq + (size_t) tok * q8_row_bytes;
  float g = 0.0f, u = 0.0f;
  for (int b = lane; b < kNvBlocksPerGuRow; b += 32) {
    const uint8_t* qblk = qrow + (size_t) (2 * b) * kQ8Blk;
    g += nvfp4_block_dot(grow + (size_t) b * kNvBlkBytes, qblk);
    u += nvfp4_block_dot(urow + (size_t) b * kNvBlkBytes, qblk);
  }
  g = warp_sum_dev(g);
  u = warp_sum_dev(u);
  if (lane == 0) {
    const float G = g * s_gate, U = u * s_up;
    ff[(size_t) tok * kNvFF + row] = (G / (1.0f + expf(-G))) * U;
  }
}

// 一个 warp 负责一个 (行, token)：out[t][r] = dot(row r, act t) * scale。
__global__ void nvfp4_row_dot_kernel(const uint8_t* __restrict__ w, int rows, size_t row_bytes, int nblocks,
                                     const uint8_t* __restrict__ xq, int q8_row_bytes, int tokens, float scale,
                                     float* __restrict__ out) {
  const int lane = threadIdx.x & 31;
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int row = wid % rows;
  const int tok = wid / rows;
  if (tok >= tokens) return;
  const uint8_t* wrow = w + (size_t) row * row_bytes;
  const uint8_t* qrow = xq + (size_t) tok * q8_row_bytes;
  float acc = 0.0f;
  for (int b = lane; b < nblocks; b += 32) {
    acc += nvfp4_block_dot(wrow + (size_t) b * kNvBlkBytes, qrow + (size_t) (2 * b) * kQ8Blk);
  }
  acc = warp_sum_dev(acc);
  if (lane == 0) out[(size_t) tok * rows + row] = acc * scale;
}

inline int warps_to_blocks(size_t warps) {
  const size_t threads = warps * 32;
  size_t b = (threads + 255) / 256;
  if (b < 1) b = 1;
  if (b > 2147483647u) b = 2147483647u;
  return (int) b;
}

// ======================= v2 内核：sdot4 + LDS LUT + 权重寄存器驻留复用 =======================
// 数值路径与 v1 完全一致（每 16 值子块整数精确点积 × ue4m3 × q8 scale 后 fp 累加），
// 只改并行结构：LDS 查 LUT 消除 __constant__ 串行、v_dot4_i32_i8 取代标量 MAD、
// 权重展开一次寄存器驻留、8 token 一批复用、激活重排进 LDS 对齐 int4。

// gfx1151 实测只有 u8×u8 的 dot4（i32_i8 / i32_iu8 助记符均被编成同一条 u8 指令，
// dbg_dot 探针以确定向量两次验证）。因此走双偏转：
//   Σw·q = dot_u8(w+12, q+128) − 128·Σ(w+12) − 12·Σ(q+128) + 24576   （每 16 值子块）
__device__ __forceinline__ int sdot4_dev(int a, int b, int c) {
  int d;
  asm volatile("v_dot4_u32_u8 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
  return d;
}

__device__ __forceinline__ int pack4i8_dev(int a, int b, int c, int d) {
  return __byte_perm(__byte_perm(a, b, 0x0040), __byte_perm(c, d, 0x0040), 0x5410);
}

// 8B qs（16 个 4bit E2M1）经 LDS LUT 展开成 4×int32（16 个 int8，与激活值序一致：
// val[j]=lo(byte_j) (j<8)，val[8+j]=hi(byte_j)）。
__device__ __forceinline__ void unpack16_dev(const uint8_t* __restrict__ p, const int* __restrict__ lut,
                                             int pk[4], int& sum_w) {
  int b[8], v[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) b[i] = p[i];
#pragma unroll
  for (int i = 0; i < 8; ++i) { v[i] = lut[b[i] & 15]; v[8 + i] = lut[b[i] >> 4]; }
  pk[0] = pack4i8_dev(v[0], v[1], v[2], v[3]);
  pk[1] = pack4i8_dev(v[4], v[5], v[6], v[7]);
  pk[2] = pack4i8_dev(v[8], v[9], v[10], v[11]);
  pk[3] = pack4i8_dev(v[12], v[13], v[14], v[15]);
  sum_w = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) sum_w += v[i];
}

// 设备端 q8_0 量化（v3 版）：与 quantize_q8_0_host 同算术，但输出分体三数组——
// biased u8 激活（q+128）、每 16 值子块偏转和、fp16 scale，供 u8×u8 dot4 双偏转。
__global__ void nvfp4_quant_kernel(const float* __restrict__ ff, uint8_t* __restrict__ hqu,
                                   int* __restrict__ hqs, float* __restrict__ hqd, int tokens) {
  const int lane = threadIdx.x & 31;
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (wid >= tokens * kQ8BlocksPerFF) return;
  const int t = wid / kQ8BlocksPerFF, blk = wid % kQ8BlocksPerFF;
  const float v = ff[(size_t) t * kNvFF + blk * 32 + lane];
  float a = fabsf(v);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, a, o));
  a = __shfl_sync(0xFFFFFFFFFFFFFFFFull, a, 0);
  const float d = a / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  if (lane == 0) hqd[(size_t) t * kQ8BlocksPerFF + blk] = d;
  int q = (int) lroundf(v * id);
  q = q > 127 ? 127 : (q < -128 ? -128 : q);
  const int u = q + 128;
  hqu[(size_t) t * kNvFF + blk * 32 + lane] = (uint8_t) u;
  // 子块和：lane 0..15 是子块 0，16..31 是子块 1
  int part = u;
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) part += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, part, o);
  if (lane == 0) hqs[(size_t) t * 40 + blk * 2] = part;
  if (lane == 16) hqs[(size_t) t * 40 + blk * 2 + 1] = part;
}

// 80B 块激活预处理（引擎 quantize_act_native 格式：64 int8 + 尾部 float4{a0/127, a1/127,0,0}）。
// 每 warp 一块：xqu 偏转 u8、xqs 四个 16 值子块和、xqd 两个 fp32 scale（b*2+{0,1}）。
__global__ void nvfp4_prep_x80_kernel(const uint8_t* __restrict__ xq, uint8_t* __restrict__ xqu,
                                      int* __restrict__ xqs, float* __restrict__ xqd, int64_t rows) {
  const int lane = threadIdx.x & 31;
  const int64_t wid = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (wid >= rows * 40) return;
  const int64_t t = wid / 40;
  const int b = (int) (wid % 40);
  const uint8_t* src = xq + (size_t) t * (40 * 80) + (size_t) b * 80;
  const int u0 = (int) (int8_t) src[lane] + 128;
  const int u1 = (int) (int8_t) src[32 + lane] + 128;
  xqu[(size_t) t * kNvH + b * 64 + lane] = (uint8_t) u0;
  xqu[(size_t) t * kNvH + b * 64 + 32 + lane] = (uint8_t) u1;
  if (lane < 2) xqd[(size_t) t * 80 + b * 2 + lane] = ((const float*) (src + 64))[lane];
  int s0 = (lane < 16) ? u0 : 0, s1 = (lane < 16) ? 0 : u0;
  int s2 = (lane < 16) ? u1 : 0, s3 = (lane < 16) ? 0 : u1;
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) {
    s0 += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, s0, o);
    s1 += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, s1, o);
    s2 += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, s2, o);
    s3 += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, s3, o);
  }
  if (lane == 0) { xqs[(size_t) t * 160 + b * 4] = s0; xqs[(size_t) t * 160 + b * 4 + 2] = s2; }
  if (lane == 16) { xqs[(size_t) t * 160 + b * 4 + 1] = s1; xqs[(size_t) t * 160 + b * 4 + 3] = s3; }
}

// x 激活预处理（v3）：host 量化的 q8_0（34B 块）转成分体 biased 形式，一次/调用。
__global__ void nvfp4_prep_x_kernel(const uint8_t* __restrict__ xq, uint8_t* __restrict__ xqu,
                                    int* __restrict__ xqs, float* __restrict__ xqd, int tokens) {
  const int lane = threadIdx.x & 31;
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (wid >= tokens * kQ8BlocksPerH) return;
  const int t = wid / kQ8BlocksPerH, blk = wid % kQ8BlocksPerH;
  const uint8_t* src = xq + ((size_t) t * kQ8BlocksPerH + blk) * kQ8Blk;
  if (lane == 0) xqd[(size_t) t * kQ8BlocksPerH + blk] = h2f_dev(*(const uint16_t*) src);
  const int u = (int) (int8_t) src[2 + lane] + 128;
  xqu[(size_t) t * kNvH + blk * 32 + lane] = (uint8_t) u;
  int part = u;
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) part += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, part, o);
  if (lane == 0) xqs[(size_t) t * 160 + blk * 2] = part;
  if (lane == 16) xqs[(size_t) t * 160 + blk * 2 + 1] = part;
}

// gu v2：CTA 256 = 8 warp，每 warp 一行（gate+up），grid.x = kNvFF/8，grid.y = token 批。
// 权重先由 CTA 协作连续 16B 整段搬入 LDS（zero-copy 8B 小读事务效率太差，必须大读），
// 每 lane 再从 LDS 展开 5+5 个子块常驻寄存器，8 token 一批复用。
__global__ void __launch_bounds__(256) nvfp4_gu_kernel_v2(
    const uint8_t* __restrict__ blob, const uint8_t* __restrict__ xqu,
    const int* __restrict__ xqs, const float* __restrict__ xqd,
    int tokens, float s_gate, float s_up, float* __restrict__ ff,
    const int* __restrict__ tidx = nullptr) {
  __shared__ int s_lut[16];
  const int tid = threadIdx.x;
  if (tid < 16) s_lut[tid] = kAmpUDev[tid];
  __syncthreads();
  const int t0 = blockIdx.y * HIPCOLD_TB;
  const int tb = min(HIPCOLD_TB, tokens - t0);
  const int lane = tid & 31, wid = tid >> 5;
  const int row = blockIdx.x * 8 + wid;
  const uint8_t* grow = blob + (size_t) row * kNvGuRow;
  const uint8_t* urow = blob + kNvUpOff + (size_t) row * kNvGuRow;
  int pkg[5][4], pku[5][4], swg[5], swu[5];
  float wsg[5], wsu[5];
#pragma unroll
  for (int k = 0; k < 5; ++k) {
    const int c = lane + 32 * k, b = c >> 2, s = c & 3;
    const uint8_t* bg = grow + b * kNvBlkBytes;
    wsg[k] = ue4m3_dev(bg[s]);
    unpack16_dev(bg + 4 + s * 8, s_lut, pkg[k], swg[k]);
    const uint8_t* bu = urow + b * kNvBlkBytes;
    wsu[k] = ue4m3_dev(bu[s]);
    unpack16_dev(bu + 4 + s * 8, s_lut, pku[k], swu[k]);
  }
  float accg[HIPCOLD_TB] = {0.0f}, accu[HIPCOLD_TB] = {0.0f};
#pragma unroll
  for (int t = 0; t < HIPCOLD_TB; ++t) {
    if (t >= tb) break;
    const int tg = tidx ? tidx[t0 + t] : (t0 + t);
    const uint8_t* qx = xqu + (size_t) tg * kNvH;
    const float* qd = xqd + (size_t) tg * kQ8BlocksPerH;
    const int* qs = xqs + (size_t) tg * 160;
#pragma unroll
    for (int k = 0; k < 5; ++k) {
      const int c = lane + 32 * k, b = c >> 2, s = c & 3;
      const int4 q = *(const int4*) (qx + b * 64 + s * 16);
      const float d = qd[b * 2 + (s < 2 ? 0 : 1)];
      const int corr = 128 * swg[k] + 12 * qs[b * 4 + s] - 24576;
      int ag = sdot4_dev(pkg[k][0], q.x, 0);
      ag = sdot4_dev(pkg[k][1], q.y, ag);
      ag = sdot4_dev(pkg[k][2], q.z, ag);
      ag = sdot4_dev(pkg[k][3], q.w, ag);
      accg[t] += (float) (ag - corr) * (wsg[k] * d);
      const int corru = 128 * swu[k] + 12 * qs[b * 4 + s] - 24576;
      int au = sdot4_dev(pku[k][0], q.x, 0);
      au = sdot4_dev(pku[k][1], q.y, au);
      au = sdot4_dev(pku[k][2], q.z, au);
      au = sdot4_dev(pku[k][3], q.w, au);
      accu[t] += (float) (au - corru) * (wsu[k] * d);
    }
  }
  for (int t = 0; t < tb; ++t) {
    float g = accg[t], u = accu[t];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      g += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, g, o);
      u += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, u, o);
    }
    if (lane == 0) {
      const float G = g * s_gate, U = u * s_up;
      ff[(size_t) (t0 + t) * kNvFF + row] = (G / (1.0f + expf(-G))) * U;
    }
  }
}

// down v2：warp 一行（640 值 = 40 子块，lane 1 子块 + lane<8 加 1），grid.x = kNvH/8。
// 权重同样先整段搬 LDS；按路由权重直接累加写 out（mapped host 零拷贝）。
__global__ void __launch_bounds__(256) nvfp4_row_dot_kernel_v2(
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ hqu,
    const int* __restrict__ hqs, const float* __restrict__ hqd,
    int tokens, float scale, const float* __restrict__ rw, float* __restrict__ out,
    const int* __restrict__ tidx = nullptr) {
  __shared__ int s_lut[16];
  const int tid = threadIdx.x;
  if (tid < 16) s_lut[tid] = kAmpUDev[tid];
  __syncthreads();
  const int t0 = blockIdx.y * HIPCOLD_TB;
  const int tb = min(HIPCOLD_TB, tokens - t0);
  const int lane = tid & 31, wid = tid >> 5;
  const int row = blockIdx.x * 8 + wid;
  const uint8_t* wrow = w + (size_t) row * kNvDRow;
  // 子块 c = lane（+ lane<8 时 c2 = lane+32）：b = c>>2, s = c&3
  int pk0[4], pk1[4], sw0, sw1 = 0;
  float ws0, ws1 = 0.0f;
  {
    const int b = lane >> 2, s = lane & 3;
    const uint8_t* bp = wrow + b * kNvBlkBytes;
    ws0 = ue4m3_dev(bp[s]);
    unpack16_dev(bp + 4 + s * 8, s_lut, pk0, sw0);
    if (lane < 8) {
      const int c2 = lane + 32, b2 = c2 >> 2, s2 = c2 & 3;
      const uint8_t* bp2 = wrow + b2 * kNvBlkBytes;
      ws1 = ue4m3_dev(bp2[s2]);
      unpack16_dev(bp2 + 4 + s2 * 8, s_lut, pk1, sw1);
    }
  }
  float acc[HIPCOLD_TB] = {0.0f};
#pragma unroll
  for (int t = 0; t < HIPCOLD_TB; ++t) {
    if (t >= tb) break;
    const int tg = tidx ? tidx[t0 + t] : (t0 + t);
    const uint8_t* hx = hqu + (size_t) (t0 + t) * kNvFF;   // ff 是局部行，不间接
    const float* hd = hqd + (size_t) (t0 + t) * kQ8BlocksPerFF;
    const int* hs = hqs + (size_t) (t0 + t) * 40;
    (void) tg;
    {
      const int b = lane >> 2, s = lane & 3;
      const int4 q = *(const int4*) (hx + b * 64 + s * 16);
      const float d = hd[b * 2 + (s < 2 ? 0 : 1)];
      const int corr = 128 * sw0 + 12 * hs[b * 4 + s] - 24576;
      int a = sdot4_dev(pk0[0], q.x, 0);
      a = sdot4_dev(pk0[1], q.y, a);
      a = sdot4_dev(pk0[2], q.z, a);
      a = sdot4_dev(pk0[3], q.w, a);
      acc[t] += (float) (a - corr) * (ws0 * d);
    }
    if (lane < 8) {
      const int c2 = lane + 32, b2 = c2 >> 2, s2 = c2 & 3;
      const int4 q = *(const int4*) (hx + b2 * 64 + s2 * 16);
      const float d = hd[b2 * 2 + (s2 < 2 ? 0 : 1)];
      const int corr = 128 * sw1 + 12 * hs[b2 * 4 + s2] - 24576;
      int a = sdot4_dev(pk1[0], q.x, 0);
      a = sdot4_dev(pk1[1], q.y, a);
      a = sdot4_dev(pk1[2], q.z, a);
      a = sdot4_dev(pk1[3], q.w, a);
      acc[t] += (float) (a - corr) * (ws1 * d);
    }
  }
  for (int t = 0; t < tb; ++t) {
    float v = acc[t];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, o);
    if (lane == 0) {
      const int tg = tidx ? tidx[t0 + t] : (t0 + t);
      const float we = rw ? rw[t0 + t] : 1.0f;
      out[(size_t) tg * kNvH + row] += we * v * scale;
    }
  }
}

}  // namespace

extern "C" {

int hipcold_init(int device) {
  if (g_inited) { clear_err(); return 0; }
  hipError_t e = hipInit(0);
  if (e != hipSuccess) { set_err_hip("hipInit", e); return -(int)e; }
  if (device < 0) device = 0;
  e = hipSetDevice(device);
  if (e != hipSuccess) { set_err_hip("hipSetDevice", e); return -(int)e; }
  // r435：decode 场景的同步原语在 CPU 全忙时被 sched_yield 退避饿死（引擎 CPU 池占满核时
  // hipDeviceSynchronize/hipEventSynchronize 实测 ~14.7ms/次，一个请求 28s 全烧在等待上）。
  // HIPCOLD_SYNC_MODE=spin（自旋，唤醒最快但白烧一个核）| block（futex 内核等待，默认）| auto（不动）。
  {
    const char* m = std::getenv("HIPCOLD_SYNC_MODE");
    unsigned flags = hipDeviceScheduleBlockingSync;
    if (m && std::strcmp(m, "spin") == 0) flags = hipDeviceScheduleSpin;
    else if (m && std::strcmp(m, "auto") == 0) flags = 0;
    if (flags) {
      const hipError_t fe = hipSetDeviceFlags(flags);
      if (fe != hipSuccess) set_err_hip("hipSetDeviceFlags", fe);  // 只警告，不判死
      else clear_err();
    }
  }
  g_inited = 1;
  // r434f：暂存区在初始化时就分配好（probe 阶段、无请求看门狗），避免首个请求里
  // hipHostAlloc 数百 MB 卡在内存回收上触发引擎 60s 无进展看门狗（issue#29 式被杀）。
  {
    std::lock_guard<std::mutex> lk(g_stage_mtx);
    if (stage_alloc_locked() != 0) return 1;
  }
  clear_err();
  return 0;
}

int hipcold_device_count(void) {
  if (!g_inited) { set_err("未初始化"); return -1; }
  int n = 0;
  hipError_t e = hipGetDeviceCount(&n);
  if (e != hipSuccess) { set_err_hip("hipGetDeviceCount", e); return -1; }
  clear_err();
  return n;
}

const char* hipcold_last_error(void) { return g_err; }

int hipcold_device_selfcheck(void) {
  if (hipcold_init(-1) != 0) return -1;
  const size_t n = 256u << 10;  // 1MiB of float
  float* d = nullptr;
  hipError_t e = hipMalloc((void**)&d, n * sizeof(float));
  if (e != hipSuccess) { set_err_hip("hipMalloc(1MiB)", e); return -(int)e; }
  fill_kernel<<<blocks_for(n), 256>>>(d, n, 0.0f);
  e = hipDeviceSynchronize();
  if (e != hipSuccess) { set_err_hip("fill_kernel", e); hipFree(d); return -(int)e; }
  add1_kernel<<<blocks_for(n), 256>>>(d, n);
  e = hipDeviceSynchronize();
  if (e != hipSuccess) { set_err_hip("add1_kernel", e); hipFree(d); return -(int)e; }
  std::vector<float> h(n);
  e = hipMemcpy(h.data(), d, n * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(d);
  if (e != hipSuccess) { set_err_hip("hipMemcpy", e); return -(int)e; }
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) if (h[i] != 1.0f) ++bad;
  if (bad) { set_err("设备端自检回校验不一致：%zu/%zu 个元素不为 1.0f", bad, n); return 1; }
  clear_err();
  return 0;
}

int hipcold_host_register(void* host_ptr, size_t bytes) {
  if (hipcold_init(-1) != 0) return -1;
  if (!host_ptr || bytes == 0) { set_err("host_ptr/bytes 非法"); return 1; }
  Mapping* m = find_map(host_ptr);
  if (m) {
    if (m->bytes >= bytes) { clear_err(); return 0; }  // 幂等
    // r434g：同一指针申请更大注册（xq80 行数随 prompt 变长）——先撤销再按新长度注册；
    // 旧代码直接幂等返回，GPU 读超出旧注册长度即 illegal memory access（1K 过、4K 必炸）。
    // 安全：调用点都在上一 slice fetch 的 hipDeviceSynchronize 之后，没有在飞读。
    hipError_t ue = hipHostUnregister(host_ptr);
    if (ue != hipSuccess) { set_err_hip("hipHostUnregister(扩容重注册)", ue); return -(int) ue; }
    for (size_t i = 0; i < g_maps.size(); ++i) {
      if (g_maps[i].host == host_ptr) { g_maps.erase(g_maps.begin() + (long) i); break; }
    }
  }
  hipError_t e = hipHostRegister(host_ptr, bytes, hipHostRegisterMapped);
  if (e != hipSuccess) { set_err_hip("hipHostRegister(malloc 内存)", e); return -(int)e; }
  Mapping nm;
  nm.host = host_ptr;
  nm.bytes = bytes;
  g_maps.push_back(nm);
  clear_err();
  return 0;
}

int hipcold_host_unregister(void* host_ptr) {
  Mapping* m = find_map(host_ptr);
  if (!m) { clear_err(); return 0; }
  hipError_t e = hipHostUnregister(host_ptr);
  if (e != hipSuccess) { set_err_hip("hipHostUnregister", e); return -(int)e; }
  for (size_t i = 0; i < g_maps.size(); ++i) {
    if (g_maps[i].host == host_ptr) { g_maps.erase(g_maps.begin() + i); break; }
  }
  clear_err();
  return 0;
}

int hipcold_host_device_ptr(void* host_ptr, void** dev_ptr_out) {
  if (!dev_ptr_out) { set_err("dev_ptr_out 为空"); return 1; }
  if (!get_dev_ptr(host_ptr, dev_ptr_out)) return 1;
  clear_err();
  return 0;
}

int hipcold_memset(void* host_ptr, int value, size_t bytes) {
  if (!host_ptr || bytes == 0 || bytes % 4) {
    set_err("hipcold_memset 参数非法（bytes 需为 4 的倍数且>0）");
    return 1;
  }
  void* d = nullptr;
  if (!get_dev_ptr(host_ptr, &d)) return 1;
  size_t n = bytes / 4;
  fill_kernel<<<blocks_for(n), 256>>>((float*)d, n, (float)value);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) { set_err_hip("fill_kernel(mapped host)", e); return -(int)e; }
  clear_err();
  return 0;
}

int hipcold_vector_add(void* host_ptr, size_t bytes) {
  if (!host_ptr || bytes == 0 || bytes % 4) {
    set_err("hipcold_vector_add 参数非法（bytes 需为 4 的倍数且>0）");
    return 1;
  }
  void* d = nullptr;
  if (!get_dev_ptr(host_ptr, &d)) return 1;
  size_t n = bytes / 4;
  add1_kernel<<<blocks_for(n), 256>>>((float*)d, n);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) { set_err_hip("add1_kernel(mapped host)", e); return -(int)e; }
  clear_err();
  return 0;
}

int hipcold_verify_buffer(void* host_ptr, size_t bytes, float expect) {
  if (!host_ptr || bytes == 0 || bytes % 4) { set_err("参数非法"); return 1; }
  size_t n = bytes / 4;
  const float* p = (const float*)host_ptr;
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) if (p[i] != expect) ++bad;
  if (bad) { set_err("回校验不一致：%zu/%zu 个元素 != %g", bad, n, (double)expect); return 1; }
  clear_err();
  return 0;
}

int hipcold_host_mapped_bandwidth(void* host_ptr, size_t bytes) {
  if (hipcold_init(-1) != 0) return -1;
  bool owned = (host_ptr == nullptr);
  if (bytes == 0) bytes = 1024ull << 20;
  size_t want = clamp_to_budget(bytes);
  if (want == 0) {
    set_err("内存余量不足，拒绝分配（MemAvailable 太低）");
    return 1;
  }
  if (want != bytes) {
    std::printf("BUDGET: 请求 %zu MiB 缩档为 %zu MiB (MemAvailable=%.2f GiB, 引擎占 48GiB 页锁)\n",
                (size_t)(bytes >> 20), (size_t)(want >> 20),
                mem_available_bytes() / 1073741824.0);
    std::fflush(stdout);
  }
  bytes = want;
  void* buf = host_ptr;
  if (owned) {
    buf = std::malloc(bytes);
    if (!buf) { set_err("malloc(%zu MiB) 失败", (size_t)(bytes >> 20)); return 1; }
    std::memset(buf, 0, bytes);
  }
  int rc = hipcold_host_register(buf, bytes);
  if (rc != 0) { if (owned) std::free(buf); return rc; }
  void* d = nullptr;
  if (!get_dev_ptr(buf, &d)) { if (owned) std::free(buf); return 1; }

  size_t n = bytes / 4;
  int iters = 32;
  hipStream_t st = nullptr;
  hipError_t e = hipStreamCreate(&st);
  if (e != hipSuccess) { set_err_hip("hipStreamCreate", e); if (owned) std::free(buf); return -(int)e; }
  float* sink = nullptr;
  e = hipMalloc((void**)&sink, sizeof(float));
  if (e != hipSuccess) {
    set_err_hip("hipMalloc(sink)", e);
    hipStreamDestroy(st);
    if (owned) std::free(buf);
    return -(int)e;
  }
  // 预热：首次触页/迁移不计入（预热 3 次，排除 page-fault 与 TLB 冷启动）
  for (int i = 0; i < 3; ++i) {
    read_kernel<<<blocks_for(n), 256, 0, st>>>((const float*)d, n, sink);
  }
  e = hipStreamSynchronize(st);
  if (e != hipSuccess) {
    set_err_hip("warmup read_kernel", e);
    hipFree(sink); hipStreamDestroy(st);
    if (owned) std::free(buf);
    return -(int)e;
  }

  // 逐轮计时，报告最好一轮与平均（单轮计时能剔除系统抖动，最好值反映上限）
  double best = 0.0;
  double total = 0.0;
  for (int i = 0; i < iters; ++i) {
    double t = now_s();
    read_kernel<<<blocks_for(n), 256, 0, st>>>((const float*)d, n, sink);
    e = hipStreamSynchronize(st);
    double dt = now_s() - t;
    if (e != hipSuccess) break;
    double bw = (double)bytes / 1073741824.0 / dt;
    if (bw > best) best = bw;
    total += bw;
  }
  hipFree(sink);
  hipStreamDestroy(st);
  if (e != hipSuccess) {
    set_err_hip("read_kernel", e);
    if (owned) std::free(buf);
    return -(int)e;
  }
  std::printf("BW_mapped_host_read best %.1f GiB/s (avg %.1f GiB/s, %zu MiB x %d iters)\n",
              best, total / (double)iters, (size_t)(bytes >> 20), iters);
  std::fflush(stdout);
  if (owned) {
    hipcold_host_unregister(buf);
    std::free(buf);
  }
  clear_err();
  return 0;
}

int hipcold_kv_probe(size_t bytes) {
  if (hipcold_init(-1) != 0) return -1;
  if (bytes == 0) bytes = 64ull << 20;
  size_t want = clamp_to_budget(bytes);
  if (want == 0) { set_err("内存余量不足"); return 1; }
  bytes = want;
  float* buf = (float*)std::malloc(bytes);
  if (!buf) { set_err("malloc 失败"); return 1; }
  int rc = hipcold_host_register(buf, bytes);
  if (rc != 0) { std::free(buf); return rc; }
  void* d = nullptr;
  if (!get_dev_ptr(buf, &d)) { hipcold_host_unregister(buf); std::free(buf); return 1; }

  size_t n = bytes / 4;
  const int magic = 7;
  pattern_kernel<<<blocks_for(n), 256>>>((float*)d, n, magic);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) {
    set_err_hip("pattern_kernel(mapped host)", e);
    hipcold_host_unregister(buf);
    std::free(buf);
    return -(int)e;
  }
  // host 侧直接按同样公式读回比对：证明内核写的就是 CPU 能直接看到的那块内存
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    float exp = (float)(((int)(i % 251u) * magic) % 1021) * 0.5f;
    if (buf[i] != exp) {
      if (bad < 3) std::printf("  mismatch i=%zu got=%g exp=%g\n", i, (double)buf[i], (double)exp);
      ++bad;
    }
  }
  std::printf("KV_PROBE: %zu MiB mapped host, host-visible 比对 %s (bad=%zu)\n",
              (size_t)(bytes >> 20), bad ? "FAIL" : "PASS", bad);
  std::fflush(stdout);
  hipcold_host_unregister(buf);
  std::free(buf);
  if (bad) { set_err("kv_probe 回校验 %zu 个不一致", bad); return 1; }
  clear_err();
  return 0;
}

/* ======================= 冷专家 prefill（r429） ======================= */

int64_t hipcold_expert_blob_bytes(void) { return (int64_t) kNvBlobBytes; }

int64_t hipcold_expert_layer_stride(void) { return (int64_t) kNvLayerStride; }

int hipcold_unmap_expert_blob(int64_t layer, int32_t expert) {
  // r434：注册永不撤销。撤销会按页取整踩到同页的相邻 blob，且 8060S 可能仍有在飞访问
  //（r432/r433 的 gfxhub page fault、引擎 exit 1 即此因）。保留接口仅为 ABI 兼容。
  (void) layer;
  (void) expert;
  clear_err();
  return 0;
}

int hipcold_map_arena(void* base, size_t bytes) {
  if (hipcold_init(-1) != 0) return -1;
  if (!base || bytes == 0) { set_err("arena base/bytes 非法"); return 1; }
  if (g_arena.mapped) {
    if (g_arena.base == base && g_arena.bytes == bytes) { clear_err(); return 0; }  // 幂等
    set_err("已有 arena %p (%zu B)，先 hipcold_unmap_arena", g_arena.base, g_arena.bytes);
    return 1;
  }
  // r434b：纯元数据登记，不再对引擎内存做任何注册；blob 用前 memcpy 进库自持暂存槽。
  g_arena.base = base;
  g_arena.bytes = bytes;
  g_arena.dev = nullptr;
  g_arena.mapped = 1;
  clear_err();
  return 0;
}

// r430：挂偏移表的 arena 映射。base/bytes 与 hipcold_map_arena 相同，额外给一张
// offsets[layer*kNvExpertsPerLayer + expert] = 相对 base 的字节偏移表（~(uint64_t)0 = 不在
// arena 里），count 是表项数。表的内存属于调用方：本库只读它，且要求在 arena 反注册之前
// 一直有效。map 成功返回 0；重复映射同一 base/bytes/offsets 幂等返回 0。
int hipcold_set_arena_table(void* base, size_t bytes, const uint64_t* offsets, int64_t count) {
  if (!base || bytes == 0 || !offsets || count <= 0) {
    set_err("arena metadata invalid");
    return 1;
  }
  if (g_arena.mapped) hipcold_unmap_arena();
  g_arena.base = base;
  g_arena.bytes = bytes;
  g_arena.dev = nullptr;
  g_arena.mapped = 2;
  g_arena.offsets = offsets;
  g_arena.offsets_count = count;
  clear_err();
  return 0;
}

int hipcold_map_expert_blob(int64_t layer, int32_t expert, void* host_ptr) {
  if (!g_arena.mapped || !host_ptr || layer < 0 || expert < 0 || expert >= kNvExpertsPerLayer) {
    set_err("lazy blob arguments invalid");
    return 1;
  }
  size_t off = 0;
  if (arena_blob_offset(layer, expert, &off) != 0 || off + kNvBlobBytes > g_arena.bytes ||
      (uint8_t*) g_arena.base + off != host_ptr) {
    set_err("lazy blob offset invalid");
    return 1;
  }
  // r434b：不再注册引擎内存，blob memcpy 进库自持暂存槽（跨请求缓存，幂等）。
  trace("map_blob L%lld E%d enter", (long long) layer, expert);
  BlobMapping* bm = blob_ensure(layer, expert, host_ptr);
  if (!bm) return 1;
  clear_err();
  return 0;
}

int hipcold_map_arena_table(void* base, size_t bytes, const uint64_t* offsets, int64_t count) {
  if (offsets == nullptr || count <= 0) { set_err("hipcold_map_arena_table: offsets/count 非法"); return 1; }
  if (hipcold_init(-1) != 0) return -1;
  if (!base || bytes == 0) { set_err("arena base/bytes 非法"); return 1; }
  if (g_arena.mapped) {
    if (g_arena.base == base && g_arena.bytes == bytes && g_arena.offsets == offsets &&
        g_arena.offsets_count == count) {
      clear_err();
      return 0;  // 幂等
    }
    set_err("已有 arena %p (%zu B)，先 hipcold_unmap_arena", g_arena.base, g_arena.bytes);
    return 1;
  }
  // r434b：同 hipcold_map_arena——纯元数据登记，不注册引擎内存；blob 走暂存槽。
  g_arena.base = base;
  g_arena.bytes = bytes;
  g_arena.dev = nullptr;
  g_arena.mapped = 1;
  g_arena.offsets = offsets;
  g_arena.offsets_count = count;
  clear_err();
  return 0;
}

int hipcold_prep_xq80(const uint8_t* xq80, int64_t rows) {
  if (hipcold_init(-1) != 0) return -1;
  if (!xq80 || rows < 1 || rows > 8192) { set_err("hipcold_prep_xq80: xq80/rows 非法（rows 1..8192）"); return 1; }
  trace("prep enter rows=%lld", (long long) rows);
  int rc = ensure_big_bufs();
  if (rc != 0) return rc;
  trace("prep big_bufs ok");
  rc = hipcold_host_register((void*) xq80, (size_t) rows * (40 * 80));
  if (rc != 0) return rc;
  trace("prep xq80 registered");
  void* alias = nullptr;
  if (!get_dev_ptr((void*) xq80, &alias)) return 1;
  trace("prep devptr ok");
  const int64_t warps = rows * 40;
  nvfp4_prep_x80_kernel<<<(unsigned) ((warps * 32 + 255) / 256), 256>>>((const uint8_t*) alias, (uint8_t*) g_dxqu,
                                                                        (int*) g_dxqs, (float*) g_dxqd, rows);
  hipError_t e = hipGetLastError();
  if (e == hipSuccess) e = hipMemsetAsync(g_dout_big, 0, (size_t) rows * kNvH * sizeof(float));
  if (e != hipSuccess) { set_err_hip("prep_x80", e); return -(int)e; }
  trace("prep done");
  g_xrows = rows;
  clear_err();
  return 0;
}

int hipcold_prefill_expert_sparse(int64_t layer, int32_t expert, const int32_t* token_idx, int ntok,
                                  const float* weights) {
  if (hipcold_init(-1) != 0) return -1;
  if (!g_arena.mapped) { set_err("arena 未映射"); return 1; }
  if (!token_idx) { set_err("token_idx 不得为空"); return 1; }
  if (expert < 0 || expert >= kNvExpertsPerLayer || layer < 0) { set_err("layer/expert 非法"); return 1; }
  if (ntok < 1 || ntok > 8192) { set_err("ntok 必须在 1..8192（现 %d）", ntok); return 1; }
  if (g_xrows < 1) { set_err("尚未 hipcold_prep_xq80"); return 1; }
  for (int i = 0; i < ntok; ++i) {
    if (token_idx[i] < 0 || token_idx[i] >= g_xrows) { set_err("token_idx[%d]=%d 越界（行数 %lld）", i,
                                                               token_idx[i], (long long) g_xrows);
      return 1; }
  }
  int rc = ensure_big_bufs();
  if (rc != 0) return rc;
  size_t boff = 0;
  rc = arena_blob_offset(layer, expert, &boff);
  if (rc != 0) return rc;
  if (boff + kNvBlobBytes > g_arena.bytes) { set_err("arena 太小：blob 偏移 %zu 越界", boff); return 1; }
  trace("sparse L%lld E%d ensure", (long long) layer, expert);
  const uint8_t* blob = nullptr;
  BlobMapping* bm = nullptr;
  if (g_arena.mapped == 1) {
    // r434b：走暂存槽（源 = 引擎 arena 里的 blob，库内 memcpy 缓存）。
    bm = blob_ensure(layer, expert, (const uint8_t*) g_arena.base + boff);
    if (!bm) return 1;
    blob = (const uint8_t*) bm->dev;
  } else {
    bm = find_blob_map(layer, expert);
    if (!bm) {
      set_err("cold expert blob is not mapped");
      return 1;
    }
    blob = (const uint8_t*) bm->dev;
  }
  // r435：后台暂存模式下，发射 kernel 前等目标槽就绪（稳态近乎零阻塞；见 stage_ensure_locked 注释）
  if (g_stage_workers > 0 && bm->slot >= 0) stage_wait_ready(bm->slot);
  float tail[4];
  std::memcpy(tail, (const uint8_t*) bm->host + kNvTailOff, sizeof(tail));  // r434n：tail 从 arena 源读（dev 是设备内存）
  // r434h/m：tidx/weights 工作区 64 槽轮转——复用某槽前只等该槽自己上次的事件
  //（未记录过的事件视为已完成，立即返回），不再满圈全设备排干。
  if (g_pending > 0) {
    const double t0 = now_ms();
    hipError_t we;
    if (g_swork_evts_ok) {
      we = hipEventSynchronize(g_swork_evts[(size_t) g_swork_idx]);
    } else if (g_swork_idx == 0) {
      we = hipDeviceSynchronize();
    } else {
      we = hipSuccess;
    }
    g_ms_sync += now_ms() - t0;
    if (we != hipSuccess) { set_err_hip("sparse wrap sync", we); return -(int) we; }
  }
  const size_t slot_off = (size_t) g_swork_idx * (size_t) 8192 * sizeof(int32_t);
  const size_t wbase = (size_t) kWorkSlots * (size_t) 8192 * sizeof(int32_t);
  int32_t* tidx_h = (int32_t*) (g_swork.host + slot_off);
  float* w_h = (float*) (g_swork.host + wbase + slot_off);
  std::memcpy(tidx_h, token_idx, (size_t) ntok * sizeof(int32_t));
  if (weights) std::memcpy(w_h, weights, (size_t) ntok * sizeof(float));
  int32_t* tidx_d = (int32_t*) ((uint8_t*) g_swork.dev + slot_off);
  float* w_d = (float*) ((uint8_t*) g_swork.dev + wbase + slot_off);
  const int nbatch = (ntok + HIPCOLD_TB - 1) / HIPCOLD_TB;
  nvfp4_gu_kernel_v2<<<dim3(kNvFF / 8, nbatch), 256>>>(blob, (const uint8_t*) g_dxqu, (const int*) g_dxqs,
                                                       (const float*) g_dxqd, ntok, tail[0], tail[1],
                                                       (float*) g_dff, tidx_d);
  nvfp4_quant_kernel<<<warps_to_blocks((size_t) ntok * kQ8BlocksPerFF), 256>>>((const float*) g_dff,
                                                                               (uint8_t*) g_dhq,
                                                                               (int*) g_dhqs, (float*) g_dhqd,
                                                                               ntok);
  nvfp4_row_dot_kernel_v2<<<dim3(kNvH / 8, nbatch), 256>>>(blob + kNvDownOff, (const uint8_t*) g_dhq,
                                                           (const int*) g_dhqs, (const float*) g_dhqd, ntok,
                                                           tail[2], weights ? w_d : nullptr,
                                                           (float*) g_dout_big, tidx_d);
  hipError_t e = hipGetLastError();
  // r434h/m：不在此处全同步——给本工作槽和本暂存槽各记一个完成事件，
  // 复用/淘汰时只等对应事件；统一排干只在 fetch_out。
  if (e != hipSuccess) { set_err_hip("sparse pipeline", e); return -(int)e; }
  if (g_swork_evts_ok) hipEventRecord(g_swork_evts[(size_t) g_swork_idx], 0);
  if (bm && bm->slot >= 0 && g_stage[(size_t) bm->slot].evt) {
    hipEventRecord(g_stage[(size_t) bm->slot].evt, 0);
  }
  g_swork_idx = (g_swork_idx + 1) % kWorkSlots;
  ++g_pending;
  ++g_n_sparse;
  trace("sparse L%lld E%d done", (long long) layer, expert);
  clear_err();
  return 0;
}

// 全部专家的稀疏累加完成后统一取回：sync 后 D2H 覆盖写入 out（调用方语义 = 先清零）。
int hipcold_fetch_out(float* out, int64_t out_rows) {
  if (!out) { set_err("out 为空"); return 1; }
  if (g_xrows < 1) { set_err("尚未 hipcold_prep_xq80"); return 1; }
  if (out_rows < g_xrows) { set_err("out_rows %lld 小于激活行数 %lld", (long long) out_rows, (long long) g_xrows);
    return 1; }
  const double tf0 = now_ms();
  hipError_t e = hipDeviceSynchronize();
  g_ms_sync += now_ms() - tf0;
  if (e != hipSuccess) { set_err_hip("fetch_out sync", e); return -(int)e; }
  g_pending = 0;  // r434h：已排干，工作槽可安全复用
  const double tf1 = now_ms();
  e = hipMemcpy(out, g_dout_big, (size_t) g_xrows * kNvH * sizeof(float), hipMemcpyDeviceToHost);
  g_ms_fetch += now_ms() - tf1;
  if (e != hipSuccess) { set_err_hip("fetch_out D2H", e); return -(int)e; }
  trace("STATS sparse=%lld evict=%lld sync_ms=%.0f stage_ms=%.0f fetch_ms=%.0f", g_n_sparse, g_n_evict,
        g_ms_sync, g_ms_stage, g_ms_fetch);
  clear_err();
  return 0;
}

int hipcold_unmap_arena(void) {
  // r434b：引擎内存本就不注册；这里清元数据、暂存槽缓存与库自有工作缓冲。
  // 暂存区本身（hipHostAlloc）随进程保留，槽内容标记失效即可。
  g_blob_maps.clear();
  {
    std::lock_guard<std::mutex> lk(g_stage_mtx);
    for (auto& s : g_stage) { s.layer = -1; s.expert = -1; s.lru = 0; }
  }
  if (g_work.ptr) {
    hipcold_host_unregister(g_work.ptr);
    std::free(g_work.ptr);
    g_work.ptr = nullptr;
    g_work.bytes = 0;
    g_work.dev = nullptr;
  }
  free_device_bufs();
  // v2 路径会把调用方的 out/weights 也注册进来，unmap 时统一反注册，避免 pin 泄漏。
  while (!g_maps.empty()) {
    hipHostUnregister(g_maps.back().host);
    g_maps.pop_back();
  }
  if (!g_arena.mapped) { clear_err(); return 0; }
  g_arena.base = nullptr;
  g_arena.bytes = 0;
  g_arena.dev = nullptr;
  g_arena.mapped = 0;
  g_arena.offsets = nullptr;
  g_arena.offsets_count = 0;
  clear_err();
  return 0;
}

int hipcold_prefill_experts(int64_t layer, const int32_t* expert_ids, int n_experts, const float* x, int tokens,
                            const float* weights, float* out) {
  if (hipcold_init(-1) != 0) return -1;
  if (!g_arena.mapped) { set_err("arena 未映射（先 hipcold_map_arena）"); return 1; }
  if (!expert_ids || !x || !out) { set_err("expert_ids/x/out 不得为空"); return 1; }
  if (layer < 0) { set_err("layer 不得为负"); return 1; }
  if (n_experts < 1 || n_experts > 32) { set_err("n_experts 必须在 1..32（现 %d）", n_experts); return 1; }
  if (tokens < 1 || tokens > 64) { set_err("tokens 必须在 1..64（现 %d）", tokens); return 1; }
  for (int e = 0; e < n_experts; ++e) {
    if (expert_ids[e] < 0 || expert_ids[e] >= kNvExpertsPerLayer) {
      set_err("expert_id[%d]=%d 超出 0..%d", e, expert_ids[e], kNvExpertsPerLayer - 1);
      return 1;
    }
  }
  // r430：每层的 blob 起点不再假定为 layer*kNvLayerStride——带了偏移表就逐专家查表，
  // 并把实际偏移缓存下来（下面 v1/v2 两条路径都用这份）。
  size_t blob_off[32];   // n_experts <= 32（上面的检查已保证）
  for (int e = 0; e < n_experts; ++e) {
    size_t off = 0;
    if (arena_blob_offset(layer, expert_ids[e], &off) != 0) return 1;
    if (off + kNvBlobBytes > g_arena.bytes) {
      set_err("arena 太小（%zu B）：layer %lld expert %d 需到偏移 %zu", g_arena.bytes, (long long) layer,
              expert_ids[e], off + kNvBlobBytes);
      return 1;
    }
    blob_off[e] = off;
  }

  const size_t xbytes = (size_t) tokens * kNvH * sizeof(float);
  const size_t obytes = xbytes;
  const size_t ffbytes = (size_t) tokens * kNvFF * sizeof(float);
  const size_t qxbytes = (size_t) tokens * kQ8BlocksPerH * kQ8Blk;
  const size_t qdbytes = (size_t) tokens * kQ8BlocksPerFF * kQ8Blk;
  const size_t total = xbytes + qxbytes + ffbytes + qdbytes + obytes;
  if (total > (512ull << 20)) { set_err("单次工作区 %zu B 超过 512MiB 上限", total); return 1; }

  // 8095 引擎占 48GiB 页锁，先查余量再分配，宁可失败也不挤爆线上引擎。
  const unsigned long long avail = mem_available_bytes();
  if (avail != 0 && (unsigned long long) total + (512ull << 20) > avail) {
    set_err("MemAvailable %.2f GiB 不足以留 512MiB 余量（需 %zu B）", avail / 1073741824.0, total);
    return 1;
  }

  uint8_t* buf = g_work.ptr;
  if (g_work.bytes < total) {
    if (g_work.ptr) {  // 按需换代：先反注册/释放旧的，再换成更大的
      hipcold_host_unregister(g_work.ptr);
      std::free(g_work.ptr);
      g_work.ptr = nullptr;
      g_work.bytes = 0;
      g_work.dev = nullptr;
    }
    void* p = std::malloc(total);
    if (!p) { set_err("malloc(%zu B) 失败", total); return 1; }
    const int rc = hipcold_host_register(p, total);
    if (rc != 0) { std::free(p); return rc; }
    void* pd = nullptr;
    if (!get_dev_ptr(p, &pd)) { hipcold_host_unregister(p); std::free(p); return 1; }
    g_work.ptr = (uint8_t*) p;
    g_work.bytes = total;
    g_work.dev = pd;
    buf = (uint8_t*) p;
  }
  uint8_t* xq = buf;
  uint8_t* ff = xq + qxbytes;
  uint8_t* hq = ff + ffbytes;
  float* dout = (float*) (hq + qdbytes);
  for (int t = 0; t < tokens; ++t) {
    quantize_q8_0_host(x + (size_t) t * kNvH, xq + (size_t) t * kQ8BlocksPerH * kQ8Blk, kNvH);
  }

  uint8_t* dxq = (uint8_t*) g_work.dev;
  uint8_t* dff = dxq + qxbytes;
  uint8_t* dhq = dff + ffbytes;
  float* dout_dev = (float*) (dhq + qdbytes);

  if (std::getenv("HIPCOLD_V1") == nullptr) {
    // v2：注册 out/weights 为 mapped，kernel 零拷贝直写；全程无中途 sync。
    hipError_t e = hipSuccess;
    int rc = ensure_device_bufs();
    if (rc != 0) return rc;
    e = hipMemset(g_dout, 0, obytes);
    if (e != hipSuccess) { set_err_hip("hipMemset(out)", e); return -(int)e; }
    const float* rw_dev = nullptr;
    if (weights) {
      rc = hipcold_host_register((void*) weights, (size_t) n_experts * tokens * sizeof(float));
      if (rc != 0) return rc;
      void* w_alias = nullptr;
      if (!get_dev_ptr((void*) weights, &w_alias)) return 1;
      rw_dev = (const float*) w_alias;
    }
    const int nbatch = (tokens + HIPCOLD_TB - 1) / HIPCOLD_TB;
    nvfp4_prep_x_kernel<<<warps_to_blocks((size_t) tokens * kQ8BlocksPerH), 256>>>(dxq, (uint8_t*) g_dxqu,
                                                                                   (int*) g_dxqs,
                                                                                   (float*) g_dxqd, tokens);
    for (int eidx = 0; eidx < n_experts; ++eidx) {
      // r434b：blob 经暂存槽供给（源 = 引擎 arena 对应偏移）。
      BlobMapping* bm = blob_ensure(layer, expert_ids[eidx], (const uint8_t*) g_arena.base + blob_off[eidx]);
      if (!bm) return 1;
      if (g_stage_workers > 0 && bm->slot >= 0) stage_wait_ready(bm->slot);   // r435
      const uint8_t* blob = (const uint8_t*) bm->dev;
      float tail[4];
      std::memcpy(tail, (const uint8_t*) bm->host + kNvTailOff, sizeof(tail));
      nvfp4_gu_kernel_v2<<<dim3(kNvFF / 8, nbatch), 256>>>(blob, (const uint8_t*) g_dxqu, (const int*) g_dxqs,
                                                           (const float*) g_dxqd, tokens, tail[0], tail[1],
                                                           (float*) g_dff);
      nvfp4_quant_kernel<<<warps_to_blocks((size_t) tokens * kQ8BlocksPerFF), 256>>>((const float*) g_dff,
                                                                                     (uint8_t*) g_dhq,
                                                                                     (int*) g_dhqs,
                                                                                     (float*) g_dhqd, tokens);
      nvfp4_row_dot_kernel_v2<<<dim3(kNvH / 8, nbatch), 256>>>(blob + kNvDownOff, (const uint8_t*) g_dhq,
                                                               (const int*) g_dhqs, (const float*) g_dhqd,
                                                               tokens, tail[2],
                                                               rw_dev ? rw_dev + (size_t) eidx * tokens : nullptr,
                                                               (float*) g_dout);
      if (bm->slot >= 0 && g_stage[(size_t) bm->slot].evt) hipEventRecord(g_stage[(size_t) bm->slot].evt, 0);
    }
    e = hipGetLastError();
    if (e == hipSuccess) e = hipDeviceSynchronize();
    if (e != hipSuccess) { set_err_hip("v2 pipeline", e); return -(int)e; }
    // 收尾：设备累加结果 D2H 后按路由语义并入调用方 out（接口约定调用方先清零）。
    e = hipMemcpy(dout, g_dout, obytes, hipMemcpyDeviceToHost);
    if (e != hipSuccess) { set_err_hip("hipMemcpy(out D2H)", e); return -(int)e; }
    for (int t = 0; t < tokens; ++t) {
      float* orow = out + (size_t) t * kNvH;
      const float* drow = dout + (size_t) t * kNvH;
      for (int k = 0; k < kNvH; ++k) orow[k] += drow[k];
    }
  } else {
  hipError_t e = hipMemcpy(dxq, xq, qxbytes, hipMemcpyHostToDevice);
  if (e != hipSuccess) { set_err_hip("hipMemcpy(activation q8)", e); return -(int)e; }

  for (int eidx = 0; eidx < n_experts; ++eidx) {
    // r434b：blob 经暂存槽供给（源 = 引擎 arena 对应偏移）。
    BlobMapping* bm = blob_ensure(layer, expert_ids[eidx], (const uint8_t*) g_arena.base + blob_off[eidx]);
    if (!bm) return 1;
    if (g_stage_workers > 0 && bm->slot >= 0) stage_wait_ready(bm->slot);   // r435
    const uint8_t* blob = (const uint8_t*) bm->dev;
    float tail[4];
    std::memcpy(tail, (const uint8_t*) bm->host + kNvTailOff, sizeof(tail));
    const float sg = tail[0], su = tail[1], sd = tail[2];

    nvfp4_gu_kernel<<<warps_to_blocks((size_t) tokens * kNvFF), 256>>>(blob, dxq, kQ8BlocksPerH * kQ8Blk,
                                                                      tokens, sg, su, (float*) dff);
    e = hipDeviceSynchronize();
    if (e != hipSuccess) { set_err_hip("nvfp4_gu_kernel", e); break; }

    for (int t = 0; t < tokens; ++t) {
      quantize_q8_0_host((const float*) (ff + (size_t) t * kNvFF * sizeof(float)),
                         hq + (size_t) t * kQ8BlocksPerFF * kQ8Blk, kNvFF);
    }
    e = hipMemcpy(dhq, hq, qdbytes, hipMemcpyHostToDevice);
    if (e != hipSuccess) { set_err_hip("hipMemcpy(hidden q8)", e); break; }

    nvfp4_row_dot_kernel<<<warps_to_blocks((size_t) tokens * kNvH), 256>>>(blob + kNvDownOff, kNvH, kNvDRow,
                                                                         kNvBlocksPerDRow, dhq, kQ8BlocksPerFF * kQ8Blk,
                                                                         tokens, sd, dout_dev);
    e = hipDeviceSynchronize();
    if (e != hipSuccess) { set_err_hip("nvfp4_row_dot_kernel(down)", e); break; }

    e = hipMemcpy(dout, dout_dev, obytes, hipMemcpyDeviceToHost);
    if (e != hipSuccess) { set_err_hip("hipMemcpy(out)", e); break; }
    const float* w = weights ? (weights + (size_t) eidx * tokens) : nullptr;
    for (int t = 0; t < tokens; ++t) {
      const float we = w ? w[t] : 1.0f;
      float* orow = out + (size_t) t * kNvH;
      const float* drow = dout + (size_t) t * kNvH;
      for (int k = 0; k < kNvH; ++k) orow[k] += we * drow[k];
    }
  }
  if (e != hipSuccess) return -(int)e;
  }

  clear_err();
  return 0;
}

}  // extern "C"

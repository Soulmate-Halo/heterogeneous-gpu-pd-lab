// strata/kernels/mrope.hpp - multimodal rotary positions (the vision path).
//
// qwen4exp rotates with interleaved M-RoPE (llama.cpp LLAMA_ROPE_TYPE_IMROPE, rope.dimension_sections 11/11/10/0):
// each of the 32 rotated pairs takes its position from one of three streams, time t, height h and width w.  Text
// has t = h = w, which is why every rope kernel here takes one position per row.  An image does not: its tokens
// share t and differ in h and w (mtmd: t = p, h = p + y, w = p + x), and the text after it continues at
// p + max(nx, ny), not at its cell index.
//
// The table: nullptr (the default) keeps every kernel exactly as before - the position a caller passes is the
// rotary position.  Set, it is a DEVICE int32 [cells][3] (t, h, w) and the position a caller passes is read as a
// CELL index; pair i rotates by tab[cell * 3 + mrope_sector(i)].  Every caller in the engine passes cell indices
// (pos_base is 0 everywhere), so the table covers the prompt path, the verify window and the MTP drafter at once.
// The pointer must be set before any CUDA graph is captured (kernels take it as an argument); the contents may
// change between requests.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The table of the CURRENT device (a layer split sets one per device; null = the identity).
void mrope_table_set(const int32_t* device_table);
const int32_t* mrope_table();

#if defined(__CUDACC__) || defined(__HIPCC__)
/// ggml rope_multi, is_imrope, sections {11, 11, 10, 0}: sector = pair % 32; sector % 3 == 1 -> h (sector < 33),
/// == 2 -> w (sector < 30), == 0 -> t (sector < 33).  For pairs 0..31 all three bounds hold, so it is pair % 3.
__device__ __forceinline__ int mrope_pos(const int32_t* tab, int pos, int pair) {
    return tab ? __ldg(tab + (size_t) pos * 3 + pair % 3) : pos;
}
#endif

}  // namespace strata::kernels

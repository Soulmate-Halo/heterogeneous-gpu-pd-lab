#pragma once

namespace strata::kernels {
void native_rope_set_enabled(bool enabled);
bool native_rope_enabled();

// Pinned CUDA text-only IMRoPE: F32 rows, 64 rotated channels, no YaRN/frequency
// factors, equal text positions in all four IMRoPE sections. Each device position
// must be nonnegative. The position buffer remains live through graph replay.
// Supports head_dim 128/256 and exact x==out; partial overlap is rejected.
// Explicit stream required. No allocation or synchronization.
void native_rope_apply(const float* x, float* out, int rows, int head_dim,
                       int n_rot, float freq_base, const int* positions, void* stream);
}

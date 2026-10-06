// include/strata/platform/x86_pause.hpp - portable spin-pause / store-fence for the doorbell loops.
//
// The engine's low-latency waits (`_mm_pause`, `_mm_sfence`) are x86-only.  DGX Spark (GB10) is aarch64,
// where the equivalents are `yield` and `dmb ish`.  Every spin loop in the engine already falls back to a
// real-clock deadline or an atomic fence, so on any other target both macros are plain no-ops and the code
// stays correct, only less polite to the pipeline.
#pragma once

#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define STRATA_X86_PAUSE() _mm_pause()
#define STRATA_X86_SFENCE() _mm_sfence()
#define STRATA_X86_ISA 1
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64)
#define STRATA_X86_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#define STRATA_X86_SFENCE() __asm__ __volatile__("dmb ish" ::: "memory")
#define STRATA_X86_ISA 0
#else
#define STRATA_X86_PAUSE() ((void) 0)
#define STRATA_X86_SFENCE() ((void) 0)
#define STRATA_X86_ISA 0
#endif

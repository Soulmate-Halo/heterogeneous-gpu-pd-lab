// Numerical parity for the CUDA intrinsics supplied by the HIP compatibility layer.
#include <hip/hip_runtime.h>
#include "strata/hip_compat/intrinsics.hpp"

#include <cstdint>
#include <cstdio>
#include <limits>

#define CHECK(call)                                                                                                  \
    do {                                                                                                             \
        const hipError_t error = (call);                                                                             \
        if (error != hipSuccess) {                                                                                   \
            std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error));                                     \
            return 2;                                                                                                \
        }                                                                                                            \
    } while (0)

namespace {

constexpr int kLanes = 32;
constexpr int kIntegerResults = 7;

__global__ void intrinsic_probe(int* integer_results, float* float_shuffle, double* double_shuffle,
                                unsigned* ballot, const uint32_t* dot_inputs) {
    const int lane = static_cast<int>(threadIdx.x);
    constexpr unsigned mask = 0xffffffffu;
    const uint32_t a = 0x80ff017fu;
    const uint32_t b = 0x0203fe81u;

    integer_results[lane * kIntegerResults + 0] = __dp4a(static_cast<int>(a), static_cast<int>(b), 7);
    integer_results[lane * kIntegerResults + 1] = __dp4a(0x7f7f7f7f, 0x7f7f7f7f, 0x7fffffff);
    integer_results[lane * kIntegerResults + 2] = __vsub4(0x00ff0102, 0x01010103);
    integer_results[lane * kIntegerResults + 3] = __vsub4(0x00000000, 0x01020304);
    integer_results[lane * kIntegerResults + 4] = __vsubss4(0x807f0080, 0x017f0180);
    integer_results[lane * kIntegerResults + 5] = __vcmpne4(0x00ff0102, 0x01010103);
    integer_results[lane * kIntegerResults + 6] =
        __dp4a(static_cast<int>(dot_inputs[lane * 2]), static_cast<int>(dot_inputs[lane * 2 + 1]),
               0x7fffffff - lane);

    float_shuffle[lane * 4 + 0] = __shfl_xor_sync(mask, static_cast<float>(lane), 1);
    float_shuffle[lane * 4 + 1] = static_cast<float>(__shfl_down_sync(mask, lane, 4, 8));
    float_shuffle[lane * 4 + 2] = static_cast<float>(__shfl_up_sync(mask, lane, 1));
    float_shuffle[lane * 4 + 3] = static_cast<float>(__shfl_sync(mask, lane, 0));
    double_shuffle[lane] = __shfl_xor_sync(mask, static_cast<double>(lane) + 0.25, 1);

    const unsigned hit = __ballot_sync(mask, (lane & 1) == 0);
    if (lane == 0) {
        *ballot = hit;
        __nanosleep(100);
    }
}

int signed_byte(uint32_t word, int lane) {
    const unsigned value = (word >> (lane * 8)) & 0xffu;
    return value < 0x80u ? static_cast<int>(value) : static_cast<int>(value) - 0x100;
}

int dp4a_reference(uint32_t a, uint32_t b, int c) {
    int64_t sum = c;
    for (int lane = 0; lane < 4; ++lane) sum += signed_byte(a, lane) * signed_byte(b, lane);
    return static_cast<int32_t>(static_cast<uint32_t>(sum));
}

uint32_t byte_sub_reference(uint32_t a, uint32_t b) {
    uint32_t out = 0;
    for (int lane = 0; lane < 4; ++lane)
        out |= ((((a >> (lane * 8)) & 0xffu) - ((b >> (lane * 8)) & 0xffu)) & 0xffu) << (lane * 8);
    return out;
}

uint32_t signed_saturating_sub_reference(uint32_t a, uint32_t b) {
    uint32_t out = 0;
    for (int lane = 0; lane < 4; ++lane) {
        int value = signed_byte(a, lane) - signed_byte(b, lane);
        if (value < -128) value = -128;
        if (value > 127) value = 127;
        out |= (static_cast<uint32_t>(value) & 0xffu) << (lane * 8);
    }
    return out;
}

uint32_t not_equal_reference(uint32_t a, uint32_t b) {
    uint32_t out = 0;
    for (int lane = 0; lane < 4; ++lane)
        if (((a >> (lane * 8)) & 0xffu) != ((b >> (lane * 8)) & 0xffu)) out |= 0xffu << (lane * 8);
    return out;
}

}  // namespace

int main() {
    int device = 0;
    CHECK(hipGetDevice(&device));
    hipDeviceProp_t properties{};
    CHECK(hipGetDeviceProperties(&properties, device));
    if (properties.warpSize != kLanes) {
        std::fprintf(stderr, "expected wave32, got wave%d\n", properties.warpSize);
        return 1;
    }

    int* device_integers = nullptr;
    float* device_float_shuffle = nullptr;
    double* device_double_shuffle = nullptr;
    unsigned* device_ballot = nullptr;
    uint32_t* device_dot_inputs = nullptr;
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_integers), kLanes * kIntegerResults * sizeof(int)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_float_shuffle), kLanes * 4 * sizeof(float)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_double_shuffle), kLanes * sizeof(double)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_ballot), sizeof(unsigned)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_dot_inputs), kLanes * 2 * sizeof(uint32_t)));

    uint32_t dot_inputs[kLanes * 2]{};
    for (int lane = 0; lane < kLanes; ++lane) {
        dot_inputs[lane * 2] = 0x80ff017fu ^ (0x01010101u * static_cast<uint32_t>(lane));
        dot_inputs[lane * 2 + 1] = 0x0203fe81u + (0x11111111u * static_cast<uint32_t>(lane));
    }
    CHECK(hipMemcpy(device_dot_inputs, dot_inputs, sizeof(dot_inputs), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(intrinsic_probe, dim3(1), dim3(kLanes), 0, 0,
                       device_integers, device_float_shuffle, device_double_shuffle, device_ballot, device_dot_inputs);
    CHECK(hipDeviceSynchronize());

    int integers[kLanes * kIntegerResults]{};
    float float_shuffle[kLanes * 4]{};
    double double_shuffle[kLanes]{};
    unsigned ballot = 0;
    CHECK(hipMemcpy(integers, device_integers, sizeof(integers), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(float_shuffle, device_float_shuffle, sizeof(float_shuffle), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(double_shuffle, device_double_shuffle, sizeof(double_shuffle), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&ballot, device_ballot, sizeof(ballot), hipMemcpyDeviceToHost));

    constexpr uint32_t a = 0x80ff017fu, b = 0x0203fe81u;
    const uint32_t expected_integer[kIntegerResults - 1] = {
        static_cast<uint32_t>(dp4a_reference(a, b, 7)),
        static_cast<uint32_t>(dp4a_reference(0x7f7f7f7fu, 0x7f7f7f7fu, std::numeric_limits<int>::max())),
        byte_sub_reference(0x00ff0102u, 0x01010103u),
        byte_sub_reference(0x00000000u, 0x01020304u),
        signed_saturating_sub_reference(0x807f0080u, 0x017f0180u),
        not_equal_reference(0x00ff0102u, 0x01010103u),
    };
    bool ok = ballot == 0x55555555u;
    for (int lane = 0; lane < kLanes; ++lane) {
        for (int i = 0; i < kIntegerResults - 1; ++i)
            ok = ok && static_cast<uint32_t>(integers[lane * kIntegerResults + i]) == expected_integer[i];
        ok = ok && static_cast<uint32_t>(integers[lane * kIntegerResults + kIntegerResults - 1]) ==
                     static_cast<uint32_t>(dp4a_reference(dot_inputs[lane * 2], dot_inputs[lane * 2 + 1],
                                                          0x7fffffff - lane));
        ok = ok && float_shuffle[lane * 4 + 0] == static_cast<float>(lane ^ 1);
        const int down = lane % 8 < 4 ? lane + 4 : lane;
        ok = ok && float_shuffle[lane * 4 + 1] == static_cast<float>(down);
        ok = ok && float_shuffle[lane * 4 + 2] == static_cast<float>(lane == 0 ? 0 : lane - 1);
        ok = ok && float_shuffle[lane * 4 + 3] == 0.0f;
        ok = ok && double_shuffle[lane] == static_cast<double>(lane ^ 1) + 0.25;
    }

    CHECK(hipFree(device_ballot));
    CHECK(hipFree(device_dot_inputs));
    CHECK(hipFree(device_double_shuffle));
    CHECK(hipFree(device_float_shuffle));
    CHECK(hipFree(device_integers));
    if (!ok) {
        std::fprintf(stderr, "HIP intrinsic parity failed (ballot 0x%08x)\n", ballot);
        return 1;
    }
    std::puts("HIP intrinsics parity OK: dynamic signed dot4/overflow, packed integer boundaries, wave32 shuffles, ballot and sleep");
    return 0;
}

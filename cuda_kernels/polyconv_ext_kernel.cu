#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void gather_windows_kernel(
    const float* __restrict__ flat_windows,
    const int64_t* __restrict__ flat_idx,
    float* __restrict__ out,
    int64_t B,
    int64_t P,
    int64_t F,
    int64_t OC,
    int64_t L,
    int64_t I
) {
    int64_t total = B * OC * P * L * I;
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    int64_t i = linear % I;
    linear /= I;
    int64_t l = linear % L;
    linear /= L;
    int64_t p = linear % P;
    linear /= P;
    int64_t oc = linear % OC;
    linear /= OC;
    int64_t b = linear;

    int64_t idx_offset = (oc * L + l) * I + i;
    int64_t feature_idx = flat_idx[idx_offset];

    if (feature_idx < 0 || feature_idx >= F) {
        out[(((b * OC + oc) * P + p) * L + l) * I + i] = 0.0f;
        return;
    }

    int64_t in_offset = (b * P + p) * F + feature_idx;
    int64_t out_offset = (((b * OC + oc) * P + p) * L + l) * I + i;
    out[out_offset] = flat_windows[in_offset];
}

}  // namespace

at::Tensor polyconv_gather_windows_cuda(
    at::Tensor flat_windows,
    at::Tensor flat_idx
) {
    const auto B = flat_windows.size(0);
    const auto P = flat_windows.size(1);
    const auto F = flat_windows.size(2);

    const auto OC = flat_idx.size(0);
    const auto L = flat_idx.size(1);
    const auto I = flat_idx.size(2);

    auto out = at::zeros({B, OC, P, L, I}, flat_windows.options());

    int64_t total = B * OC * P * L * I;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    gather_windows_kernel<<<blocks, threads>>>(
        flat_windows.data_ptr<float>(),
        flat_idx.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        B,
        P,
        F,
        OC,
        L,
        I
    );

    return out;
}

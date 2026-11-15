#include <torch/extension.h>

at::Tensor polyconv_gather_windows_cuda(
    at::Tensor flat_windows,
    at::Tensor flat_idx
);

at::Tensor gather_windows(
    at::Tensor flat_windows,
    at::Tensor flat_idx
) {
    TORCH_CHECK(flat_windows.is_cuda(), "flat_windows must be a CUDA tensor");
    TORCH_CHECK(flat_idx.is_cuda(), "flat_idx must be a CUDA tensor");
    TORCH_CHECK(flat_windows.scalar_type() == at::kFloat, "flat_windows must be float32");
    TORCH_CHECK(flat_idx.scalar_type() == at::kLong, "flat_idx must be int64");
    TORCH_CHECK(flat_windows.dim() == 3, "flat_windows must have shape [B, P, F]");
    TORCH_CHECK(flat_idx.dim() == 3, "flat_idx must have shape [OC, L, I]");

    return polyconv_gather_windows_cuda(flat_windows.contiguous(), flat_idx.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_windows", &gather_windows, "PolyConv gather windows (CUDA)");
}

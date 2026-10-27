#pragma once

#include "../core/cuda_buffer.cuh"
#include "options.h"
#include <memory>
#ifdef GPU_FEMM_ENABLE_CUDSS
#include <cudss.h>
#endif

namespace gpu_femm
{

#ifdef GPU_FEMM_ENABLE_CUDSS
struct CudssContext;
class CudssDirectPlan
{
public:
  ~CudssDirectPlan();
  CudssDirectPlan();
  CudssDirectPlan(const CudssDirectPlan &) = delete;
  CudssDirectPlan &operator=(const CudssDirectPlan &) = delete;

  void Reset();

  void InvalidateValues();

  Status Initialize(int n, const std::vector<int32_t> &rows, const std::vector<int32_t> &columns,
                    int32_t *device_rows, int32_t *device_columns);

  Status SolveBatch(size_t count, size_t value_stride, const double *values, const double *diagonal,
                    const double *rhs, double *solution, double tolerance, int max_iterations,
                    std::vector<SolveInfo> *infos, std::vector<std::vector<double>> *solutions,
                    int *launches, GpuBatchSolveTiming *timing);

private:
  Status EnsureTransferCapacity(size_t count);
  int n_ = 0, nnz_ = 0;
  int32_t *rows_ = nullptr;
  int32_t *columns_ = nullptr;
  // Host/device indices are borrowed from the owning GpuCsrSolver until Reset.
  const std::vector<int32_t> *host_rows_ = nullptr, *host_columns_ = nullptr;
  DeviceBuffer<int32_t> transpose_;
  DeviceBuffer<SolveInfo> info_;
  std::vector<std::unique_ptr<CudssContext>> contexts_;
  SolveInfo *host_info_ = nullptr;
  double *host_solutions_ = nullptr;
  size_t transfer_capacity_ = 0;
};
#endif

} // namespace gpu_femm

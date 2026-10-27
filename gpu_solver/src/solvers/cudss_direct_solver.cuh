#pragma once

#include "../core/cuda_buffer.cuh"
#include "options.h"
#ifdef GPU_FEMM_ENABLE_CUDSS
#include <cudss.h>
#endif

namespace gpu_femm
{

#ifdef GPU_FEMM_ENABLE_CUDSS
class CudssDirectPlan
{
public:
  ~CudssDirectPlan();
  CudssDirectPlan() = default;
  CudssDirectPlan(const CudssDirectPlan &) = delete;
  CudssDirectPlan &operator=(const CudssDirectPlan &) = delete;

  void Reset();

  void InvalidateValues();

  Status Initialize(int n, const std::vector<int32_t> &rows, const std::vector<int32_t> &columns,
                    int32_t *device_rows, int32_t *device_columns, double *values, double *rhs,
                    double *solution);

  Status SolveBatch(size_t count, size_t value_stride, const double *values, const double *diagonal,
                    const double *rhs, double *solution, double tolerance, int max_iterations,
                    std::vector<SolveInfo> *infos, std::vector<std::vector<double>> *solutions,
                    int *launches, GpuBatchSolveTiming *timing);

private:
  static Status Call(cudssStatus_t value);
  Status Execute(int phase);
  Status Check(const double *values, const double *diagonal, const double *rhs, double *x,
               bool solved, double tolerance, SolveInfo *info);
  int n_ = 0, nnz_ = 0;
  int32_t *rows_ = nullptr;
  int32_t *columns_ = nullptr;
  cudssHandle_t handle_ = nullptr;
  cudssConfig_t config_ = nullptr;
  cudssData_t data_ = nullptr;
  cudssMatrix_t a_ = nullptr, b_ = nullptr, x_ = nullptr;
  bool analyzed_ = false, factor_valid_ = false;
  const double *fact_values_ = nullptr;
  DeviceBuffer<int32_t> transpose_;
  DeviceBuffer<SolveInfo> info_;
};
#endif

} // namespace gpu_femm

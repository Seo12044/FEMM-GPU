#pragma once

#include "../core/cuda_buffer.cuh"
#include "options.h"

namespace gpu_femm
{

class BlockJacobiPlan
{
public:
  void Invalidate();
  void ReuseUnmodifiedOperator();
  const double *inverse() const;
  const int *owner() const;
  const PreconditionerStats &stats() const;
  size_t bytes() const;

  Status Prepare(int n, size_t nnz, size_t count, const int32_t *rows, const int32_t *columns,
                 const double *values, const double *diagonal);

private:
  int n_ = 0;
  size_t nnz_ = 0, count_ = 0;
  bool valid_ = false;
  PreconditionerStats stats_;
  DeviceBuffer<double> inverse_, previous_values_, previous_diagonal_;
  DeviceBuffer<int> owner_, previous_owner_, rebuild_, counters_;
};

} // namespace gpu_femm

#pragma once

#include "../core/cuda_buffer.cuh"
#include "options.h"

namespace gpu_femm
{

class PcgSpmvLayout
{
public:
  void Reset();

  Status Initialize(const std::vector<int32_t> &rows, const std::vector<int32_t> &columns,
                    bool enabled);

  Status Pack(size_t count, const double *csr);

  bool enabled() const;
  int slots() const;
  const int32_t *slices() const;
  const int32_t *columns() const;
  const double *values() const;
  size_t numeric_bytes() const;

private:
  int nnz_ = 0, slots_ = 0;
  DeviceBuffer<int32_t> slices_, source_, columns_;
  DeviceBuffer<double> values_;
};

} // namespace gpu_femm

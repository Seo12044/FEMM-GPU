#pragma once

#include "options.h"
#include <cuda_runtime.h>
#include <cmath>

namespace gpu_femm
{

__device__ inline double DeterministicBlockSum(double local, double *scratch)
{
  const int lane = threadIdx.x;
  scratch[lane] = local;
  __syncthreads();
  for (int stride = kPcgBlockThreads / 2; stride >= 64; stride /= 2) {
    if (lane < stride)
      scratch[lane] += scratch[lane + stride];
    __syncthreads();
  }
  if (lane < warpSize) {
    double value = scratch[lane] + scratch[lane + warpSize];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      value += __shfl_down_sync(0xffffffffu, value, offset);
    if (lane == 0)
      scratch[0] = value;
  }
  __syncthreads();
  const double result = scratch[0];
  __syncthreads(); // A following reduction must not overwrite unread results.
  return result;
}

__device__ inline double DeterministicBlockMax(double local, double *scratch)
{
  const int lane = threadIdx.x;
  scratch[lane] = local;
  __syncthreads();
  for (int stride = kPcgBlockThreads / 2; stride >= 64; stride /= 2) {
    if (lane < stride)
      scratch[lane] = fmax(scratch[lane], scratch[lane + stride]);
    __syncthreads();
  }
  if (lane < warpSize) {
    double value = fmax(scratch[lane], scratch[lane + warpSize]);
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      value = fmax(value, __shfl_down_sync(0xffffffffu, value, offset));
    if (lane == 0)
      scratch[0] = value;
  }
  __syncthreads();
  const double result = scratch[0];
  __syncthreads();
  return result;
}

template <bool kUseInverseDiagonal, bool kBlock>
__device__ inline double PcgPreconditionRow(int row, int n, const double *residual,
                                            const double *diagonal, const double *inverse_diagonal,
                                            const double *blocks)
{
  if constexpr (kBlock) {
    constexpr int B = kJacobiBlockSize;
    const int first = row / B * B;
    const double *block = blocks + static_cast<size_t>(row / B) * kJacobiBlockStorage;
    if (isnan(block[0]))
      return PcgPreconditionRow<kUseInverseDiagonal, false>(row, n, residual, diagonal,
                                                            inverse_diagonal, nullptr);
    double value = 0.0;
    for (int j = 0; j < min(B, n - first); ++j)
      value += block[j * B + row % B] * residual[first + j];
    return value;
  } else if constexpr (kUseInverseDiagonal) {
    return residual[row] * inverse_diagonal[row];
  } else {
    return residual[row] / diagonal[row];
  }
}

__device__ inline double PcgRowDot(int row, const int32_t *rows, const int32_t *columns,
                                   const double *values, const double *vector,
                                   const int32_t *slices)
{
  double sum = 0.0;
  if (slices) {
    const int slice = slices[row / kPcgSliceRows];
    int index = slice < 0 ? (-1 - slice) + (rows[row] - rows[row / kPcgSliceRows * kPcgSliceRows])
                          : slice + row % kPcgSliceRows;
    const int stride = slice < 0 ? 1 : kPcgSliceRows;
    for (int entry = rows[row]; entry < rows[row + 1]; ++entry, index += stride)
      sum += values[index] * vector[columns[index]];
  } else {
    for (int entry = rows[row]; entry < rows[row + 1]; ++entry)
      sum += values[entry] * vector[columns[entry]];
  }
  return sum;
}

} // namespace gpu_femm

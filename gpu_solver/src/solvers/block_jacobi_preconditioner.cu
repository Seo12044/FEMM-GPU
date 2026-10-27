#include "block_jacobi_preconditioner.cuh"
#include "../core/profile.h"
#include <climits>
#include <cmath>

namespace gpu_femm
{

__global__ void MatchBlockOperators(int n, int nnz, const double *values, const double *diagonal,
                                    const double *previous_values, const double *previous_diagonal,
                                    const int *previous_owner, bool valid, int *owner, int *rebuild,
                                    int *counters)
{
  const int item = blockIdx.x;
  const int lane = threadIdx.x;
  __shared__ int different;
  __shared__ int match;
  if (lane == 0)
    match = item;
  __syncthreads();
  for (int candidate = 0; candidate < item; ++candidate) {
    if (lane == 0)
      different = 0;
    __syncthreads();
    for (int j = lane; j < nnz + n; j += blockDim.x) {
      const double a = j < nnz ? values[static_cast<size_t>(item) * nnz + j]
                               : diagonal[static_cast<size_t>(item) * n + j - nnz];
      const double b = j < nnz ? values[static_cast<size_t>(candidate) * nnz + j]
                               : diagonal[static_cast<size_t>(candidate) * n + j - nnz];
      if (__double_as_longlong(a) != __double_as_longlong(b))
        atomicExch(&different, 1);
    }
    __syncthreads();
    if (!different) {
      if (lane == 0)
        match = candidate;
      break;
    }
    __syncthreads(); // All warps must consume the flag before lane zero resets it.
  }
  __syncthreads();
  if (match == item) {
    if (lane == 0)
      different = !valid || previous_owner[item] != item;
    __syncthreads();
    if (!different) {
      // All threads take this branch together; do not mutate `different`
      // until every thread has read the branch condition.
      __syncthreads();
      int local_different = 0;
      for (int j = lane; j < nnz + n; j += blockDim.x) {
        const size_t offset =
            j < nnz ? static_cast<size_t>(item) * nnz + j : static_cast<size_t>(item) * n + j - nnz;
        const double a = j < nnz ? values[offset] : diagonal[offset];
        const double b = j < nnz ? previous_values[offset] : previous_diagonal[offset];
        local_different |= __double_as_longlong(a) != __double_as_longlong(b);
      }
      __syncthreads();
      if (local_different)
        atomicExch(&different, 1);
    }
    __syncthreads();
  }
  if (lane == 0) {
    owner[item] = match;
    rebuild[item] = match == item && different;
    atomicAdd(counters + (rebuild[item] ? 0 : 1), 1);
  }
}

// Invert each SPD principal block via a diagonally scaled Cholesky solve.
// Only the preconditioner is approximated; the PCG operator is unchanged.
__global__ void BuildBlockInverse(int n, int nnz, int blocks, const int32_t *rows,
                                  const int32_t *columns, const double *values,
                                  const double *diagonal, const int *rebuild, double *inverse,
                                  int *counters)
{
  const int item = blockIdx.y;
  if (!rebuild[item])
    return;
  const int block = blockIdx.x;
  const int lane = threadIdx.x;
  const int first = block * kJacobiBlockSize;
  const int width = min(kJacobiBlockSize, n - first);
  values += static_cast<size_t>(item) * nnz;
  diagonal += static_cast<size_t>(item) * n;
  inverse += (static_cast<size_t>(item) * blocks + block) * kJacobiBlockStorage;
  constexpr int B = kJacobiBlockSize;
  __shared__ double factor[B * B];
  __shared__ double scale[B];
  __shared__ int bad;
  if (lane == 0)
    bad = 0;
  scale[lane] = lane < width ? sqrt(diagonal[first + lane]) : 1.0;
  for (int col = 0; col < B; ++col)
    factor[lane * B + col] = 0.0;
  __syncthreads();
  if (lane < width) {
    for (int j = rows[first + lane]; j < rows[first + lane + 1]; ++j) {
      const int col = columns[j] - first;
      if (col >= 0 && col < width)
        factor[lane * B + col] += values[j] / scale[lane] / scale[col];
    }
  }
  __syncthreads();
  if (lane == 0) {
    for (int row = 0; row < width; ++row) {
      const double reciprocal = 1.0 / diagonal[first + row];
      if (!isfinite(reciprocal) || reciprocal < 2.2250738585072014e-308)
        bad = 1;
      for (int col = 0; col <= row; ++col) {
        const double a = factor[row * B + col], b = factor[col * B + row];
        if (!isfinite(a) || !isfinite(b) || fabs(a - b) > 1e-12 * fmax(1.0, fmax(fabs(a), fabs(b))))
          bad = 1;
      }
    }
    for (int col = 0; col < width && !bad; ++col) {
      double pivot = factor[col * B + col];
      for (int k = 0; k < col; ++k)
        pivot -= factor[col * B + k] * factor[col * B + k];
      if (!(pivot > 0.0) || !isfinite(pivot)) {
        bad = 1;
        break;
      }
      factor[col * B + col] = sqrt(pivot);
      for (int row = col + 1; row < width; ++row) {
        double value = factor[row * B + col];
        for (int k = 0; k < col; ++k)
          value -= factor[row * B + k] * factor[col * B + k];
        factor[row * B + col] = value / factor[col * B + col];
      }
    }
  }
  __syncthreads();
  double x[B];
  int invalid_inverse = 0;
  if (!bad && lane < width) {
    for (int row = 0; row < width; ++row) {
      double value = row == lane ? 1.0 : 0.0;
      for (int k = 0; k < row; ++k)
        value -= factor[row * B + k] * x[k];
      x[row] = value / factor[row * B + row];
    }
    for (int row = width - 1; row >= 0; --row) {
      double value = x[row];
      for (int k = row + 1; k < width; ++k)
        value -= factor[k * B + row] * x[k];
      x[row] = value / factor[row * B + row];
    }
    for (int row = 0; row < width; ++row) {
      x[row] = x[row] / scale[row] / scale[lane];
      if (!isfinite(x[row]))
        invalid_inverse = 1;
    }
  }
  __syncthreads();
  if (invalid_inverse)
    atomicExch(&bad, 1);
  __syncthreads();
  // Explicit, counted fallback for an unusable block. Original diagonal and
  // true-residual validation still reject invalid equations in PCG.
  if (lane == 0 && bad)
    atomicAdd(counters + 2, 1);
  // Column-major storage makes adjacent PCG lanes read adjacent entries.
  for (int row = 0; row < B; ++row)
    inverse[lane * B + row] =
        bad ? (row == 0 && lane == 0 ? NAN : 0.0) : (row < width && lane < width ? x[row] : 0.0);
}

void BlockJacobiPlan::Invalidate() { valid_ = false; }

void BlockJacobiPlan::ReuseUnmodifiedOperator() { ++stats_.reuses; }

const double *BlockJacobiPlan::inverse() const { return inverse_.get(); }

const int *BlockJacobiPlan::owner() const { return owner_.get(); }

const PreconditionerStats &BlockJacobiPlan::stats() const { return stats_; }

size_t BlockJacobiPlan::bytes() const
{
  return (inverse_.count() + previous_values_.count() + previous_diagonal_.count()) *
             sizeof(double) +
         (owner_.count() + previous_owner_.count() + rebuild_.count() + counters_.count()) *
             sizeof(int);
}

Status BlockJacobiPlan::Prepare(int n, size_t nnz, size_t count, const int32_t *rows,
                                const int32_t *columns, const double *values,
                                const double *diagonal)
{
  const auto start = ProfileClock::now();
  if (n <= 0 || nnz > static_cast<size_t>(INT_MAX - n) || count == 0 || count > 65535)
    return Status::kInvalidArgument;
  const size_t blocks = (static_cast<size_t>(n) + kJacobiBlockSize - 1) / kJacobiBlockSize;
  if (n_ != n || nnz_ != nnz || count_ != count) {
    valid_ = false;
    n_ = 0; // A failed allocation cannot leave a usable partial plan.
    Status status;
    if ((status = inverse_.allocate(count * blocks * kJacobiBlockStorage)) != Status::kOk ||
        (status = previous_values_.allocate(count * nnz)) != Status::kOk ||
        (status = previous_diagonal_.allocate(count * n)) != Status::kOk ||
        (status = owner_.allocate(count)) != Status::kOk ||
        (status = previous_owner_.allocate(count)) != Status::kOk ||
        (status = rebuild_.allocate(count)) != Status::kOk ||
        (status = counters_.allocate(3)) != Status::kOk)
      return status;
    n_ = n;
    nnz_ = nnz;
    count_ = count;
  }
  if (cudaMemset(counters_.get(), 0, 3 * sizeof(int)) != cudaSuccess)
    return Status::kInternalError;
  MatchBlockOperators<<<static_cast<unsigned>(count), 256>>>(
      n, static_cast<int>(nnz), values, diagonal, previous_values_.get(), previous_diagonal_.get(),
      previous_owner_.get(), valid_, owner_.get(), rebuild_.get(), counters_.get());
  if (cudaGetLastError() != cudaSuccess) {
    valid_ = false;
    return Status::kInternalError;
  }
  BuildBlockInverse<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(count)),
                      kJacobiBlockSize>>>(n, static_cast<int>(nnz), static_cast<int>(blocks), rows,
                                          columns, values, diagonal, rebuild_.get(), inverse_.get(),
                                          counters_.get());
  valid_ = false;
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
    return Status::kInternalError;
  if (cudaMemcpy(previous_values_.get(), values, count * nnz * sizeof(double),
                 cudaMemcpyDeviceToDevice) != cudaSuccess ||
      cudaMemcpy(previous_diagonal_.get(), diagonal, count * n * sizeof(double),
                 cudaMemcpyDeviceToDevice) != cudaSuccess ||
      cudaMemcpy(previous_owner_.get(), owner_.get(), count * sizeof(int),
                 cudaMemcpyDeviceToDevice) != cudaSuccess)
    return Status::kInternalError;
  int counters[3];
  if (CopyToHost(counters, counters_.get(), sizeof(counters)) != Status::kOk)
    return Status::kInternalError;
  valid_ = true;
  stats_.builds += counters[0];
  stats_.reuses += counters[1];
  stats_.fallback_blocks += counters[2];
  stats_.setup_seconds += ProfileSecondsSince(start);
  return Status::kOk;
}

} // namespace gpu_femm

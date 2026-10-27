#include "pcg_kernels.cuh"
#include "pcg_device.cuh"
#include <cooperative_groups.h>

namespace gpu_femm
{

namespace cg = cooperative_groups;

template <bool kUseInverseDiagonal, bool kBlock = false>
__global__ void
DeterministicPcgKernel(int n, const int32_t *row_offsets, const int32_t *column_indices,
                       const double *values, const double *diagonal, const double *inverse_diagonal,
                       const double *rhs, double *x, double *residual, double *direction,
                       double *preconditioned, double *matrix_direction, double relative_tolerance,
                       int max_iterations, SolveInfo *info, int values_per_item,
                       const double *block_inverse = nullptr, const int *block_owner = nullptr,
                       const int32_t *slices = nullptr)
{
  // One fixed-width block owns one independent current-state.  B=1 and B>1
  // use the same fixed reduction tree; sums may differ slightly from the
  // former lane-zero order but are deterministic.
  const int item = blockIdx.x;
  if constexpr (kBlock)
    block_inverse += static_cast<size_t>(block_owner[values_per_item == 0 ? 0 : item]) *
                     ((n + kJacobiBlockSize - 1) / kJacobiBlockSize) * kJacobiBlockStorage;
  // values_per_item==0 is the explicit shared-operator mode: every block
  // reads the same already-uploaded K/diagonal/preconditioner, while all PCG
  // vector workspaces remain disjoint per RHS.
  values += static_cast<size_t>(item) * values_per_item;
  const size_t operator_row_offset = values_per_item == 0 ? 0 : static_cast<size_t>(item) * n;
  diagonal += operator_row_offset;
  inverse_diagonal += operator_row_offset;
  rhs += static_cast<size_t>(item) * n;
  x += static_cast<size_t>(item) * n;
  residual += static_cast<size_t>(item) * n;
  direction += static_cast<size_t>(item) * n;
  preconditioned += static_cast<size_t>(item) * n;
  matrix_direction += static_cast<size_t>(item) * n;
  info += item;

  const int lane = threadIdx.x;
  __shared__ double scalar_a;
  __shared__ double scalar_b;
  __shared__ double reduction_scratch[kPcgBlockThreads];
  __shared__ int stop;
  __shared__ int threshold_reached;
  __shared__ int true_residual_verified;
  if (lane == 0) {
    info->status = Status::kInternalError;
    info->iterations = 0;
    info->residual_l2 = INFINITY;
    stop = 0;
    threshold_reached = 0;
    true_residual_verified = 0;
    if (n <= 0 || max_iterations < 0 || !(relative_tolerance > 0.0) ||
        !(relative_tolerance < 1.0) || !isfinite(relative_tolerance)) {
      info->status = Status::kInvalidArgument;
      stop = 1;
    }
  }
  __syncthreads();
  if (stop)
    return;

  double invalid_local = 0.0;
  for (int i = lane; i < n; i += kPcgBlockThreads) {
    if (!isfinite(rhs[i]) || !isfinite(diagonal[i]) || !(diagonal[i] > 0.0))
      invalid_local = fmax(invalid_local, static_cast<double>(n - i));
  }
  const double invalid_priority = DeterministicBlockMax(invalid_local, reduction_scratch);
  if (lane == 0 && invalid_priority != 0.0) {
    // Preserve the old scan's first-invalid-row error precedence.
    const int first = n - static_cast<int>(invalid_priority);
    info->status = !isfinite(rhs[first]) || !isfinite(diagonal[first])
                       ? Status::kNumericalNonfinite
                       : Status::kLinearSolveBreakdown;
    stop = 1;
  }
  __syncthreads();
  if (stop)
    return;

  for (int i = lane; i < n; i += kPcgBlockThreads)
    x[i] = 0.0;
  __syncthreads();
  double local_max = 0.0;
  for (int i = lane; i < n; i += kPcgBlockThreads)
    local_max = fmax(local_max, fabs(rhs[i]));
  const double rhs_scale = DeterministicBlockMax(local_max, reduction_scratch);
  if (rhs_scale == 0.0) {
    if (lane == 0) {
      info->residual_l2 = 0.0;
      info->status = Status::kOk;
    }
    return;
  }

  for (int i = lane; i < n; i += kPcgBlockThreads) {
    residual[i] = rhs[i] / rhs_scale;
  }
  if constexpr (kBlock)
    __syncthreads();
  for (int i = lane; i < n; i += kPcgBlockThreads) {
    preconditioned[i] = PcgPreconditionRow<kUseInverseDiagonal, kBlock>(
        i, n, residual, diagonal, inverse_diagonal, block_inverse);
    direction[i] = preconditioned[i];
  }
  __syncthreads();
  double initial_residual_local = 0.0;
  double initial_rho_local = 0.0;
  for (int i = lane; i < n; i += kPcgBlockThreads) {
    initial_residual_local += residual[i] * residual[i];
    initial_rho_local += residual[i] * preconditioned[i];
  }
  const double initial_residual_sq =
      DeterministicBlockSum(initial_residual_local, reduction_scratch);
  const double initial_rho = DeterministicBlockSum(initial_rho_local, reduction_scratch);
  if (lane == 0) {
    scalar_a = initial_residual_sq;
    scalar_b = initial_rho;
    if (!isfinite(scalar_a) || !isfinite(scalar_b)) {
      info->status = Status::kNumericalNonfinite;
      stop = 1;
    } else {
      info->residual_l2 = sqrt(scalar_a);
    }
  }
  __syncthreads();
  if (stop)
    return;
  double rho = scalar_b;
  const double rhs_norm = sqrt(scalar_a);
  const double threshold = relative_tolerance * rhs_norm;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    double direction_matrix_local = 0.0;
    for (int row = lane; row < n; row += kPcgBlockThreads) {
      const double sum = PcgRowDot(row, row_offsets, column_indices, values, direction, slices);
      matrix_direction[row] = sum;
      direction_matrix_local += direction[row] * sum;
    }
    const double direction_matrix_sum =
        DeterministicBlockSum(direction_matrix_local, reduction_scratch);
    if (lane == 0) {
      scalar_a = direction_matrix_sum;
      if (!isfinite(scalar_a) || !isfinite(rho)) {
        info->status = Status::kNumericalNonfinite;
        stop = 1;
      } else if (!(scalar_a > 0.0) || !(rho > 0.0)) {
        info->status = Status::kLinearSolveBreakdown;
        stop = 1;
      }
    }
    __syncthreads();
    if (stop)
      return;

    const double alpha = rho / scalar_a;
    double residual_local = 0.0;
    [[maybe_unused]] double updated_rho_local = 0.0;
    for (int i = lane; i < n; i += kPcgBlockThreads) {
      x[i] += alpha * direction[i];
      residual[i] -= alpha * matrix_direction[i];
      residual_local += residual[i] * residual[i];
      if constexpr (!kBlock) {
        preconditioned[i] = PcgPreconditionRow<kUseInverseDiagonal, false>(
            i, n, residual, diagonal, inverse_diagonal, nullptr);
        updated_rho_local += residual[i] * preconditioned[i];
      }
    }
    const double residual_sum = DeterministicBlockSum(residual_local, reduction_scratch);
    if (lane == 0) {
      scalar_a = residual_sum;
      info->iterations = iteration + 1;
      if (!isfinite(scalar_a)) {
        info->status = Status::kNumericalNonfinite;
        stop = 1;
      } else {
        scalar_b = sqrt(scalar_a);
        info->residual_l2 = scalar_b;
        if (!isfinite(scalar_b)) {
          info->status = Status::kNumericalNonfinite;
          stop = 1;
        } else {
          threshold_reached = scalar_b <= threshold ? 1 : 0;
        }
      }
    }
    __syncthreads();
    if (stop)
      return;
    if (threshold_reached) {
      for (int row = lane; row < n; row += kPcgBlockThreads) {
        const double matrix_solution =
            PcgRowDot(row, row_offsets, column_indices, values, x, slices);
        residual[row] = rhs[row] / rhs_scale - matrix_solution;
      }
      __syncthreads();
      double true_residual_local = 0.0;
      for (int i = lane; i < n; i += kPcgBlockThreads)
        true_residual_local += residual[i] * residual[i];
      const double true_residual_sum =
          DeterministicBlockSum(true_residual_local, reduction_scratch);
      if (lane == 0) {
        scalar_a = true_residual_sum;
        if (!isfinite(scalar_a)) {
          info->status = Status::kNumericalNonfinite;
          stop = 1;
        } else {
          scalar_b = sqrt(scalar_a);
          info->residual_l2 = scalar_b;
          true_residual_verified = scalar_b <= threshold ? 1 : 0;
        }
      }
      __syncthreads();
      if (stop)
        return;
      if (true_residual_verified) {
        for (int i = lane; i < n; i += kPcgBlockThreads) {
          x[i] *= rhs_scale;
        }
        __syncthreads();
        if (lane == 0) {
          for (int i = 0; i < n; ++i) {
            if (!isfinite(x[i])) {
              info->status = Status::kNumericalNonfinite;
              stop = 1;
              break;
            }
          }
          if (!stop) {
            info->residual_l2 *= rhs_scale;
            if (!isfinite(info->residual_l2)) {
              info->status = Status::kNumericalNonfinite;
              stop = 1;
            } else {
              info->status = Status::kOk;
            }
          }
        }
        __syncthreads();
        if (stop)
          return;
        return;
      }
      for (int i = lane; i < n; i += kPcgBlockThreads) {
        preconditioned[i] = PcgPreconditionRow<kUseInverseDiagonal, kBlock>(
            i, n, residual, diagonal, inverse_diagonal, block_inverse);
        direction[i] = preconditioned[i];
      }
      __syncthreads();
      double restart_rho_local = 0.0;
      for (int i = lane; i < n; i += kPcgBlockThreads)
        restart_rho_local += residual[i] * preconditioned[i];
      const double restart_rho = DeterministicBlockSum(restart_rho_local, reduction_scratch);
      if (lane == 0) {
        rho = restart_rho;
        if (!isfinite(rho) || !(rho > 0.0)) {
          // Historical behavior maps a failed restart rho to BREAKDOWN.
          info->status = Status::kLinearSolveBreakdown;
          stop = 1;
        } else {
          scalar_b = rho;
        }
      }
      __syncthreads();
      if (stop)
        return;
      rho = scalar_b;
      continue;
    }

    double preconditioned_local = updated_rho_local;
    if constexpr (kBlock) {
      preconditioned_local = 0.0;
      for (int i = lane; i < n; i += kPcgBlockThreads) {
        preconditioned[i] = PcgPreconditionRow<kUseInverseDiagonal, true>(
            i, n, residual, diagonal, inverse_diagonal, block_inverse);
        preconditioned_local += residual[i] * preconditioned[i];
      }
    }
    const double preconditioned_sum =
        DeterministicBlockSum(preconditioned_local, reduction_scratch);
    if (lane == 0) {
      scalar_a = preconditioned_sum;
      if (!isfinite(scalar_a) || !(scalar_a > 0.0)) {
        info->status = Status::kLinearSolveBreakdown;
        stop = 1;
      }
    }
    __syncthreads();
    if (stop)
      return;
    const double beta = scalar_a / rho;
    for (int i = lane; i < n; i += kPcgBlockThreads) {
      direction[i] = preconditioned[i] + beta * direction[i];
    }
    rho = scalar_a;
    __syncthreads();
  }

  for (int row = lane; row < n; row += kPcgBlockThreads) {
    const double matrix_solution = PcgRowDot(row, row_offsets, column_indices, values, x, slices);
    const double true_residual = rhs[row] / rhs_scale - matrix_solution;
    residual[row] = true_residual;
  }
  __syncthreads();
  double final_residual_local = 0.0;
  for (int i = lane; i < n; i += kPcgBlockThreads)
    final_residual_local += residual[i] * residual[i];
  const double final_residual_sum = DeterministicBlockSum(final_residual_local, reduction_scratch);
  if (lane == 0) {
    scalar_a = final_residual_sum;
    if (!isfinite(scalar_a)) {
      info->status = Status::kNumericalNonfinite;
      stop = 1;
    }
  }
  __syncthreads();
  if (stop)
    return;
  for (int i = lane; i < n; i += kPcgBlockThreads) {
    x[i] *= rhs_scale;
  }
  __syncthreads();
  if (lane == 0) {
    for (int i = 0; i < n; ++i) {
      if (!isfinite(x[i])) {
        info->status = Status::kNumericalNonfinite;
        stop = 1;
        break;
      }
    }
    if (!stop) {
      info->residual_l2 = sqrt(scalar_a) * rhs_scale;
      if (!isfinite(info->residual_l2)) {
        info->status = Status::kNumericalNonfinite;
        stop = 1;
      } else {
        info->status = Status::kLinearSolveNotConverged;
      }
    }
  }
}

// The regular PCG kernel deliberately keeps a one-block-per-item mapping for
// large batches.  Small batches underfill this GPU, so this cooperative path
// partitions one item into a fixed number of shards.  It is launched only
// when every block is known to be resident; every grid-wide synchronization is
// therefore a cooperative-groups barrier, never an unsafe software barrier.
__device__ double CooperativeItemSum(double local, double *reduction_scratch, double *partials,
                                     double *scalars, int item, int shard, int shard_count,
                                     int scalar_slot, cg::grid_group grid)
{
  const double block_sum = DeterministicBlockSum(local, reduction_scratch);
  if (threadIdx.x == 0)
    partials[static_cast<size_t>(item) * shard_count + shard] = block_sum;
  grid.sync();
  if (shard == 0 && threadIdx.x == 0) {
    double sum = 0.0;
    const size_t first = static_cast<size_t>(item) * shard_count;
    for (int other = 0; other < shard_count; ++other)
      sum += partials[first + other];
    scalars[static_cast<size_t>(item) * 3 + scalar_slot] = sum;
  }
  grid.sync();
  return scalars[static_cast<size_t>(item) * 3 + scalar_slot];
}

__device__ double CooperativeItemMax(double local, double *reduction_scratch, double *partials,
                                     double *scalars, int item, int shard, int shard_count,
                                     int scalar_slot, cg::grid_group grid)
{
  const double block_max = DeterministicBlockMax(local, reduction_scratch);
  if (threadIdx.x == 0)
    partials[static_cast<size_t>(item) * shard_count + shard] = block_max;
  grid.sync();
  if (shard == 0 && threadIdx.x == 0) {
    double maximum = 0.0;
    const size_t first = static_cast<size_t>(item) * shard_count;
    for (int other = 0; other < shard_count; ++other)
      maximum = fmax(maximum, partials[first + other]);
    scalars[static_cast<size_t>(item) * 3 + scalar_slot] = maximum;
  }
  grid.sync();
  return scalars[static_cast<size_t>(item) * 3 + scalar_slot];
}

// Independent sums keep the established block tree and shard order, but
// exchange their partials together. The second bank is sized for the whole
// cooperative batch, including inactive items.
__device__ void CooperativeItemSumPair(double local_a, double local_b, double *reduction_scratch,
                                       double *partials, double *scalars, int item, int shard,
                                       int shard_count, int item_count, double &sum_a,
                                       double &sum_b, cg::grid_group grid)
{
  const double block_a = DeterministicBlockSum(local_a, reduction_scratch);
  const double block_b = DeterministicBlockSum(local_b, reduction_scratch);
  const size_t bank = static_cast<size_t>(item_count) * shard_count;
  const size_t index = static_cast<size_t>(item) * shard_count + shard;
  if (threadIdx.x == 0) {
    partials[index] = block_a;
    partials[bank + index] = block_b;
  }
  grid.sync();
  if (shard == 0 && threadIdx.x == 0) {
    double a = 0.0, b = 0.0;
    const size_t first = static_cast<size_t>(item) * shard_count;
    for (int other = 0; other < shard_count; ++other) {
      a += partials[first + other];
      b += partials[bank + first + other];
    }
    scalars[static_cast<size_t>(item) * 3 + 2] = a;
    scalars[static_cast<size_t>(item) * 3] = b;
  }
  grid.sync();
  sum_a = scalars[static_cast<size_t>(item) * 3 + 2];
  sum_b = scalars[static_cast<size_t>(item) * 3];
}

template <bool kUseInverseDiagonal, bool kBlock = false>
__global__ void CooperativeDeterministicPcgKernel(
    int n, const int32_t *row_offsets, const int32_t *column_indices, const double *values,
    const double *diagonal, const double *inverse_diagonal, const double *rhs, double *x,
    double *residual, double *direction, double *preconditioned, double *matrix_direction,
    double relative_tolerance, int max_iterations, SolveInfo *info, int values_per_item,
    int item_count, int shards_per_item, double *partials, double *scalars, int *controls,
    const double *block_inverse = nullptr, const int *block_owner = nullptr,
    const int32_t *slices = nullptr)
{
  cg::grid_group grid = cg::this_grid();
  const int block = static_cast<int>(blockIdx.x);
  const int item = block / shards_per_item;
  if constexpr (kBlock)
    block_inverse += static_cast<size_t>(block_owner[values_per_item == 0 ? 0 : item]) *
                     ((n + kJacobiBlockSize - 1) / kJacobiBlockSize) * kJacobiBlockStorage;
  const int shard = block % shards_per_item;
  const int lane = threadIdx.x;
  const int begin = static_cast<int>(static_cast<int64_t>(n) * shard / shards_per_item);
  const int end = static_cast<int>(static_cast<int64_t>(n) * (shard + 1) / shards_per_item);
  constexpr int kActive = 0;
  constexpr int kThreshold = 1;
  constexpr int kRestart = 2;
  constexpr int kVerified = 3;
  int *item_controls = controls + static_cast<size_t>(item) * 4;
  const int global_any_index = 4 * item_count;
  values += static_cast<size_t>(item) * values_per_item;
  const size_t operator_row_offset = values_per_item == 0 ? 0 : static_cast<size_t>(item) * n;
  diagonal += operator_row_offset;
  inverse_diagonal += operator_row_offset;
  rhs += static_cast<size_t>(item) * n;
  x += static_cast<size_t>(item) * n;
  residual += static_cast<size_t>(item) * n;
  direction += static_cast<size_t>(item) * n;
  preconditioned += static_cast<size_t>(item) * n;
  matrix_direction += static_cast<size_t>(item) * n;
  info += item;

  __shared__ double reduction_scratch[kPcgBlockThreads];
  if (shard == 0 && lane == 0) {
    info->status = Status::kInternalError;
    info->iterations = 0;
    info->residual_l2 = INFINITY;
    item_controls[kActive] = n > 0 && max_iterations >= 0 && relative_tolerance > 0.0 &&
                                     relative_tolerance < 1.0 && isfinite(relative_tolerance)
                                 ? 1
                                 : 0;
    item_controls[kThreshold] = 0;
    item_controls[kRestart] = 0;
    item_controls[kVerified] = 0;
    if (!item_controls[kActive])
      info->status = Status::kInvalidArgument;
  }
  grid.sync();
  double invalid_local = 0.0;
  if (item_controls[kActive]) {
    for (int i = begin + lane; i < end; i += kPcgBlockThreads) {
      if (!isfinite(rhs[i]) || !isfinite(diagonal[i]) || !(diagonal[i] > 0.0))
        invalid_local = fmax(invalid_local, static_cast<double>(n - i));
    }
  }
  const double invalid_priority = CooperativeItemMax(
      invalid_local, reduction_scratch, partials, scalars, item, shard, shards_per_item, 0, grid);
  if (item_controls[kActive] && invalid_priority != 0.0 && shard == 0 && lane == 0) {
    const int first = n - static_cast<int>(invalid_priority);
    info->status = !isfinite(rhs[first]) || !isfinite(diagonal[first])
                       ? Status::kNumericalNonfinite
                       : Status::kLinearSolveBreakdown;
    item_controls[kActive] = 0;
  }
  grid.sync();

  if (item_controls[kActive]) {
    for (int i = begin + lane; i < end; i += kPcgBlockThreads)
      x[i] = 0.0;
  }
  grid.sync();
  double local_max = 0.0;
  if (item_controls[kActive]) {
    for (int i = begin + lane; i < end; i += kPcgBlockThreads)
      local_max = fmax(local_max, fabs(rhs[i]));
  }
  const double rhs_scale = CooperativeItemMax(local_max, reduction_scratch, partials, scalars, item,
                                              shard, shards_per_item, 0, grid);
  if (item_controls[kActive] && shard == 0 && lane == 0) {
    if (rhs_scale == 0.0) {
      info->residual_l2 = 0.0;
      info->status = Status::kOk;
      item_controls[kActive] = 0;
    }
  }
  grid.sync();

  if (item_controls[kActive]) {
    for (int i = begin + lane; i < end; i += kPcgBlockThreads) {
      residual[i] = rhs[i] / rhs_scale;
    }
  }
  if constexpr (kBlock)
    grid.sync();
  if (item_controls[kActive]) {
    for (int i = begin + lane; i < end; i += kPcgBlockThreads) {
      preconditioned[i] = PcgPreconditionRow<kUseInverseDiagonal, kBlock>(
          i, n, residual, diagonal, inverse_diagonal, block_inverse);
      direction[i] = preconditioned[i];
    }
  }
  grid.sync();
  double initial_residual_local = 0.0;
  double initial_rho_local = 0.0;
  if (item_controls[kActive]) {
    for (int i = begin + lane; i < end; i += kPcgBlockThreads) {
      initial_residual_local += residual[i] * residual[i];
      initial_rho_local += residual[i] * preconditioned[i];
    }
  }
  const double initial_residual_sq =
      CooperativeItemSum(initial_residual_local, reduction_scratch, partials, scalars, item, shard,
                         shards_per_item, 2, grid);
  const double initial_rho = CooperativeItemSum(initial_rho_local, reduction_scratch, partials,
                                                scalars, item, shard, shards_per_item, 1, grid);
  if (item_controls[kActive] && shard == 0 && lane == 0) {
    if (!isfinite(initial_residual_sq) || !isfinite(initial_rho)) {
      info->status = Status::kNumericalNonfinite;
      item_controls[kActive] = 0;
    } else {
      info->residual_l2 = sqrt(initial_residual_sq);
    }
  }
  grid.sync();
  const double rhs_norm = sqrt(initial_residual_sq);
  const double threshold = relative_tolerance * rhs_norm;

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    // A cooperative grid may only break on a decision observed by every
    // block.  Without this scan, an all-converged batch would keep executing
    // grid barriers through max_iterations.
    if (block == 0 && lane == 0) {
      int any_active = 0;
      for (int other = 0; other < item_count; ++other)
        any_active |= controls[4 * other + kActive];
      controls[global_any_index] = any_active;
    }
    grid.sync();
    if (!controls[global_any_index])
      break;
    // Each thread owns both K*p and its contribution to p'*K*p. No other
    // thread consumes matrix_direction until the collective sum has completed.
    double direction_matrix_local = 0.0;
    if (item_controls[kActive]) {
      for (int row = begin + lane; row < end; row += kPcgBlockThreads) {
        const double sum = PcgRowDot(row, row_offsets, column_indices, values, direction, slices);
        matrix_direction[row] = sum;
        direction_matrix_local += direction[row] * sum;
      }
    }
    const double direction_matrix_sum =
        CooperativeItemSum(direction_matrix_local, reduction_scratch, partials, scalars, item,
                           shard, shards_per_item, 2, grid);
    if (item_controls[kActive] && shard == 0 && lane == 0) {
      const double rho = scalars[static_cast<size_t>(item) * 3 + 1];
      if (!isfinite(direction_matrix_sum) || !isfinite(rho)) {
        info->status = Status::kNumericalNonfinite;
        item_controls[kActive] = 0;
      } else if (!(direction_matrix_sum > 0.0) || !(rho > 0.0)) {
        info->status = Status::kLinearSolveBreakdown;
        item_controls[kActive] = 0;
      }
    }
    grid.sync();
    double residual_local = 0.0;
    [[maybe_unused]] double updated_rho_local = 0.0;
    if (item_controls[kActive]) {
      const double alpha = scalars[static_cast<size_t>(item) * 3 + 1] / direction_matrix_sum;
      for (int i = begin + lane; i < end; i += kPcgBlockThreads) {
        x[i] += alpha * direction[i];
        residual[i] -= alpha * matrix_direction[i];
        residual_local += residual[i] * residual[i];
        if constexpr (!kBlock) {
          // Jacobi is row-local; no other lane's updated residual is needed.
          preconditioned[i] = PcgPreconditionRow<kUseInverseDiagonal, false>(
              i, n, residual, diagonal, inverse_diagonal, nullptr);
          updated_rho_local += residual[i] * preconditioned[i];
        }
      }
    }
    double residual_sum = 0.0, updated_rho = 0.0;
    if constexpr (kBlock) {
      residual_sum = CooperativeItemSum(residual_local, reduction_scratch, partials, scalars, item,
                                        shard, shards_per_item, 2, grid);
    } else {
      CooperativeItemSumPair(residual_local, updated_rho_local, reduction_scratch, partials,
                             scalars, item, shard, shards_per_item, item_count, residual_sum,
                             updated_rho, grid);
    }
    if (item_controls[kActive] && shard == 0 && lane == 0) {
      info->iterations = iteration + 1;
      if (!isfinite(residual_sum)) {
        info->status = Status::kNumericalNonfinite;
        item_controls[kActive] = 0;
      } else {
        info->residual_l2 = sqrt(residual_sum);
        if (!isfinite(info->residual_l2)) {
          info->status = Status::kNumericalNonfinite;
          item_controls[kActive] = 0;
        } else {
          item_controls[kThreshold] = info->residual_l2 <= threshold ? 1 : 0;
        }
      }
    }
    grid.sync();
    if (block == 0 && lane == 0) {
      int any_threshold = 0;
      for (int other = 0; other < item_count; ++other)
        any_threshold |= controls[4 * other + kThreshold];
      controls[global_any_index] = any_threshold;
    }
    grid.sync();
    const bool verifying_residual = controls[global_any_index] != 0;
    if (verifying_residual) {
      if (item_controls[kActive] && item_controls[kThreshold]) {
        for (int row = begin + lane; row < end; row += kPcgBlockThreads) {
          const double matrix_solution =
              PcgRowDot(row, row_offsets, column_indices, values, x, slices);
          residual[row] = rhs[row] / rhs_scale - matrix_solution;
        }
      }
      grid.sync();
      double true_residual_local = 0.0;
      if (item_controls[kActive] && item_controls[kThreshold]) {
        for (int i = begin + lane; i < end; i += kPcgBlockThreads)
          true_residual_local += residual[i] * residual[i];
      }
      const double true_residual_sum =
          CooperativeItemSum(true_residual_local, reduction_scratch, partials, scalars, item, shard,
                             shards_per_item, 2, grid);
      if (item_controls[kActive] && item_controls[kThreshold] && shard == 0 && lane == 0) {
        if (!isfinite(true_residual_sum)) {
          info->status = Status::kNumericalNonfinite;
          item_controls[kActive] = 0;
        } else {
          info->residual_l2 = sqrt(true_residual_sum);
          item_controls[kVerified] = info->residual_l2 <= threshold ? 1 : 0;
          item_controls[kRestart] = item_controls[kVerified] ? 0 : 1;
        }
      }
      grid.sync();
      if (item_controls[kActive] && item_controls[kVerified]) {
        for (int i = begin + lane; i < end; i += kPcgBlockThreads)
          x[i] *= rhs_scale;
      }
      grid.sync();
      double nonfinite_local = 0.0;
      if (item_controls[kActive] && item_controls[kVerified]) {
        for (int i = begin + lane; i < end; i += kPcgBlockThreads)
          nonfinite_local = fmax(nonfinite_local, isfinite(x[i]) ? 0.0 : 1.0);
      }
      const double nonfinite = CooperativeItemMax(nonfinite_local, reduction_scratch, partials,
                                                  scalars, item, shard, shards_per_item, 2, grid);
      if (item_controls[kActive] && item_controls[kVerified] && shard == 0 && lane == 0) {
        if (nonfinite != 0.0) {
          info->status = Status::kNumericalNonfinite;
        } else {
          info->residual_l2 *= rhs_scale;
          info->status = isfinite(info->residual_l2) ? Status::kOk : Status::kNumericalNonfinite;
        }
        item_controls[kActive] = 0;
      }
      grid.sync();
    }

    double new_rho = updated_rho;
    // Verification may replace r with the true Ax-b residual. Never reuse
    // the recursive-residual preconditioner on that restart path. This
    // condition is uniform across the grid, including failed/finished items.
    if (kBlock || verifying_residual) {
      double preconditioned_local = 0.0;
      if (item_controls[kActive]) {
        for (int i = begin + lane; i < end; i += kPcgBlockThreads) {
          preconditioned[i] = PcgPreconditionRow<kUseInverseDiagonal, kBlock>(
              i, n, residual, diagonal, inverse_diagonal, block_inverse);
          if (item_controls[kRestart])
            direction[i] = preconditioned[i];
          preconditioned_local += residual[i] * preconditioned[i];
        }
      }
      new_rho = CooperativeItemSum(preconditioned_local, reduction_scratch, partials, scalars, item,
                                   shard, shards_per_item, 2, grid);
    }
    if (item_controls[kActive] && shard == 0 && lane == 0) {
      const double old_rho = scalars[static_cast<size_t>(item) * 3 + 1];
      if (!isfinite(new_rho) || !(new_rho > 0.0)) {
        info->status = Status::kLinearSolveBreakdown;
        item_controls[kActive] = 0;
      } else {
        if (!item_controls[kRestart])
          scalars[static_cast<size_t>(item) * 3 + 2] = new_rho / old_rho;
        scalars[static_cast<size_t>(item) * 3 + 1] = new_rho;
      }
    }
    grid.sync();
    if (item_controls[kActive] && !item_controls[kRestart]) {
      const double beta = scalars[static_cast<size_t>(item) * 3 + 2];
      for (int i = begin + lane; i < end; i += kPcgBlockThreads)
        direction[i] = preconditioned[i] + beta * direction[i];
    }
    grid.sync();
    if (shard == 0 && lane == 0) {
      item_controls[kThreshold] = 0;
      item_controls[kRestart] = 0;
      item_controls[kVerified] = 0;
    }
    grid.sync();
  }

  if (item_controls[kActive]) {
    for (int row = begin + lane; row < end; row += kPcgBlockThreads) {
      const double matrix_solution = PcgRowDot(row, row_offsets, column_indices, values, x, slices);
      residual[row] = rhs[row] / rhs_scale - matrix_solution;
    }
  }
  grid.sync();
  double final_residual_local = 0.0;
  if (item_controls[kActive]) {
    for (int i = begin + lane; i < end; i += kPcgBlockThreads)
      final_residual_local += residual[i] * residual[i];
  }
  const double final_residual_sum =
      CooperativeItemSum(final_residual_local, reduction_scratch, partials, scalars, item, shard,
                         shards_per_item, 2, grid);
  if (item_controls[kActive] && shard == 0 && lane == 0 && !isfinite(final_residual_sum)) {
    info->status = Status::kNumericalNonfinite;
    item_controls[kActive] = 0;
  }
  grid.sync();
  if (item_controls[kActive]) {
    for (int i = begin + lane; i < end; i += kPcgBlockThreads)
      x[i] *= rhs_scale;
  }
  grid.sync();
  double final_nonfinite_local = 0.0;
  if (item_controls[kActive]) {
    for (int i = begin + lane; i < end; i += kPcgBlockThreads)
      final_nonfinite_local = fmax(final_nonfinite_local, isfinite(x[i]) ? 0.0 : 1.0);
  }
  const double final_nonfinite =
      CooperativeItemMax(final_nonfinite_local, reduction_scratch, partials, scalars, item, shard,
                         shards_per_item, 2, grid);
  if (item_controls[kActive] && shard == 0 && lane == 0) {
    if (final_nonfinite != 0.0) {
      info->status = Status::kNumericalNonfinite;
    } else {
      info->residual_l2 = sqrt(final_residual_sum) * rhs_scale;
      info->status = isfinite(info->residual_l2) ? Status::kLinearSolveNotConverged
                                                 : Status::kNumericalNonfinite;
    }
    item_controls[kActive] = 0;
  }
}

template <bool kUseInverseDiagonal>
void LaunchDeterministicPcg(unsigned count, const double *blocks, const int *owners,
                            const int32_t *slices, int n, const int32_t *row_offsets,
                            const int32_t *column_indices, const double *values,
                            const double *diagonal, const double *inverse_diagonal,
                            const double *rhs, double *x, double *residual, double *direction,
                            double *preconditioned, double *matrix_direction,
                            double relative_tolerance, int max_iterations, SolveInfo *info,
                            int values_per_item)
{
  if (blocks != nullptr)
    DeterministicPcgKernel<kUseInverseDiagonal, true><<<count, kPcgBlockThreads>>>(
        n, row_offsets, column_indices, values, diagonal, inverse_diagonal, rhs, x, residual,
        direction, preconditioned, matrix_direction, relative_tolerance, max_iterations, info,
        values_per_item, blocks, owners, slices);
  else
    DeterministicPcgKernel<kUseInverseDiagonal><<<count, kPcgBlockThreads>>>(
        n, row_offsets, column_indices, values, diagonal, inverse_diagonal, rhs, x, residual,
        direction, preconditioned, matrix_direction, relative_tolerance, max_iterations, info,
        values_per_item, blocks, owners, slices);
}

template void LaunchDeterministicPcg<true>(unsigned, const double *, const int *, const int32_t *,
                                           int, const int32_t *, const int32_t *, const double *,
                                           const double *, const double *, const double *, double *,
                                           double *, double *, double *, double *, double, int,
                                           SolveInfo *, int);
template void LaunchDeterministicPcg<false>(unsigned, const double *, const int *, const int32_t *,
                                            int, const int32_t *, const int32_t *, const double *,
                                            const double *, const double *, const double *,
                                            double *, double *, double *, double *, double *,
                                            double, int, SolveInfo *, int);

const void *CooperativePcgKernel(bool use_inverse_diagonal, bool block_preconditioner)
{
  return block_preconditioner
             ? (use_inverse_diagonal
                    ? reinterpret_cast<void *>(CooperativeDeterministicPcgKernel<true, true>)
                    : reinterpret_cast<void *>(CooperativeDeterministicPcgKernel<false, true>))
             : (use_inverse_diagonal
                    ? reinterpret_cast<void *>(CooperativeDeterministicPcgKernel<true>)
                    : reinterpret_cast<void *>(CooperativeDeterministicPcgKernel<false>));
}

} // namespace gpu_femm

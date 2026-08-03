#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "native_airgap_matrix.h"

// Legacy CUDA/MSVC compatibility headers reserve these otherwise ordinary
// FEMM local-variable names.  They are not used as macros in this source.
#undef element
#undef near
#undef far

namespace gpu_femm {

namespace cg = cooperative_groups;

using ProfileClock = std::chrono::steady_clock;

double ProfileSecondsSince(const ProfileClock::time_point& start)
{
  return std::chrono::duration<double>(ProfileClock::now() - start).count();
}

enum class Status : int {
  kOk = 0,
  kInputIo = 1,
  kInvalidArgument = 2,
  kUnsupportedFeature = 3,
  kMeshInvalid = 4,
  kBoundaryInvalid = 5,
  kGpuUnavailable = 6,
  kGpuAllocationFailed = 7,
  kAssemblyFailed = 8,
  kLinearSolveNotConverged = 9,
  kLinearSolveBreakdown = 10,
  kNumericalNonfinite = 11,
  kOutputIo = 12,
  kInternalError = 13,
  kNonlinearSolveNotConverged = 14,
  kInvalidMaterial = 15,
};

const char* StatusName(Status status)
{
  switch (status) {
  case Status::kOk:
    return "OK";
  case Status::kInputIo:
    return "INPUT_IO";
  case Status::kInvalidArgument:
    return "INVALID_ARGUMENT";
  case Status::kUnsupportedFeature:
    return "UNSUPPORTED_FEATURE";
  case Status::kMeshInvalid:
    return "MESH_INVALID";
  case Status::kBoundaryInvalid:
    return "BOUNDARY_INVALID";
  case Status::kGpuUnavailable:
    return "GPU_UNAVAILABLE";
  case Status::kGpuAllocationFailed:
    return "GPU_ALLOCATION_FAILED";
  case Status::kAssemblyFailed:
    return "ASSEMBLY_FAILED";
  case Status::kLinearSolveNotConverged:
    return "LINEAR_SOLVE_NOT_CONVERGED";
  case Status::kLinearSolveBreakdown:
    return "LINEAR_SOLVE_BREAKDOWN";
  case Status::kNumericalNonfinite:
    return "NUMERICAL_NONFINITE";
  case Status::kOutputIo:
    return "OUTPUT_IO";
  case Status::kInternalError:
    return "INTERNAL_ERROR";
  case Status::kNonlinearSolveNotConverged:
    return "NONLINEAR_SOLVE_NOT_CONVERGED";
  case Status::kInvalidMaterial:
    return "INVALID_MATERIAL";
  }
  return "INTERNAL_ERROR";
}

struct Node {
  double x_m;
  double y_m;
};

struct Triangle {
  int32_t node[3];
  double reluctivity_m_per_h;
  double source_j_per_a;
};

struct Model {
  std::vector<Node> nodes;
  std::vector<Triangle> triangles;
  std::vector<int32_t> dirichlet_nodes;
  std::vector<double> dirichlet_a_wb_per_m;
  double depth_m = 0.0;
};

struct SolveInfo {
  Status status = Status::kInternalError;
  int iterations = 0;
  double residual_l2 = std::numeric_limits<double>::infinity();
};

struct SolveResult {
  SolveInfo info;
  std::vector<double> a_wb_per_m;
  std::vector<double> bx_t;
  std::vector<double> by_t;
  double flux_linkage_wb = std::numeric_limits<double>::quiet_NaN();
};

template <typename T>
class DeviceBuffer {
  public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  Status allocate(size_t count)
  {
    reset();
    count_ = count;
    if (count == 0)
      return Status::kOk;
    if (cudaMalloc(&ptr_, count * sizeof(T)) != cudaSuccess) {
      ptr_ = nullptr;
      count_ = 0;
      return Status::kGpuAllocationFailed;
    }
    return Status::kOk;
  }

  void reset()
  {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
    ptr_ = nullptr;
    count_ = 0;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t count() const { return count_; }

  private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

Status CopyToDevice(void* destination, const void* source, size_t bytes)
{
  if (bytes == 0)
    return Status::kOk;
  return cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) == cudaSuccess
      ? Status::kOk
      : Status::kInternalError;
}

Status CopyToHost(void* destination, const void* source, size_t bytes)
{
  if (bytes == 0)
    return Status::kOk;
  return cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost) == cudaSuccess
      ? Status::kOk
      : Status::kInternalError;
}

// Keep the original diagonal on device for the kernel's established
// finite/positive validation.  This auxiliary buffer only removes repeated
// Jacobi divisions when every reciprocal is representable as a normal finite
// value; unusual diagonals retain the original division path.
bool BuildJacobiInverseDiagonal(const std::vector<double>& diagonal,
    std::vector<double>* inverse_diagonal)
{
  if (inverse_diagonal == nullptr) return false;
  inverse_diagonal->resize(diagonal.size());
  bool usable = true;
  for (size_t index = 0; index < diagonal.size(); ++index) {
    const double value = diagonal[index];
    if (!std::isfinite(value) || !(value > 0.0)) {
      (*inverse_diagonal)[index] = 0.0;
      usable = false;
      continue;
    }
    const double inverse = 1.0 / value;
    if (!std::isnormal(inverse)) {
      (*inverse_diagonal)[index] = 0.0;
      usable = false;
      continue;
    }
    (*inverse_diagonal)[index] = inverse;
  }
  return usable;
}

// One fixed block keeps every CSR-row ownership fixed from solve to solve.
// Row/vector work and scalar reductions are parallel.  The fixed tree avoids
// atomics and keeps one deterministic reduction order for every batch width.
constexpr int kPcgBlockThreads = 512;
constexpr int kPcgCooperativeShardsPerItem = 8;

// Every lane owns the same strided index sequence for every batch width.  The
// fixed pairwise tree is deterministic; its warp tail preserves the same
// pairing while avoiding block-wide barriers once only one warp remains.
__device__ double DeterministicBlockSum(double local, double* scratch)
{
  const int lane = threadIdx.x;
  scratch[lane] = local;
  __syncthreads();
  for (int stride = kPcgBlockThreads / 2; stride >= 64; stride /= 2) {
    if (lane < stride) scratch[lane] += scratch[lane + stride];
    __syncthreads();
  }
  if (lane < warpSize) {
    double value = scratch[lane] + scratch[lane + warpSize];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      value += __shfl_down_sync(0xffffffffu, value, offset);
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

__device__ double DeterministicBlockMax(double local, double* scratch)
{
  const int lane = threadIdx.x;
  scratch[lane] = local;
  __syncthreads();
  for (int stride = kPcgBlockThreads / 2; stride >= 64; stride /= 2) {
    if (lane < stride) scratch[lane] = fmax(scratch[lane], scratch[lane + stride]);
    __syncthreads();
  }
  if (lane < warpSize) {
    double value = fmax(scratch[lane], scratch[lane + warpSize]);
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      value = fmax(value, __shfl_down_sync(0xffffffffu, value, offset));
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

template <bool kUseInverseDiagonal>
__global__ void DeterministicPcgKernel(
    int n, const int32_t* row_offsets, const int32_t* column_indices,
    const double* values, const double* diagonal, const double* inverse_diagonal,
    const double* rhs, double* x,
    double* residual, double* direction, double* preconditioned,
    double* matrix_direction, double relative_tolerance, int max_iterations,
    SolveInfo* info, int values_per_item)
{
  // One fixed-width block owns one independent current-state.  B=1 and B>1
  // use the same fixed reduction tree; sums may differ slightly from the
  // former lane-zero order but are deterministic.
  const int item = blockIdx.x;
  values += static_cast<size_t>(item) * values_per_item;
  diagonal += static_cast<size_t>(item) * n;
  inverse_diagonal += static_cast<size_t>(item) * n;
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
    if (n <= 0 || max_iterations < 0 || !(relative_tolerance > 0.0)
        || !(relative_tolerance < 1.0) || !isfinite(relative_tolerance)) {
      info->status = Status::kInvalidArgument;
      stop = 1;
    }
  }
  __syncthreads();
  if (stop)
    return;

  if (lane == 0) {
    for (int i = 0; i < n; ++i) {
      if (!isfinite(rhs[i]) || !isfinite(diagonal[i])) {
        info->status = Status::kNumericalNonfinite;
        stop = 1;
        break;
      }
      if (!(diagonal[i] > 0.0)) {
        info->status = Status::kLinearSolveBreakdown;
        stop = 1;
        break;
      }
    }
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
    if constexpr (kUseInverseDiagonal)
      preconditioned[i] = residual[i] * inverse_diagonal[i];
    else
      preconditioned[i] = residual[i] / diagonal[i];
    direction[i] = preconditioned[i];
  }
  __syncthreads();
  double initial_residual_local = 0.0;
  double initial_rho_local = 0.0;
  for (int i = lane; i < n; i += kPcgBlockThreads) {
    initial_residual_local += residual[i] * residual[i];
    initial_rho_local += residual[i] * preconditioned[i];
  }
  const double initial_residual_sq = DeterministicBlockSum(initial_residual_local, reduction_scratch);
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
    for (int row = lane; row < n; row += kPcgBlockThreads) {
      double sum = 0.0;
      for (int32_t entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry) {
        sum += values[entry] * direction[column_indices[entry]];
      }
      matrix_direction[row] = sum;
    }
    __syncthreads();

    double direction_matrix_local = 0.0;
    for (int i = lane; i < n; i += kPcgBlockThreads)
      direction_matrix_local += direction[i] * matrix_direction[i];
    const double direction_matrix_sum = DeterministicBlockSum(direction_matrix_local, reduction_scratch);
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
    for (int i = lane; i < n; i += kPcgBlockThreads) {
      x[i] += alpha * direction[i];
      residual[i] -= alpha * matrix_direction[i];
    }
    __syncthreads();
    double residual_local = 0.0;
    for (int i = lane; i < n; i += kPcgBlockThreads)
      residual_local += residual[i] * residual[i];
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
        double matrix_solution = 0.0;
        for (int32_t entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry)
          matrix_solution += values[entry] * x[column_indices[entry]];
        residual[row] = rhs[row] / rhs_scale - matrix_solution;
      }
      __syncthreads();
      double true_residual_local = 0.0;
      for (int i = lane; i < n; i += kPcgBlockThreads)
        true_residual_local += residual[i] * residual[i];
      const double true_residual_sum = DeterministicBlockSum(true_residual_local, reduction_scratch);
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
        if constexpr (kUseInverseDiagonal)
          preconditioned[i] = residual[i] * inverse_diagonal[i];
        else
          preconditioned[i] = residual[i] / diagonal[i];
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

    for (int i = lane; i < n; i += kPcgBlockThreads) {
      if constexpr (kUseInverseDiagonal)
        preconditioned[i] = residual[i] * inverse_diagonal[i];
      else
        preconditioned[i] = residual[i] / diagonal[i];
    }
    __syncthreads();
    double preconditioned_local = 0.0;
    for (int i = lane; i < n; i += kPcgBlockThreads)
      preconditioned_local += residual[i] * preconditioned[i];
    const double preconditioned_sum = DeterministicBlockSum(preconditioned_local, reduction_scratch);
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
    double matrix_solution = 0.0;
    for (int32_t entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry) {
      matrix_solution += values[entry] * x[column_indices[entry]];
    }
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
__device__ double CooperativeItemSum(double local, double* reduction_scratch,
    double* partials, double* scalars, int item, int shard, int shard_count,
    int scalar_slot, cg::grid_group grid)
{
  const double block_sum = DeterministicBlockSum(local, reduction_scratch);
  if (threadIdx.x == 0)
    partials[static_cast<size_t>(item) * shard_count + shard] =
        block_sum;
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

__device__ double CooperativeItemMax(double local, double* reduction_scratch,
    double* partials, double* scalars, int item, int shard, int shard_count,
    int scalar_slot, cg::grid_group grid)
{
  const double block_max = DeterministicBlockMax(local, reduction_scratch);
  if (threadIdx.x == 0)
    partials[static_cast<size_t>(item) * shard_count + shard] =
        block_max;
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

template <bool kUseInverseDiagonal>
__global__ void CooperativeDeterministicPcgKernel(
    int n, const int32_t* row_offsets, const int32_t* column_indices,
    const double* values, const double* diagonal, const double* inverse_diagonal,
    const double* rhs, double* x, double* residual, double* direction,
    double* preconditioned, double* matrix_direction, double relative_tolerance,
    int max_iterations, SolveInfo* info, int values_per_item, int item_count,
    int shards_per_item, double* partials, double* scalars, int* controls)
{
  cg::grid_group grid = cg::this_grid();
  const int block = static_cast<int>(blockIdx.x);
  const int item = block / shards_per_item;
  const int shard = block % shards_per_item;
  const int lane = threadIdx.x;
  const int begin = static_cast<int>(static_cast<int64_t>(n) * shard / shards_per_item);
  const int end = static_cast<int>(static_cast<int64_t>(n) * (shard + 1) / shards_per_item);
  constexpr int kActive = 0;
  constexpr int kThreshold = 1;
  constexpr int kRestart = 2;
  constexpr int kVerified = 3;
  int* item_controls = controls + static_cast<size_t>(item) * 4;
  const int global_any_index = 4 * item_count;
  values += static_cast<size_t>(item) * values_per_item;
  diagonal += static_cast<size_t>(item) * n;
  inverse_diagonal += static_cast<size_t>(item) * n;
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
    item_controls[kActive] = n > 0 && max_iterations >= 0
            && relative_tolerance > 0.0 && relative_tolerance < 1.0
            && isfinite(relative_tolerance) ? 1 : 0;
    item_controls[kThreshold] = 0;
    item_controls[kRestart] = 0;
    item_controls[kVerified] = 0;
    if (!item_controls[kActive]) info->status = Status::kInvalidArgument;
  }
  grid.sync();
  // Preserve the established diagonal validation before using an optional
  // reciprocal buffer.  One leader performs this only once per item.
  if (item_controls[kActive] && shard == 0 && lane == 0) {
    for (int i = 0; i < n; ++i) {
      if (!isfinite(rhs[i]) || !isfinite(diagonal[i])) {
        info->status = Status::kNumericalNonfinite;
        item_controls[kActive] = 0;
        break;
      }
      if (!(diagonal[i] > 0.0)) {
        info->status = Status::kLinearSolveBreakdown;
        item_controls[kActive] = 0;
        break;
      }
    }
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
  const double rhs_scale = CooperativeItemMax(local_max, reduction_scratch,
      partials, scalars, item, shard, shards_per_item, 0, grid);
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
      if constexpr (kUseInverseDiagonal)
        preconditioned[i] = residual[i] * inverse_diagonal[i];
      else
        preconditioned[i] = residual[i] / diagonal[i];
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
  const double initial_residual_sq = CooperativeItemSum(initial_residual_local,
      reduction_scratch, partials, scalars, item, shard, shards_per_item, 2, grid);
  const double initial_rho = CooperativeItemSum(initial_rho_local, reduction_scratch,
      partials, scalars, item, shard, shards_per_item, 1, grid);
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
    if (!controls[global_any_index]) break;
    if (item_controls[kActive]) {
      for (int row = begin + lane; row < end; row += kPcgBlockThreads) {
        double sum = 0.0;
        for (int32_t entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry)
          sum += values[entry] * direction[column_indices[entry]];
        matrix_direction[row] = sum;
      }
    }
    grid.sync();
    double direction_matrix_local = 0.0;
    if (item_controls[kActive]) {
      for (int i = begin + lane; i < end; i += kPcgBlockThreads)
        direction_matrix_local += direction[i] * matrix_direction[i];
    }
    const double direction_matrix_sum = CooperativeItemSum(direction_matrix_local,
        reduction_scratch, partials, scalars, item, shard, shards_per_item, 2, grid);
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
    if (item_controls[kActive]) {
      const double alpha = scalars[static_cast<size_t>(item) * 3 + 1]
          / direction_matrix_sum;
      for (int i = begin + lane; i < end; i += kPcgBlockThreads) {
        x[i] += alpha * direction[i];
        residual[i] -= alpha * matrix_direction[i];
      }
    }
    grid.sync();
    double residual_local = 0.0;
    if (item_controls[kActive]) {
      for (int i = begin + lane; i < end; i += kPcgBlockThreads)
        residual_local += residual[i] * residual[i];
    }
    const double residual_sum = CooperativeItemSum(residual_local, reduction_scratch,
        partials, scalars, item, shard, shards_per_item, 2, grid);
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
    if (controls[global_any_index]) {
      if (item_controls[kActive] && item_controls[kThreshold]) {
        for (int row = begin + lane; row < end; row += kPcgBlockThreads) {
          double matrix_solution = 0.0;
          for (int32_t entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry)
            matrix_solution += values[entry] * x[column_indices[entry]];
          residual[row] = rhs[row] / rhs_scale - matrix_solution;
        }
      }
      grid.sync();
      double true_residual_local = 0.0;
      if (item_controls[kActive] && item_controls[kThreshold]) {
        for (int i = begin + lane; i < end; i += kPcgBlockThreads)
          true_residual_local += residual[i] * residual[i];
      }
      const double true_residual_sum = CooperativeItemSum(true_residual_local,
          reduction_scratch, partials, scalars, item, shard, shards_per_item, 2, grid);
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
      const double nonfinite = CooperativeItemMax(nonfinite_local, reduction_scratch,
          partials, scalars, item, shard, shards_per_item, 2, grid);
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

    if (item_controls[kActive]) {
      for (int i = begin + lane; i < end; i += kPcgBlockThreads) {
        if constexpr (kUseInverseDiagonal)
          preconditioned[i] = residual[i] * inverse_diagonal[i];
        else
          preconditioned[i] = residual[i] / diagonal[i];
        if (item_controls[kRestart]) direction[i] = preconditioned[i];
      }
    }
    grid.sync();
    double preconditioned_local = 0.0;
    if (item_controls[kActive]) {
      for (int i = begin + lane; i < end; i += kPcgBlockThreads)
        preconditioned_local += residual[i] * preconditioned[i];
    }
    const double new_rho = CooperativeItemSum(preconditioned_local, reduction_scratch,
        partials, scalars, item, shard, shards_per_item, 2, grid);
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
      double matrix_solution = 0.0;
      for (int32_t entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry)
        matrix_solution += values[entry] * x[column_indices[entry]];
      residual[row] = rhs[row] / rhs_scale - matrix_solution;
    }
  }
  grid.sync();
  double final_residual_local = 0.0;
  if (item_controls[kActive]) {
    for (int i = begin + lane; i < end; i += kPcgBlockThreads)
      final_residual_local += residual[i] * residual[i];
  }
  const double final_residual_sum = CooperativeItemSum(final_residual_local,
      reduction_scratch, partials, scalars, item, shard, shards_per_item, 2, grid);
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
  const double final_nonfinite = CooperativeItemMax(final_nonfinite_local,
      reduction_scratch, partials, scalars, item, shard, shards_per_item, 2, grid);
  if (item_controls[kActive] && shard == 0 && lane == 0) {
    if (final_nonfinite != 0.0) {
      info->status = Status::kNumericalNonfinite;
    } else {
      info->residual_l2 = sqrt(final_residual_sum) * rhs_scale;
      info->status = isfinite(info->residual_l2)
          ? Status::kLinearSolveNotConverged : Status::kNumericalNonfinite;
    }
    item_controls[kActive] = 0;
  }
}

__global__ void ComputeFieldKernel(
    int triangle_count, const Node* nodes, const Triangle* triangles,
    const double* nodal_a, double* bx, double* by)
{
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= triangle_count)
    return;
  const Triangle triangle = triangles[element];
  const Node p0 = nodes[triangle.node[0]];
  const Node p1 = nodes[triangle.node[1]];
  const Node p2 = nodes[triangle.node[2]];
  const double determinant = (p1.x_m - p0.x_m) * (p2.y_m - p0.y_m) - (p2.x_m - p0.x_m) * (p1.y_m - p0.y_m);
  const double b0 = p1.y_m - p2.y_m;
  const double b1 = p2.y_m - p0.y_m;
  const double b2 = p0.y_m - p1.y_m;
  const double c0 = p2.x_m - p1.x_m;
  const double c1 = p0.x_m - p2.x_m;
  const double c2 = p1.x_m - p0.x_m;
  const double d_a_dx = (nodal_a[triangle.node[0]] * b0 + nodal_a[triangle.node[1]] * b1 + nodal_a[triangle.node[2]] * b2) / determinant;
  const double d_a_dy = (nodal_a[triangle.node[0]] * c0 + nodal_a[triangle.node[1]] * c1 + nodal_a[triangle.node[2]] * c2) / determinant;
  bx[element] = d_a_dy;
  by[element] = -d_a_dx;
}

__global__ void ComputeFluxKernel(
    int node_count, const double* source_load_per_amp, const double* nodal_a,
    double depth_m, double* flux_linkage)
{
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  double sum = 0.0;
  for (int i = 0; i < node_count; ++i)
    sum += source_load_per_amp[i] * nodal_a[i];
  *flux_linkage = depth_m * sum;
}

struct Assembly {
  std::vector<int32_t> row_offsets;
  std::vector<int32_t> column_indices;
  std::vector<double> values;
  std::vector<double> diagonal;
  std::vector<double> rhs_per_amp;
  std::vector<double> rhs_offset;
  std::vector<double> source_load_per_amp;
  // Nonlinear path only: source vectors are indexed by NonlinearMaterial::circuit_index.
  std::vector<std::vector<double>> source_load_per_circuit;
  std::vector<int32_t> free_nodes;
  std::vector<double> boundary_values;
};

// Element traversal is deliberately the accumulation order: map insertion
// keeps each coefficient's floating-point sum deterministic while avoiding an
// O(node_count^2) host matrix for the intrinsically sparse P1 stencil.
using SparseRows = std::vector<std::map<int32_t, double>>;

Status AddSparseEntry(SparseRows* rows, int32_t row, int32_t column, double value)
{
  if (rows == nullptr || row < 0 || column < 0
      || static_cast<size_t>(row) >= rows->size()
      || static_cast<size_t>(column) >= rows->size() || !std::isfinite(value))
    return Status::kNumericalNonfinite;
  double& accumulated = (*rows)[row][column];
  accumulated += value;
  return std::isfinite(accumulated) ? Status::kOk : Status::kNumericalNonfinite;
}

Status ValidateSparseRows(const SparseRows& rows)
{
  for (const auto& row : rows) {
    for (const auto& entry : row) {
      if (!std::isfinite(entry.second))
        return Status::kNumericalNonfinite;
    }
  }
  return Status::kOk;
}

Status ValidateModel(const Model& model)
{
  if (model.nodes.empty() || model.triangles.empty() || !(model.depth_m > 0.0) || !std::isfinite(model.depth_m)) {
    return Status::kInvalidArgument;
  }
  if (model.dirichlet_nodes.empty() || model.dirichlet_nodes.size() != model.dirichlet_a_wb_per_m.size()) {
    return Status::kBoundaryInvalid;
  }
  std::vector<bool> is_boundary(model.nodes.size(), false);
  for (size_t i = 0; i < model.dirichlet_nodes.size(); ++i) {
    const int32_t node = model.dirichlet_nodes[i];
    if (node < 0 || static_cast<size_t>(node) >= model.nodes.size() || is_boundary[node] || !std::isfinite(model.dirichlet_a_wb_per_m[i])) {
      return Status::kBoundaryInvalid;
    }
    is_boundary[node] = true;
  }
  if (model.dirichlet_nodes.size() == model.nodes.size())
    return Status::kBoundaryInvalid;

  for (const Node& node : model.nodes) {
    if (!std::isfinite(node.x_m) || !std::isfinite(node.y_m)) {
      return Status::kMeshInvalid;
    }
  }
  for (const Triangle& triangle : model.triangles) {
    for (int local = 0; local < 3; ++local) {
      if (triangle.node[local] < 0 || static_cast<size_t>(triangle.node[local]) >= model.nodes.size()) {
        return Status::kMeshInvalid;
      }
    }
    const Node p0 = model.nodes[triangle.node[0]];
    const Node p1 = model.nodes[triangle.node[1]];
    const Node p2 = model.nodes[triangle.node[2]];
    const double determinant = (p1.x_m - p0.x_m) * (p2.y_m - p0.y_m) - (p2.x_m - p0.x_m) * (p1.y_m - p0.y_m);
    if (!(determinant > 0.0) || !std::isfinite(determinant) || !(triangle.reluctivity_m_per_h > 0.0) || !std::isfinite(triangle.reluctivity_m_per_h) || !std::isfinite(triangle.source_j_per_a)) {
      return Status::kMeshInvalid;
    }
  }

  std::vector<std::vector<int32_t>> adjacency(model.nodes.size());
  for (const Triangle& triangle : model.triangles) {
    for (int edge = 0; edge < 3; ++edge) {
      const int32_t first = triangle.node[edge];
      const int32_t second = triangle.node[(edge + 1) % 3];
      adjacency[first].push_back(second);
      adjacency[second].push_back(first);
    }
  }
  std::vector<bool> visited(model.nodes.size(), false);
  for (size_t start = 0; start < model.nodes.size(); ++start) {
    if (visited[start])
      continue;
    bool component_has_boundary = false;
    std::vector<int32_t> pending = { static_cast<int32_t>(start) };
    visited[start] = true;
    while (!pending.empty()) {
      const int32_t node = pending.back();
      pending.pop_back();
      component_has_boundary = component_has_boundary || is_boundary[node];
      for (const int32_t neighbor : adjacency[node]) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          pending.push_back(neighbor);
        }
      }
    }
    if (!component_has_boundary)
      return Status::kBoundaryInvalid;
  }
  return Status::kOk;
}

Status Assemble(const Model& model, Assembly* assembly)
{
  const Status validation = ValidateModel(model);
  if (validation != Status::kOk)
    return validation;
  const size_t node_count = model.nodes.size();
  SparseRows stiffness(node_count);
  assembly->source_load_per_amp.assign(node_count, 0.0);

  for (const Triangle& triangle : model.triangles) {
    const Node p[3] = { model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
      model.nodes[triangle.node[2]] };
    const double determinant = (p[1].x_m - p[0].x_m) * (p[2].y_m - p[0].y_m) - (p[2].x_m - p[0].x_m) * (p[1].y_m - p[0].y_m);
    const double area = 0.5 * determinant;
    const double b[3] = { p[1].y_m - p[2].y_m, p[2].y_m - p[0].y_m,
      p[0].y_m - p[1].y_m };
    const double c[3] = { p[2].x_m - p[1].x_m, p[0].x_m - p[2].x_m,
      p[1].x_m - p[0].x_m };
    for (int i = 0; i < 3; ++i) {
      const int global_i = triangle.node[i];
      assembly->source_load_per_amp[global_i] += triangle.source_j_per_a * area / 3.0;
      for (int j = 0; j < 3; ++j) {
        const int global_j = triangle.node[j];
        const Status added = AddSparseEntry(&stiffness, global_i, global_j,
            triangle.reluctivity_m_per_h * (b[i] * b[j] + c[i] * c[j]) / (4.0 * area));
        if (added != Status::kOk)
          return added;
      }
    }
  }
  const Status sparse_validation = ValidateSparseRows(stiffness);
  if (sparse_validation != Status::kOk)
    return sparse_validation;
  for (const double value : assembly->source_load_per_amp) {
    if (!std::isfinite(value))
      return Status::kNumericalNonfinite;
  }

  std::vector<bool> is_boundary(node_count, false);
  assembly->boundary_values.assign(node_count, 0.0);
  for (size_t i = 0; i < model.dirichlet_nodes.size(); ++i) {
    is_boundary[model.dirichlet_nodes[i]] = true;
    assembly->boundary_values[model.dirichlet_nodes[i]] = model.dirichlet_a_wb_per_m[i];
  }
  assembly->free_nodes.clear();
  std::vector<int32_t> free_index(node_count, -1);
  for (size_t node = 0; node < node_count; ++node) {
    if (!is_boundary[node]) {
      free_index[node] = static_cast<int32_t>(assembly->free_nodes.size());
      assembly->free_nodes.push_back(static_cast<int32_t>(node));
    }
  }

  const size_t free_count = assembly->free_nodes.size();
  assembly->row_offsets.assign(free_count + 1, 0);
  assembly->column_indices.clear();
  assembly->values.clear();
  assembly->diagonal.assign(free_count, 0.0);
  assembly->rhs_per_amp.assign(free_count, 0.0);
  assembly->rhs_offset.assign(free_count, 0.0);

  for (size_t row = 0; row < free_count; ++row) {
    const int global_row = assembly->free_nodes[row];
    assembly->row_offsets[row] = static_cast<int32_t>(assembly->values.size());
    assembly->rhs_per_amp[row] = assembly->source_load_per_amp[global_row];
    for (const auto& entry : stiffness[global_row]) {
      const int32_t global_column = entry.first;
      const double value = entry.second;
      if (is_boundary[global_column]) {
        assembly->rhs_offset[row] -= value * assembly->boundary_values[global_column];
      } else if (value != 0.0) {
        const int32_t column = free_index[global_column];
        if (column < 0)
          return Status::kAssemblyFailed;
        assembly->column_indices.push_back(column);
        assembly->values.push_back(value);
        if (static_cast<size_t>(column) == row)
          assembly->diagonal[row] = value;
      }
    }
    if (!std::isfinite(assembly->rhs_per_amp[row]) || !std::isfinite(assembly->rhs_offset[row]))
      return Status::kNumericalNonfinite;
    if (!(assembly->diagonal[row] > 0.0) || !std::isfinite(assembly->diagonal[row])) {
      return Status::kAssemblyFailed;
    }
  }
  assembly->row_offsets[free_count] = static_cast<int32_t>(assembly->values.size());
  return Status::kOk;
}

struct GpuBatchSolveTiming {
  double upload_seconds = 0.0;
  double kernel_sync_seconds = 0.0;
  double download_seconds = 0.0;
};

class GpuCsrSolver {
  public:
  Status Initialize(const Assembly& assembly)
  {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
      return Status::kGpuUnavailable;
    }
    n_ = static_cast<int>(assembly.free_nodes.size());
    if (n_ <= 0 || assembly.row_offsets.size() != static_cast<size_t>(n_ + 1) || assembly.diagonal.size() != static_cast<size_t>(n_) || assembly.row_offsets.front() != 0 || assembly.row_offsets.back() != static_cast<int32_t>(assembly.values.size()) || assembly.column_indices.size() != assembly.values.size()) {
      return Status::kInvalidArgument;
    }
    for (int row = 0; row < n_; ++row) {
      if (assembly.row_offsets[row] > assembly.row_offsets[row + 1] || !std::isfinite(assembly.diagonal[row])) {
        return Status::kInvalidArgument;
      }
    }
    for (size_t entry = 0; entry < assembly.values.size(); ++entry) {
      if (assembly.column_indices[entry] < 0 || assembly.column_indices[entry] >= n_ || !std::isfinite(assembly.values[entry])) {
        return Status::kInvalidArgument;
      }
    }
    std::vector<double> inverse_diagonal;
    inverse_diagonal_usable_ = BuildJacobiInverseDiagonal(
        assembly.diagonal, &inverse_diagonal);
    Status status = Status::kOk;
    if ((status = row_offsets_.allocate(assembly.row_offsets.size())) != Status::kOk || (status = column_indices_.allocate(assembly.column_indices.size())) != Status::kOk || (status = values_.allocate(assembly.values.size())) != Status::kOk || (status = diagonal_.allocate(assembly.diagonal.size())) != Status::kOk || (status = inverse_diagonal_.allocate(assembly.diagonal.size())) != Status::kOk || (status = rhs_.allocate(n_)) != Status::kOk || (status = solution_.allocate(n_)) != Status::kOk || (status = residual_.allocate(n_)) != Status::kOk || (status = direction_.allocate(n_)) != Status::kOk || (status = preconditioned_.allocate(n_)) != Status::kOk || (status = matrix_direction_.allocate(n_)) != Status::kOk || (status = info_.allocate(1)) != Status::kOk) {
      return status;
    }
    if ((status = CopyToDevice(row_offsets_.get(), assembly.row_offsets.data(),
             assembly.row_offsets.size() * sizeof(int32_t)))
            != Status::kOk
        || (status = CopyToDevice(column_indices_.get(), assembly.column_indices.data(),
                assembly.column_indices.size() * sizeof(int32_t)))
            != Status::kOk
        || (status = CopyToDevice(values_.get(), assembly.values.data(),
                assembly.values.size() * sizeof(double)))
            != Status::kOk
        || (status = CopyToDevice(diagonal_.get(), assembly.diagonal.data(),
                assembly.diagonal.size() * sizeof(double)))
            != Status::kOk
        || (status = CopyToDevice(inverse_diagonal_.get(), inverse_diagonal.data(),
                inverse_diagonal.size() * sizeof(double)))
            != Status::kOk) {
      return status;
    }
    host_row_offsets_ = assembly.row_offsets;
    host_column_indices_ = assembly.column_indices;
    initialized_ = true;
    return Status::kOk;
  }

  bool HasMatchingStructure(const Assembly& assembly) const
  {
    return initialized_ && n_ == static_cast<int>(assembly.free_nodes.size())
        && assembly.row_offsets == host_row_offsets_
        && assembly.column_indices == host_column_indices_;
  }

  Status UpdateValues(const Assembly& assembly)
  {
    if (!HasMatchingStructure(assembly) || assembly.values.size() != host_column_indices_.size()
        || assembly.diagonal.size() != static_cast<size_t>(n_)) return Status::kInvalidArgument;
    for (double value : assembly.values) if (!std::isfinite(value)) return Status::kInvalidArgument;
    for (double value : assembly.diagonal) if (!std::isfinite(value)) return Status::kInvalidArgument;
    std::vector<double> inverse_diagonal;
    inverse_diagonal_usable_ = BuildJacobiInverseDiagonal(
        assembly.diagonal, &inverse_diagonal);
    Status status = CopyToDevice(values_.get(), assembly.values.data(), assembly.values.size() * sizeof(double));
    if (status != Status::kOk) return status;
    if ((status = CopyToDevice(diagonal_.get(), assembly.diagonal.data(),
             assembly.diagonal.size() * sizeof(double))) != Status::kOk) return status;
    return CopyToDevice(inverse_diagonal_.get(), inverse_diagonal.data(),
        inverse_diagonal.size() * sizeof(double));
  }

  SolveInfo Solve(const std::vector<double>& rhs, double relative_tolerance,
      int max_iterations, std::vector<double>* solution)
  {
    SolveInfo result;
    result.status = Status::kInvalidArgument;
    if (rhs.size() != static_cast<size_t>(n_) || solution == nullptr)
      return result;
    Status copy_status = CopyToDevice(rhs_.get(), rhs.data(), rhs.size() * sizeof(double));
    if (copy_status != Status::kOk) {
      result.status = copy_status;
      return result;
    }
    if (inverse_diagonal_usable_) {
      DeterministicPcgKernel<true><<<1, kPcgBlockThreads>>>(
          n_, row_offsets_.get(), column_indices_.get(), values_.get(), diagonal_.get(),
          inverse_diagonal_.get(), rhs_.get(), solution_.get(), residual_.get(), direction_.get(),
          preconditioned_.get(), matrix_direction_.get(), relative_tolerance,
          max_iterations, info_.get(), static_cast<int>(host_column_indices_.size()));
    } else {
      DeterministicPcgKernel<false><<<1, kPcgBlockThreads>>>(
          n_, row_offsets_.get(), column_indices_.get(), values_.get(), diagonal_.get(),
          inverse_diagonal_.get(), rhs_.get(), solution_.get(), residual_.get(), direction_.get(),
          preconditioned_.get(), matrix_direction_.get(), relative_tolerance,
          max_iterations, info_.get(), static_cast<int>(host_column_indices_.size()));
    }
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      result.status = Status::kInternalError;
      return result;
    }
    if (CopyToHost(&result, info_.get(), sizeof(result)) != Status::kOk) {
      result.status = Status::kInternalError;
      return result;
    }
    solution->resize(n_);
    if (CopyToHost(solution->data(), solution_.get(), n_ * sizeof(double)) != Status::kOk) {
      result.status = Status::kInternalError;
    }
    return result;
  }

  Status SolveBatch(const std::vector<Assembly>& assemblies,
      const std::vector<std::vector<double>>& rhs_values, double relative_tolerance,
      int max_iterations, std::vector<SolveInfo>* infos,
      std::vector<std::vector<double>>* solutions, int* launches,
      GpuBatchSolveTiming* timing = nullptr)
  {
    if (infos == nullptr || solutions == nullptr || assemblies.empty()
        || assemblies.size() != rhs_values.size()) return Status::kInvalidArgument;
    const size_t count = assemblies.size();
    for (size_t item = 0; item < count; ++item) {
      if (!HasMatchingStructure(assemblies[item]) || rhs_values[item].size() != static_cast<size_t>(n_))
        return Status::kInvalidArgument;
    }
    const size_t nnz = host_column_indices_.size();
    const ProfileClock::time_point upload_start = ProfileClock::now();
    if (batch_capacity_ < count) {
      Status status = Status::kOk;
      if ((status = batch_values_.allocate(count * nnz)) != Status::kOk
          || (status = batch_diagonal_.allocate(count * n_)) != Status::kOk
          || (status = batch_inverse_diagonal_.allocate(count * n_)) != Status::kOk
          || (status = batch_rhs_.allocate(count * n_)) != Status::kOk
          || (status = batch_solution_.allocate(count * n_)) != Status::kOk
          || (status = batch_residual_.allocate(count * n_)) != Status::kOk
          || (status = batch_direction_.allocate(count * n_)) != Status::kOk
          || (status = batch_preconditioned_.allocate(count * n_)) != Status::kOk
          || (status = batch_matrix_direction_.allocate(count * n_)) != Status::kOk
          || (status = batch_info_.allocate(count)) != Status::kOk
          || (status = batch_cooperative_partials_.allocate(
                  count * kPcgCooperativeShardsPerItem)) != Status::kOk
          || (status = batch_cooperative_scalars_.allocate(count * 3)) != Status::kOk
          || (status = batch_cooperative_controls_.allocate(count * 4 + 1)) != Status::kOk) return status;
      batch_capacity_ = count;
    }
    std::vector<double> inverse_diagonal;
    bool batch_inverse_diagonal_usable = true;
    for (size_t item = 0; item < count; ++item) {
      batch_inverse_diagonal_usable = BuildJacobiInverseDiagonal(
          assemblies[item].diagonal, &inverse_diagonal) && batch_inverse_diagonal_usable;
      Status status = CopyToDevice(batch_values_.get() + item * nnz, assemblies[item].values.data(), nnz * sizeof(double));
      if (status != Status::kOk) return status;
      if ((status = CopyToDevice(batch_diagonal_.get() + item * n_, assemblies[item].diagonal.data(), n_ * sizeof(double))) != Status::kOk
          || (status = CopyToDevice(batch_inverse_diagonal_.get() + item * n_, inverse_diagonal.data(), n_ * sizeof(double))) != Status::kOk
          || (status = CopyToDevice(batch_rhs_.get() + item * n_, rhs_values[item].data(), n_ * sizeof(double))) != Status::kOk) return status;
    }
    if (timing != nullptr) timing->upload_seconds += ProfileSecondsSince(upload_start);
    const ProfileClock::time_point kernel_start = ProfileClock::now();
    const CooperativeLaunchResult cooperative_result = TryLaunchCooperativeBatch(
        count, nnz, relative_tolerance, max_iterations, batch_inverse_diagonal_usable);
    if (cooperative_result == CooperativeLaunchResult::kFailed)
      return Status::kInternalError;
    const bool launched_cooperatively =
        cooperative_result == CooperativeLaunchResult::kCompleted;
    if (!launched_cooperatively && batch_inverse_diagonal_usable) {
      DeterministicPcgKernel<true><<<static_cast<unsigned int>(count), kPcgBlockThreads>>>(
          n_, row_offsets_.get(), column_indices_.get(), batch_values_.get(), batch_diagonal_.get(),
          batch_inverse_diagonal_.get(), batch_rhs_.get(), batch_solution_.get(),
          batch_residual_.get(), batch_direction_.get(), batch_preconditioned_.get(),
          batch_matrix_direction_.get(), relative_tolerance, max_iterations, batch_info_.get(),
          static_cast<int>(nnz));
    } else if (!launched_cooperatively) {
      DeterministicPcgKernel<false><<<static_cast<unsigned int>(count), kPcgBlockThreads>>>(
          n_, row_offsets_.get(), column_indices_.get(), batch_values_.get(), batch_diagonal_.get(),
          batch_inverse_diagonal_.get(), batch_rhs_.get(), batch_solution_.get(),
          batch_residual_.get(), batch_direction_.get(), batch_preconditioned_.get(),
          batch_matrix_direction_.get(), relative_tolerance, max_iterations, batch_info_.get(),
          static_cast<int>(nnz));
    }
    if (launches != nullptr) ++*launches;
    if (!launched_cooperatively
        && (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)) return Status::kInternalError;
    if (timing != nullptr) timing->kernel_sync_seconds += ProfileSecondsSince(kernel_start);
    const ProfileClock::time_point download_start = ProfileClock::now();
    infos->assign(count, SolveInfo {}); solutions->assign(count, std::vector<double>(static_cast<size_t>(n_)));
    for (size_t item = 0; item < count; ++item) {
      Status status = CopyToHost(&(*infos)[item], batch_info_.get() + item, sizeof(SolveInfo));
      if (status != Status::kOk) return status;
      if ((status = CopyToHost((*solutions)[item].data(), batch_solution_.get() + item * n_, n_ * sizeof(double))) != Status::kOk) return status;
    }
    if (timing != nullptr) timing->download_seconds += ProfileSecondsSince(download_start);
    return Status::kOk;
  }

  private:
  enum class CooperativeLaunchResult { kNotSupported, kCompleted, kFailed };

  CooperativeLaunchResult TryLaunchCooperativeBatch(
      size_t count, size_t nnz, double relative_tolerance,
      int max_iterations, bool use_inverse_diagonal)
  {
    // The B=32 path already occupies almost every SM on the target GPU.  This
    // path is deliberately limited to small batches where eight shards/item
    // can fit resident at once and increase row-level parallelism.
    // Keep the tiny fixture/exact-bit-parity path on the established kernel;
    // the cooperative launch overhead cannot pay back below a real motor mesh.
    if (count == 0 || count > 4 || n_ < 4096)
      return CooperativeLaunchResult::kNotSupported;
    int device = 0;
    int cooperative_supported = 0;
    cudaDeviceProp properties {};
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaDeviceGetAttribute(&cooperative_supported, cudaDevAttrCooperativeLaunch,
               device) != cudaSuccess
        || cooperative_supported == 0
        || cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
      cudaGetLastError();
      return CooperativeLaunchResult::kNotSupported;
    }
    int active_blocks_per_sm = 0;
    const cudaError_t occupancy_status = use_inverse_diagonal
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm,
              CooperativeDeterministicPcgKernel<true>, kPcgBlockThreads, 0)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm,
              CooperativeDeterministicPcgKernel<false>, kPcgBlockThreads, 0);
    const size_t required_blocks = count * kPcgCooperativeShardsPerItem;
    if (occupancy_status != cudaSuccess || active_blocks_per_sm <= 0
        || required_blocks > static_cast<size_t>(active_blocks_per_sm)
                * static_cast<size_t>(properties.multiProcessorCount)) {
      cudaGetLastError();
      return CooperativeLaunchResult::kNotSupported;
    }
    int item_count = static_cast<int>(count);
    int shard_count = kPcgCooperativeShardsPerItem;
    int values_per_item = static_cast<int>(nnz);
    const int32_t* row_offsets = row_offsets_.get();
    const int32_t* column_indices = column_indices_.get();
    const double* values = batch_values_.get();
    const double* diagonal = batch_diagonal_.get();
    const double* inverse_diagonal = batch_inverse_diagonal_.get();
    const double* rhs = batch_rhs_.get();
    double* solution = batch_solution_.get();
    double* residual = batch_residual_.get();
    double* direction = batch_direction_.get();
    double* preconditioned = batch_preconditioned_.get();
    double* matrix_direction = batch_matrix_direction_.get();
    SolveInfo* info = batch_info_.get();
    double* partials = batch_cooperative_partials_.get();
    double* scalars = batch_cooperative_scalars_.get();
    int* controls = batch_cooperative_controls_.get();
    void* arguments[] = { &n_, &row_offsets, &column_indices, &values, &diagonal,
      &inverse_diagonal, &rhs, &solution, &residual, &direction, &preconditioned,
      &matrix_direction, &relative_tolerance, &max_iterations, &info, &values_per_item,
      &item_count, &shard_count, &partials, &scalars, &controls };
    const dim3 grid(static_cast<unsigned int>(required_blocks));
    const dim3 block(kPcgBlockThreads);
    const cudaError_t launch_status = use_inverse_diagonal
        ? cudaLaunchCooperativeKernel(
              reinterpret_cast<void*>(CooperativeDeterministicPcgKernel<true>),
              grid, block, arguments)
        : cudaLaunchCooperativeKernel(
              reinterpret_cast<void*>(CooperativeDeterministicPcgKernel<false>),
              grid, block, arguments);
    if (launch_status != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      cudaGetLastError();
      return CooperativeLaunchResult::kFailed;
    }
    return CooperativeLaunchResult::kCompleted;
  }

  int n_ = 0;
  bool initialized_ = false;
  std::vector<int32_t> host_row_offsets_;
  std::vector<int32_t> host_column_indices_;
  DeviceBuffer<int32_t> row_offsets_;
  DeviceBuffer<int32_t> column_indices_;
  DeviceBuffer<double> values_;
  // Original diagonal remains available to the kernel's validation path.
  DeviceBuffer<double> diagonal_;
  DeviceBuffer<double> inverse_diagonal_;
  bool inverse_diagonal_usable_ = false;
  DeviceBuffer<double> rhs_;
  DeviceBuffer<double> solution_;
  DeviceBuffer<double> residual_;
  DeviceBuffer<double> direction_;
  DeviceBuffer<double> preconditioned_;
  DeviceBuffer<double> matrix_direction_;
  DeviceBuffer<SolveInfo> info_;
  size_t batch_capacity_ = 0;
  DeviceBuffer<double> batch_values_, batch_diagonal_, batch_inverse_diagonal_, batch_rhs_, batch_solution_, batch_residual_, batch_direction_, batch_preconditioned_, batch_matrix_direction_;
  DeviceBuffer<SolveInfo> batch_info_;
  DeviceBuffer<double> batch_cooperative_partials_, batch_cooperative_scalars_;
  DeviceBuffer<int> batch_cooperative_controls_;
};

class LinearP1FixtureSolver {
  public:
  Status Initialize(Model model)
  {
    model_ = std::move(model);
    Status status = Assemble(model_, &assembly_);
    if (status != Status::kOk)
      return status;
    if ((status = solver_.Initialize(assembly_)) != Status::kOk || (status = nodes_.allocate(model_.nodes.size())) != Status::kOk || (status = triangles_.allocate(model_.triangles.size())) != Status::kOk || (status = nodal_a_.allocate(model_.nodes.size())) != Status::kOk || (status = bx_.allocate(model_.triangles.size())) != Status::kOk || (status = by_.allocate(model_.triangles.size())) != Status::kOk || (status = source_load_per_amp_.allocate(model_.nodes.size())) != Status::kOk || (status = flux_linkage_.allocate(1)) != Status::kOk) {
      return status;
    }
    if ((status = CopyToDevice(nodes_.get(), model_.nodes.data(),
             model_.nodes.size() * sizeof(Node)))
            != Status::kOk
        || (status = CopyToDevice(triangles_.get(), model_.triangles.data(),
                model_.triangles.size() * sizeof(Triangle)))
            != Status::kOk
        || (status = CopyToDevice(source_load_per_amp_.get(),
                assembly_.source_load_per_amp.data(),
                assembly_.source_load_per_amp.size() * sizeof(double)))
            != Status::kOk) {
      return status;
    }
    initialized_ = true;
    return Status::kOk;
  }

  SolveResult Solve(double current_a, double relative_tolerance = 1e-13,
      int max_iterations = 128)
  {
    SolveResult result;
    result.info.status = Status::kInvalidArgument;
    if (!initialized_ || !std::isfinite(current_a))
      return result;
    std::vector<double> rhs(assembly_.free_nodes.size());
    for (size_t i = 0; i < rhs.size(); ++i) {
      rhs[i] = current_a * assembly_.rhs_per_amp[i] + assembly_.rhs_offset[i];
    }
    std::vector<double> free_solution;
    result.info = solver_.Solve(rhs, relative_tolerance, max_iterations, &free_solution);
    if (result.info.status != Status::kOk)
      return result;

    result.a_wb_per_m = assembly_.boundary_values;
    for (size_t i = 0; i < assembly_.free_nodes.size(); ++i) {
      result.a_wb_per_m[assembly_.free_nodes[i]] = free_solution[i];
    }
    Status status = CopyToDevice(nodal_a_.get(), result.a_wb_per_m.data(),
        result.a_wb_per_m.size() * sizeof(double));
    if (status != Status::kOk) {
      result.info.status = status;
      return result;
    }
    const int threads = 128;
    const int blocks = (static_cast<int>(model_.triangles.size()) + threads - 1) / threads;
    ComputeFieldKernel<<<blocks, threads>>>(
        static_cast<int>(model_.triangles.size()), nodes_.get(), triangles_.get(),
        nodal_a_.get(), bx_.get(), by_.get());
    ComputeFluxKernel<<<1, 1>>>(
        static_cast<int>(model_.nodes.size()), source_load_per_amp_.get(),
        nodal_a_.get(), model_.depth_m, flux_linkage_.get());
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      result.info.status = Status::kInternalError;
      return result;
    }
    result.bx_t.resize(model_.triangles.size());
    result.by_t.resize(model_.triangles.size());
    if (CopyToHost(result.bx_t.data(), bx_.get(), result.bx_t.size() * sizeof(double)) != Status::kOk || CopyToHost(result.by_t.data(), by_.get(), result.by_t.size() * sizeof(double)) != Status::kOk || CopyToHost(&result.flux_linkage_wb, flux_linkage_.get(), sizeof(double)) != Status::kOk) {
      result.info.status = Status::kInternalError;
      return result;
    }
    if (!std::isfinite(result.flux_linkage_wb)) {
      result.info.status = Status::kNumericalNonfinite;
    }
    return result;
  }

  private:
  bool initialized_ = false;
  Model model_;
  Assembly assembly_;
  GpuCsrSolver solver_;
  DeviceBuffer<Node> nodes_;
  DeviceBuffer<Triangle> triangles_;
  DeviceBuffer<double> nodal_a_;
  DeviceBuffer<double> bx_;
  DeviceBuffer<double> by_;
  DeviceBuffer<double> source_load_per_amp_;
  DeviceBuffer<double> flux_linkage_;
};

struct NonlinearBhEvaluation {
  double h_a_per_m = 0.0;
  double dh_db = 0.0;
  double reluctivity_m_per_h = 0.0;
  double dv_db2 = 0.0;
};

struct NonlinearBhCurve {
  std::vector<double> b_t;
  std::vector<double> h_a_per_m;
  std::vector<double> slope;
  double cold_secant_reluctivity_m_per_h = 0.0;
  int smoothing_passes = 0;
};

bool BuildNonlinearBhSpline(NonlinearBhCurve* curve)
{
  const int n = static_cast<int>(curve->b_t.size());
  if (n < 2 || curve->h_a_per_m.size() != curve->b_t.size()
      || curve->b_t.front() != 0.0 || curve->h_a_per_m.front() != 0.0)
    return false;
  if (!(curve->cold_secant_reluctivity_m_per_h > 0.0))
    curve->cold_secant_reluctivity_m_per_h = curve->h_a_per_m[1] / curve->b_t[1];
  for (int i = 1; i < n; ++i) {
    if (!(curve->b_t[i] > curve->b_t[i - 1]) || curve->h_a_per_m[i] < 0.0
        || !std::isfinite(curve->b_t[i]) || !std::isfinite(curve->h_a_per_m[i]))
      return false;
  }
  // Exact FEMM DC natural-spline linear system (including its end rows).
  std::vector<double> matrix(static_cast<size_t>(n) * n, 0.0), rhs(n, 0.0);
  const double first = curve->b_t[1] - curve->b_t[0];
  matrix[0] = 4.0 / first;
  matrix[1] = 2.0 / first;
  rhs[0] = 6.0 * (curve->h_a_per_m[1] - curve->h_a_per_m[0]) / (first * first);
  const double last = curve->b_t[n - 1] - curve->b_t[n - 2];
  matrix[(n - 1) * n + n - 1] = 4.0 / last;
  matrix[(n - 1) * n + n - 2] = 2.0 / last;
  rhs[n - 1] = 6.0 * (curve->h_a_per_m[n - 1] - curve->h_a_per_m[n - 2]) / (last * last);
  for (int i = 1; i < n - 1; ++i) {
    const double left = curve->b_t[i] - curve->b_t[i - 1];
    const double right = curve->b_t[i + 1] - curve->b_t[i];
    matrix[i * n + i - 1] = 2.0 / left;
    matrix[i * n + i] = 4.0 * (left + right) / (left * right);
    matrix[i * n + i + 1] = 2.0 / right;
    rhs[i] = 6.0 * (curve->h_a_per_m[i] - curve->h_a_per_m[i - 1]) / (left * left)
        + 6.0 * (curve->h_a_per_m[i + 1] - curve->h_a_per_m[i]) / (right * right);
  }
  for (int pivot_col = 0; pivot_col < n; ++pivot_col) {
    int pivot = pivot_col;
    for (int row = pivot_col + 1; row < n; ++row) {
      if (std::abs(matrix[row * n + pivot_col]) > std::abs(matrix[pivot * n + pivot_col]))
        pivot = row;
    }
    if (!(std::abs(matrix[pivot * n + pivot_col]) > 0.0))
      return false;
    for (int col = pivot_col; col < n; ++col)
      std::swap(matrix[pivot_col * n + col], matrix[pivot * n + col]);
    std::swap(rhs[pivot_col], rhs[pivot]);
    for (int row = pivot_col + 1; row < n; ++row) {
      const double factor = matrix[row * n + pivot_col] / matrix[pivot_col * n + pivot_col];
      for (int col = pivot_col; col < n; ++col)
        matrix[row * n + col] -= factor * matrix[pivot_col * n + col];
      rhs[row] -= factor * rhs[pivot_col];
    }
  }
  curve->slope.assign(n, 0.0);
  for (int row = n - 1; row >= 0; --row) {
    double value = rhs[row];
    for (int col = row + 1; col < n; ++col)
      value -= matrix[row * n + col] * curve->slope[col];
    curve->slope[row] = value / matrix[row * n + row];
    if (!std::isfinite(curve->slope[row]))
      return false;
  }
  // FEMM repeats the natural-spline solve after 3-point smoothing until no
  // segment has an interior derivative root. Keep the original first secant
  // above for the cold (iteration zero) assembly rule.
  bool monotone = true;
  for (int i = 1; i < n; ++i) {
    const double length = curve->b_t[i] - curve->b_t[i - 1];
    const double d0 = curve->slope[i - 1], d1 = curve->slope[i];
    const double h0 = curve->h_a_per_m[i - 1], h1 = curve->h_a_per_m[i];
    const double c0 = d0;
    const double c1 = -2.0 * (2.0 * d0 * length + d1 * length + 3.0 * h0 - 3.0 * h1)
        / (length * length);
    const double c2 = 3.0 * (d0 * length + d1 * length + 2.0 * h0 - 2.0 * h1)
        / (length * length * length);
    double root0 = -1.0, root1 = -1.0;
    const double discriminant = c1 * c1 - 4.0 * c0 * c2;
    if (c2 == 0.0) {
      if (c1 != 0.0)
        root0 = -c0 / c1;
    } else if (discriminant > 0.0) {
      const double root = std::sqrt(discriminant);
      root0 = -(c1 + root) / (2.0 * c2);
      root1 = (-c1 + root) / (2.0 * c2);
    }
    if ((root0 >= 0.0 && root0 <= length) || (root1 >= 0.0 && root1 <= length)) {
      monotone = false;
      break;
    }
  }
  if (monotone)
    return true;
  if (++curve->smoothing_passes > 1024)
    return false;
  std::vector<double> next_b = curve->b_t, next_h = curve->h_a_per_m;
  for (int i = 1; i < n - 1; ++i) {
    next_b[i] = (curve->b_t[i - 1] + curve->b_t[i] + curve->b_t[i + 1]) / 3.0;
    next_h[i] = (curve->h_a_per_m[i - 1] + curve->h_a_per_m[i] + curve->h_a_per_m[i + 1]) / 3.0;
  }
  curve->b_t = std::move(next_b);
  curve->h_a_per_m = std::move(next_h);
  curve->slope.clear();
  return BuildNonlinearBhSpline(curve);
}

NonlinearBhEvaluation EvaluateNonlinearBh(const NonlinearBhCurve& curve, double field_b)
{
  NonlinearBhEvaluation result;
  const double b = std::abs(field_b);
  if (curve.slope.size() != curve.b_t.size() || b == 0.0) {
    result.dh_db = curve.slope.empty() ? std::numeric_limits<double>::quiet_NaN() : curve.slope[0];
    result.reluctivity_m_per_h = result.dh_db;
    return result;
  }
  const int n = static_cast<int>(curve.b_t.size());
  if (b > curve.b_t.back()) {
    result.h_a_per_m = curve.h_a_per_m.back() + curve.slope.back() * (b - curve.b_t.back());
    result.dh_db = curve.slope.back();
  } else {
    for (int i = 0; i < n - 1; ++i) {
      if (b >= curve.b_t[i] && b <= curve.b_t[i + 1]) {
        const double length = curve.b_t[i + 1] - curve.b_t[i];
        const double z = (b - curve.b_t[i]) / length, z2 = z * z;
        result.h_a_per_m = (1.0 - 3.0 * z2 + 2.0 * z2 * z) * curve.h_a_per_m[i]
            + z * (1.0 - 2.0 * z + z2) * length * curve.slope[i]
            + z2 * (3.0 - 2.0 * z) * curve.h_a_per_m[i + 1]
            + z2 * (z - 1.0) * length * curve.slope[i + 1];
        result.dh_db = 6.0 * z * (z - 1.0) * curve.h_a_per_m[i] / length
            + (1.0 - 4.0 * z + 3.0 * z2) * curve.slope[i]
            + 6.0 * z * (1.0 - z) * curve.h_a_per_m[i + 1] / length
            + z * (3.0 * z - 2.0) * curve.slope[i + 1];
        break;
      }
    }
  }
  result.reluctivity_m_per_h = result.h_a_per_m / b;
  result.dv_db2 = 0.5 * (result.dh_db / (b * b) - result.h_a_per_m / (b * b * b));
  return result;
}

bool ReadNonlinearBhTable(const std::string& path, NonlinearBhCurve* curve)
{
  std::ifstream input(path);
  std::string header;
  if (!input || !std::getline(input, header) || header.find("H (A_per_meter)") == std::string::npos
      || header.find("B (tesla)") == std::string::npos)
    return false;
  double h = 0.0, b = 0.0;
  while (input >> h >> b) {
    curve->h_a_per_m.push_back(h);
    curve->b_t.push_back(b);
  }
  return input.eof() && BuildNonlinearBhSpline(curve);
}

// Package 3A keeps nonlinear assembly on the host for now, but routes every
// Newton correction through the existing deterministic CUDA CSR PCG backend.
// Each element explicitly names a material; an omitted label is a hard error.
struct NonlinearMaterial {
  // v(B) = v0 * (1 + alpha * |B|^2).  alpha==0 is a linear material.
  double reluctivity_zero_m_per_h = 0.0;
  double alpha_per_t2 = 0.0;
  double source_j_per_a = 0.0;
  double h_c_a_per_m = 0.0;
  double magnetization_deg = 0.0;
  int32_t bh_curve_index = -1;
  // -1 denotes a non-driven region.  Driven labels point into the ordered
  // circuit-current vector owned by NonlinearModel.
  int32_t circuit_index = -1;
  // FEMM block-label group, retained for group-based postprocessing.
  int32_t group_number = -1;
};

struct NonlinearTriangle {
  int32_t node[3];
  int32_t material = -1;
};

// FEMM's native Air Gap Element (AGE) represents the annular air band without
// re-triangulating it.  The four node/weight pairs map each annulus endpoint
// to the independently meshed rotor and stator rings.
struct AirGapQuadPoint {
  int32_t node[4] = {};
  double weight[4] = {};
};

struct AirGapElement {
  bool antiperiodic = false;
  double center_x_m = 0.0;
  double center_y_m = 0.0;
  double inner_radius_m = 0.0;
  double outer_radius_m = 0.0;
  double arc_length_deg = 0.0;
  double inner_shift = 0.0;
  double outer_shift = 0.0;
  std::vector<AirGapQuadPoint> quad_points;
};

struct NonlinearModel {
  std::vector<Node> nodes;
  std::vector<NonlinearMaterial> materials;
  std::vector<NonlinearBhCurve> bh_curves;
  std::vector<NonlinearTriangle> triangles;
  std::vector<AirGapElement> air_gap_elements;
  std::vector<int32_t> dirichlet_nodes;
  std::vector<double> dirichlet_a_wb_per_m;
  int32_t circuit_count = 1;
  double depth_m = 0.0;
};

struct NonlinearOptions {
  double relative_tolerance = 1e-12;
  int max_newton_iterations = 32;
  int max_linear_iterations = 128;
  double linear_relative_tolerance = 1e-13;
};

constexpr double kMu0 = 4.0e-7 * 3.141592653589793238462643383279502884;
// fkn's AGE matrix is expressed against relative permeability; P1 assembly
// below is SI and therefore needs this free-space reluctivity multiplier.
constexpr double kNativeFemmAgeToSiReluctivity = 1.0 / kMu0;

// Full motor meshes can need substantially more PCG steps than the compact
// fixtures, especially for thin V-magnet bridges.  This remains an upper
// bound: the deterministic solver exits as soon as the motor-specific 1e-12
// relative linear tolerance is verified against the true residual.  This is
// still four orders tighter than the outer nonlinear tolerance while avoiding
// the measured double-precision residual plateau on the V-magnet mesh.
constexpr int kMotorMaxLinearIterations = 16384;
constexpr double kMotorLinearRelativeTolerance = 1e-12;

struct NonlinearSolveResult {
  SolveInfo info;
  std::vector<double> a_wb_per_m;
  std::vector<double> bx_t;
  std::vector<double> by_t;
  std::vector<double> residual_history;
  std::vector<double> circuit_currents_a;
  std::vector<double> circuit_flux_linkage_wb;
  // Compatibility field for frozen single-circuit callers only.
  double flux_linkage_wb = std::numeric_limits<double>::quiet_NaN();
};

struct NonlinearElementTerms {
  double area_m2 = 0.0;
  double b_x[3] = {}; // curl basis x component: dN/dy
  double b_y[3] = {}; // curl basis y component: -dN/dx
};

bool BuildElementTerms(const Node p[3], NonlinearElementTerms* terms)
{
  const double determinant = (p[1].x_m - p[0].x_m) * (p[2].y_m - p[0].y_m)
      - (p[2].x_m - p[0].x_m) * (p[1].y_m - p[0].y_m);
  if (!(determinant > 0.0) || !std::isfinite(determinant))
    return false;
  terms->area_m2 = 0.5 * determinant;
  const double dx[3] = { p[1].y_m - p[2].y_m, p[2].y_m - p[0].y_m,
    p[0].y_m - p[1].y_m };
  const double dy[3] = { p[2].x_m - p[1].x_m, p[0].x_m - p[2].x_m,
    p[1].x_m - p[0].x_m };
  for (int i = 0; i < 3; ++i) {
    terms->b_x[i] = dy[i] / determinant;
    terms->b_y[i] = -dx[i] / determinant;
  }
  return true;
}

Status ValidateNonlinearModel(const NonlinearModel& model)
{
  if (model.nodes.empty() || model.triangles.empty() || model.materials.empty()
      || !(model.depth_m > 0.0) || !std::isfinite(model.depth_m)
      || model.circuit_count <= 0
      || model.dirichlet_nodes.empty()
      || model.dirichlet_nodes.size() != model.dirichlet_a_wb_per_m.size()) {
    return Status::kInvalidArgument;
  }
  std::vector<bool> boundary(model.nodes.size(), false);
  for (size_t i = 0; i < model.dirichlet_nodes.size(); ++i) {
    const int32_t node = model.dirichlet_nodes[i];
    if (node < 0 || static_cast<size_t>(node) >= model.nodes.size() || boundary[node]
        || !std::isfinite(model.dirichlet_a_wb_per_m[i]))
      return Status::kBoundaryInvalid;
    boundary[node] = true;
  }
  if (model.dirichlet_nodes.size() == model.nodes.size())
    return Status::kBoundaryInvalid;
  for (const Node& node : model.nodes) {
    if (!std::isfinite(node.x_m) || !std::isfinite(node.y_m))
      return Status::kMeshInvalid;
  }
  for (const NonlinearMaterial& material : model.materials) {
    if (!(material.reluctivity_zero_m_per_h > 0.0) || material.alpha_per_t2 < 0.0
        || !std::isfinite(material.reluctivity_zero_m_per_h)
        || !std::isfinite(material.alpha_per_t2) || !std::isfinite(material.source_j_per_a)
        || !std::isfinite(material.h_c_a_per_m)
        || !std::isfinite(material.magnetization_deg))
      return Status::kInvalidMaterial;
    if (material.circuit_index < -1 || material.circuit_index >= model.circuit_count
        || (material.source_j_per_a != 0.0 && material.circuit_index < 0))
      return Status::kInvalidMaterial;
    if (material.bh_curve_index < -1
        || (material.bh_curve_index >= 0
            && static_cast<size_t>(material.bh_curve_index) >= model.bh_curves.size()))
      return Status::kInvalidMaterial;
    if (material.bh_curve_index >= 0) {
      const NonlinearBhCurve& curve = model.bh_curves[material.bh_curve_index];
      if (curve.b_t.size() < 2 || curve.h_a_per_m.size() != curve.b_t.size()
          || curve.slope.size() != curve.b_t.size()
          || !(curve.cold_secant_reluctivity_m_per_h > 0.0))
        return Status::kInvalidMaterial;
    }
  }
  for (const NonlinearTriangle& triangle : model.triangles) {
    if (triangle.material < 0 || static_cast<size_t>(triangle.material) >= model.materials.size())
      return Status::kInvalidMaterial;
    Node p[3];
    for (int local = 0; local < 3; ++local) {
      if (triangle.node[local] < 0 || static_cast<size_t>(triangle.node[local]) >= model.nodes.size())
        return Status::kMeshInvalid;
      p[local] = model.nodes[triangle.node[local]];
    }
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms))
      return Status::kMeshInvalid;
  }
  for (const AirGapElement& age : model.air_gap_elements) {
    const size_t elements = age.quad_points.size() - 1;
    if (age.quad_points.size() < 3 || !(age.inner_radius_m > 0.0)
        || !(age.outer_radius_m > age.inner_radius_m) || !(age.arc_length_deg > 0.0)
        || !std::isfinite(age.center_x_m) || !std::isfinite(age.center_y_m)
        || !std::isfinite(age.inner_shift) || !std::isfinite(age.outer_shift))
      return Status::kMeshInvalid;
    for (const AirGapQuadPoint& point : age.quad_points) for (int local = 0; local < 4; ++local) {
      if (point.node[local] < 0 || static_cast<size_t>(point.node[local]) >= model.nodes.size()
          || !std::isfinite(point.weight[local])) return Status::kMeshInvalid;
    }
    const double dt = age.arc_length_deg / static_cast<double>(elements);
    if (!(dt > 0.0) || !std::isfinite(dt)) return Status::kMeshInvalid;
  }
  return Status::kOk;
}

Status AssembleNonlinearNewton(const NonlinearModel& model,
    const std::vector<double>& a_old, const std::vector<double>& circuit_currents_a,
    bool use_newton,
    Assembly* assembly)
{
  if (assembly == nullptr || a_old.size() != model.nodes.size()
      || circuit_currents_a.size() != static_cast<size_t>(model.circuit_count))
    return Status::kInvalidArgument;
  for (const double current_a : circuit_currents_a) {
    if (!std::isfinite(current_a))
      return Status::kInvalidArgument;
  }
  const Status validation = ValidateNonlinearModel(model);
  if (validation != Status::kOk)
    return validation;
  const size_t node_count = model.nodes.size();
  SparseRows jacobian(node_count);
  std::vector<double> rhs(node_count, 0.0);
  assembly->source_load_per_amp.assign(node_count, 0.0);
  assembly->source_load_per_circuit.assign(static_cast<size_t>(model.circuit_count),
      std::vector<double>(node_count, 0.0));

  for (const NonlinearTriangle& triangle : model.triangles) {
    const NonlinearMaterial& material = model.materials[triangle.material];
    Node p[3] = { model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
      model.nodes[triangle.node[2]] };
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms))
      return Status::kMeshInvalid;
    double bx = 0.0;
    double by = 0.0;
    for (int i = 0; i < 3; ++i) {
      bx += a_old[triangle.node[i]] * terms.b_x[i];
      by += a_old[triangle.node[i]] * terms.b_y[i];
    }
    const double b_magnitude = std::hypot(bx, by);
    const NonlinearBhEvaluation bh = material.bh_curve_index >= 0
        ? EvaluateNonlinearBh(model.bh_curves[material.bh_curve_index], b_magnitude)
        : NonlinearBhEvaluation {};
    double reluctivity = material.bh_curve_index >= 0 ? bh.reluctivity_m_per_h
        : material.reluctivity_zero_m_per_h
            * (1.0 + material.alpha_per_t2 * b_magnitude * b_magnitude);
    // The Newton term uses d(v)/d(B^2), so a v0*(1+alpha*B^2)
    // material contributes v0*alpha without an extra |B| factor.
    const double reluctivity_derivative = use_newton
        ? (material.bh_curve_index >= 0 ? bh.dv_db2
                                        : material.reluctivity_zero_m_per_h * material.alpha_per_t2)
        : 0.0;
    if (!use_newton && material.bh_curve_index >= 0) {
      const NonlinearBhCurve& curve = model.bh_curves[material.bh_curve_index];
      reluctivity = curve.cold_secant_reluctivity_m_per_h;
    }
    if (!(reluctivity > 0.0) || !std::isfinite(reluctivity)
        || !std::isfinite(reluctivity_derivative))
      return Status::kNumericalNonfinite;
    const double theta_rad = material.magnetization_deg * 3.141592653589793238462643383279502884 / 180.0;
    const double hcx = material.h_c_a_per_m * std::cos(theta_rad);
    const double hcy = material.h_c_a_per_m * std::sin(theta_rad);
    for (int i = 0; i < 3; ++i) {
      const int global_i = triangle.node[i];
      const double coil = terms.area_m2 * material.source_j_per_a / 3.0;
      const double pm = terms.area_m2 * (hcx * terms.b_x[i] + hcy * terms.b_y[i]);
      if (material.circuit_index >= 0) {
        const size_t circuit = static_cast<size_t>(material.circuit_index);
        assembly->source_load_per_circuit[circuit][global_i] += coil;
        rhs[global_i] += circuit_currents_a[circuit] * coil;
      }
      rhs[global_i] += pm;
      const double b_dot_i = bx * terms.b_x[i] + by * terms.b_y[i];
      for (int j = 0; j < 3; ++j) {
        const int global_j = triangle.node[j];
        const double b_dot_j = bx * terms.b_x[j] + by * terms.b_y[j];
        const double k = terms.area_m2 * reluctivity
            * (terms.b_x[i] * terms.b_x[j] + terms.b_y[i] * terms.b_y[j]);
        const double c = use_newton ? 2.0 * terms.area_m2 * reluctivity_derivative
                * b_dot_i * b_dot_j
                                    : 0.0;
        const Status added = AddSparseEntry(&jacobian, global_i, global_j, k + c);
        if (added != Status::kOk)
          return added;
        if (use_newton)
          rhs[global_i] += c * a_old[global_j];
      }
    }
  }
  for (const AirGapElement& age : model.air_gap_elements) {
    const size_t elements = age.quad_points.size() - 1;
    if (elements == 0) return Status::kMeshInvalid;
    const double dt = (3.141592653589793238462643383279502884 / 180.0)
        * age.arc_length_deg / static_cast<double>(elements);
    const double K = 2.0 * (age.outer_radius_m - age.inner_radius_m)
        / (dt * (age.outer_radius_m + age.inner_radius_m));
    const double Ki = 1.0 / K;
    double ci = age.inner_shift, co = age.outer_shift;
    if (ci > co) { ci -= co; co = 0.0; }
    else { ci = 1.0 - co + ci; co = 1.0; }
    double matrix[10][10];
    if (!BuildNativeFemmAirGapMatrix(K, Ki, ci, co, matrix)) return Status::kMeshInvalid;
    for (size_t k = 0; k < elements; ++k) {
      const size_t previous = k == 0 ? elements - 1 : k - 1;
      const size_t next = k + 1;
      const size_t next2 = k + 2 > elements ? 1 : k + 2;
      int32_t node[10] = {
        age.quad_points[previous].node[0], age.quad_points[k].node[0],
        age.quad_points[k].node[1], age.quad_points[next].node[1],
        age.quad_points[next2].node[1], age.quad_points[previous].node[2],
        age.quad_points[k].node[2], age.quad_points[k].node[3],
        age.quad_points[next].node[3], age.quad_points[next2].node[3] };
      double weight[10] = {
        age.quad_points[previous].weight[0], age.quad_points[k].weight[0],
        age.quad_points[k].weight[1], age.quad_points[next].weight[1],
        age.quad_points[next2].weight[1], age.quad_points[previous].weight[2],
        age.quad_points[k].weight[2], age.quad_points[k].weight[3],
        age.quad_points[next].weight[3], age.quad_points[next2].weight[3] };
      if (age.antiperiodic && k == 0) { weight[0] = -weight[0]; weight[5] = -weight[5]; }
      if (age.antiperiodic && k + 1 == elements) { weight[4] = -weight[4]; weight[9] = -weight[9]; }
      for (int i = 0; i < 10; ++i) for (int j = 0; j < 10; ++j) {
        // fkn works in a relative-permeability, centimetre-scaled system;
        // this solver's P1 assembly is SI and carries 1/mu0 explicitly.
        const Status added = AddSparseEntry(&jacobian, node[i], node[j],
            matrix[i][j] * weight[i] * weight[j] * kNativeFemmAgeToSiReluctivity);
        if (added != Status::kOk) return added;
      }
    }
  }
  const Status sparse_validation = ValidateSparseRows(jacobian);
  if (sparse_validation != Status::kOk)
    return sparse_validation;

  std::vector<bool> is_boundary(node_count, false);
  assembly->boundary_values.assign(node_count, 0.0);
  for (size_t i = 0; i < model.dirichlet_nodes.size(); ++i) {
    is_boundary[model.dirichlet_nodes[i]] = true;
    assembly->boundary_values[model.dirichlet_nodes[i]] = model.dirichlet_a_wb_per_m[i];
  }
  assembly->free_nodes.clear();
  std::vector<int32_t> free_index(node_count, -1);
  for (size_t node = 0; node < node_count; ++node) {
    if (!is_boundary[node]) {
      free_index[node] = static_cast<int32_t>(assembly->free_nodes.size());
      assembly->free_nodes.push_back(static_cast<int32_t>(node));
    }
  }
  const size_t free_count = assembly->free_nodes.size();
  assembly->row_offsets.assign(free_count + 1, 0);
  assembly->column_indices.clear();
  assembly->values.clear();
  assembly->diagonal.assign(free_count, 0.0);
  assembly->rhs_per_amp.assign(free_count, 0.0);
  assembly->rhs_offset.assign(free_count, 0.0);
  for (size_t row = 0; row < free_count; ++row) {
    const int global_row = assembly->free_nodes[row];
    assembly->row_offsets[row] = static_cast<int32_t>(assembly->values.size());
    assembly->rhs_per_amp[row] = rhs[global_row];
    for (const auto& entry : jacobian[global_row]) {
      const int32_t global_column = entry.first;
      const double value = entry.second;
      if (is_boundary[global_column]) {
        assembly->rhs_offset[row] -= value * assembly->boundary_values[global_column];
      } else if (value != 0.0) {
        const int32_t column = free_index[global_column];
        if (column < 0)
          return Status::kAssemblyFailed;
        assembly->column_indices.push_back(column);
        assembly->values.push_back(value);
        if (static_cast<size_t>(column) == row)
          assembly->diagonal[row] = value;
      }
    }
    if (!(assembly->diagonal[row] > 0.0) || !std::isfinite(assembly->diagonal[row])
        || !std::isfinite(assembly->rhs_per_amp[row]) || !std::isfinite(assembly->rhs_offset[row]))
      return Status::kAssemblyFailed;
  }
  assembly->row_offsets[free_count] = static_cast<int32_t>(assembly->values.size());
  return Status::kOk;
}

Status AssembleNonlinearNewton(const NonlinearModel& model,
    const std::vector<double>& a_old, double current_a, bool use_newton,
    Assembly* assembly)
{
  return AssembleNonlinearNewton(model, a_old, std::vector<double> { current_a },
      use_newton, assembly);
}

struct NonlinearBatchTiming {
  double host_assembly_seconds = 0.0;
  double gpu_upload_seconds = 0.0;
  double gpu_kernel_sync_seconds = 0.0;
  double gpu_download_seconds = 0.0;
  double state_update_seconds = 0.0;
  double finalize_seconds = 0.0;
};

class NonlinearP1FixtureSolver {
  public:
  Status Initialize(NonlinearModel model)
  {
    const Status status = ValidateNonlinearModel(model);
    if (status != Status::kOk)
      return status;
    model_ = std::move(model);
    csr_initialized_ = false;
    csr_symbolic_reuse_count_ = 0;
    initialized_ = true;
    return Status::kOk;
  }

  size_t csr_symbolic_reuse_count() const { return csr_symbolic_reuse_count_; }

  NonlinearSolveResult Solve(double current_a, const NonlinearOptions& options = {},
      const std::vector<double>* warm_start = nullptr)
  {
    return Solve(std::vector<double> { current_a }, options, warm_start);
  }

  NonlinearSolveResult Solve(const std::vector<double>& circuit_currents_a,
      const NonlinearOptions& options = {}, const std::vector<double>* warm_start = nullptr)
  {
    NonlinearSolveResult result;
    result.info.status = Status::kInvalidArgument;
    if (!initialized_ || circuit_currents_a.size() != static_cast<size_t>(model_.circuit_count)
        || !(options.relative_tolerance > 0.0)
        || !std::isfinite(options.relative_tolerance) || options.max_newton_iterations < 0)
      return result;
    for (const double current_a : circuit_currents_a) {
      if (!std::isfinite(current_a))
        return result;
    }
    result.circuit_currents_a = circuit_currents_a;
    std::vector<double> a(model_.nodes.size(), 0.0);
    for (size_t i = 0; i < model_.dirichlet_nodes.size(); ++i)
      a[model_.dirichlet_nodes[i]] = model_.dirichlet_a_wb_per_m[i];
    const bool has_warm_start = warm_start != nullptr;
    if (has_warm_start) {
      if (warm_start->size() != a.size())
        return result;
      a = *warm_start;
      for (size_t i = 0; i < model_.dirichlet_nodes.size(); ++i)
        a[model_.dirichlet_nodes[i]] = model_.dirichlet_a_wb_per_m[i];
    }
    double relaxation = 1.0;
    double previous_residual = std::numeric_limits<double>::infinity();
    for (int iteration = 0; iteration < options.max_newton_iterations; ++iteration) {
      Assembly assembly;
      const bool use_newton = has_warm_start || iteration > 0;
      Status status = AssembleNonlinearNewton(
          model_, a, circuit_currents_a, use_newton, &assembly);
      if (status != Status::kOk) {
        result.info.status = status;
        return result;
      }
      if (!csr_initialized_) {
        status = csr_solver_.Initialize(assembly);
        if (status == Status::kOk) csr_initialized_ = true;
      } else if (!csr_solver_.HasMatchingStructure(assembly)) {
        // Geometry cache identity promises this cannot change.  Fail rather
        // than silently re-uploading a different symbolic matrix.
        status = Status::kAssemblyFailed;
      } else {
        status = csr_solver_.UpdateValues(assembly);
        if (status == Status::kOk) ++csr_symbolic_reuse_count_;
      }
      if (status != Status::kOk) {
        result.info.status = status;
        return result;
      }
      std::vector<double> free_solution;
      std::vector<double> rhs(assembly.free_nodes.size());
      for (size_t row = 0; row < rhs.size(); ++row)
        rhs[row] = assembly.rhs_per_amp[row] + assembly.rhs_offset[row];
      const SolveInfo linear_info = csr_solver_.Solve(rhs, options.linear_relative_tolerance,
          options.max_linear_iterations, &free_solution);
      if (linear_info.status != Status::kOk) {
        result.info = linear_info;
        return result;
      }
      std::vector<double> candidate = assembly.boundary_values;
      for (size_t row = 0; row < free_solution.size(); ++row)
        candidate[assembly.free_nodes[row]] = free_solution[row];
      double difference_sq = 0.0;
      double candidate_sq = 0.0;
      for (size_t i = 0; i < candidate.size(); ++i) {
        const double difference = candidate[i] - a[i];
        difference_sq += difference * difference;
        candidate_sq += candidate[i] * candidate[i];
      }
      const double residual = std::sqrt(difference_sq)
          / std::max(std::sqrt(candidate_sq), std::numeric_limits<double>::min());
      if (!std::isfinite(residual)) {
        result.info.status = Status::kNumericalNonfinite;
        return result;
      }
      result.residual_history.push_back(residual);
      if (iteration > 5) {
        if (residual > previous_residual && relaxation > 0.125)
          relaxation *= 0.5;
        else
          relaxation += 0.1 * (1.0 - relaxation);
      }
      for (size_t i = 0; i < a.size(); ++i)
        a[i] += relaxation * (candidate[i] - a[i]);
      result.info.iterations = iteration + 1;
      result.info.residual_l2 = residual;
      if (iteration > 0 && residual < 100.0 * options.relative_tolerance) {
        result.info.status = Status::kOk;
        result.a_wb_per_m = std::move(a);
        return FinalizeResult(&result);
      }
      previous_residual = residual;
    }
    result.info.status = Status::kNonlinearSolveNotConverged;
    result.a_wb_per_m = std::move(a);
    return FinalizeResult(&result);
  }

  // Independent nonlinear states share only the immutable symbolic CSR.  The
  // host still assembles each Newton state separately; every active state is
  // then dispatched as one CUDA block in a single PCG launch.
  std::vector<NonlinearSolveResult> SolveBatch(
      const std::vector<std::vector<double>>& currents, const NonlinearOptions& options,
      int* batched_pcg_launches = nullptr, NonlinearBatchTiming* timing = nullptr)
  {
    std::vector<NonlinearSolveResult> results(currents.size());
    if (!initialized_ || currents.empty() || !(options.relative_tolerance > 0.0)
        || !std::isfinite(options.relative_tolerance) || options.max_newton_iterations < 0) return results;
    std::vector<std::vector<double>> states(currents.size(), std::vector<double>(model_.nodes.size(), 0.0));
    std::vector<double> relaxations(currents.size(), 1.0);
    std::vector<double> previous_residuals(currents.size(), std::numeric_limits<double>::infinity());
    std::vector<bool> active(currents.size(), true);
    for (size_t item = 0; item < currents.size(); ++item) {
      results[item].info.status = Status::kInvalidArgument;
      results[item].circuit_currents_a = currents[item];
      if (currents[item].size() != static_cast<size_t>(model_.circuit_count)
          || !std::all_of(currents[item].begin(), currents[item].end(), [](double value) { return std::isfinite(value); })) {
        active[item] = false; continue;
      }
      for (size_t boundary = 0; boundary < model_.dirichlet_nodes.size(); ++boundary)
        states[item][model_.dirichlet_nodes[boundary]] = model_.dirichlet_a_wb_per_m[boundary];
    }
    for (int iteration = 0; iteration < options.max_newton_iterations; ++iteration) {
      const ProfileClock::time_point assembly_start = ProfileClock::now();
      std::vector<size_t> slots; std::vector<Assembly> assemblies; std::vector<std::vector<double>> rhs_values;
      const bool symbolic_preexisting = csr_initialized_;
      std::vector<size_t> active_slots;
      for (size_t item = 0; item < currents.size(); ++item)
        if (active[item]) active_slots.push_back(item);
      std::vector<Assembly> iteration_assemblies(active_slots.size());
      std::vector<Status> assembly_status(active_slots.size(), Status::kInternalError);
      std::atomic<size_t> next_assembly { 0 };
      const size_t worker_count = std::min<size_t>(active_slots.size(),
          std::min<size_t>(6, std::max(1u, std::thread::hardware_concurrency())));
      auto assemble = [&]() {
        for (;;) {
          const size_t local = next_assembly.fetch_add(1, std::memory_order_relaxed);
          if (local >= active_slots.size()) return;
          const size_t item = active_slots[local];
          assembly_status[local] = AssembleNonlinearNewton(model_, states[item],
              currents[item], iteration > 0, &iteration_assemblies[local]);
        }
      };
      std::vector<std::thread> workers;
      workers.reserve(worker_count > 0 ? worker_count - 1 : 0);
      for (size_t worker = 1; worker < worker_count; ++worker) {
        try { workers.emplace_back(assemble); }
        catch (const std::system_error&) { break; }
      }
      assemble();
      for (std::thread& worker : workers) worker.join();
      for (size_t local = 0; local < active_slots.size(); ++local) {
        const size_t item = active_slots[local];
        const Status status = assembly_status[local];
        if (status != Status::kOk) { results[item].info.status = status; active[item] = false; continue; }
        Assembly& assembly = iteration_assemblies[local];
        if (!csr_initialized_) {
          const Status initialize = csr_solver_.Initialize(assembly);
          if (initialize != Status::kOk) { results[item].info.status = initialize; active[item] = false; continue; }
          csr_initialized_ = true;
        }
        if (!csr_solver_.HasMatchingStructure(assembly)) {
          results[item].info.status = Status::kAssemblyFailed; active[item] = false; continue;
        }
        std::vector<double> rhs(assembly.free_nodes.size());
        for (size_t row = 0; row < rhs.size(); ++row) rhs[row] = assembly.rhs_per_amp[row] + assembly.rhs_offset[row];
        slots.push_back(item); assemblies.push_back(std::move(assembly)); rhs_values.push_back(std::move(rhs));
      }
      if (timing != nullptr) timing->host_assembly_seconds += ProfileSecondsSince(assembly_start);
      if (slots.empty()) break;
      std::vector<SolveInfo> infos; std::vector<std::vector<double>> free_solutions;
      GpuBatchSolveTiming gpu_timing;
      const Status batch_status = csr_solver_.SolveBatch(assemblies, rhs_values,
          options.linear_relative_tolerance, options.max_linear_iterations, &infos, &free_solutions,
          batched_pcg_launches, &gpu_timing);
      if (timing != nullptr) {
        timing->gpu_upload_seconds += gpu_timing.upload_seconds;
        timing->gpu_kernel_sync_seconds += gpu_timing.kernel_sync_seconds;
        timing->gpu_download_seconds += gpu_timing.download_seconds;
      }
      if (batch_status != Status::kOk) {
        for (size_t item : slots) { results[item].info.status = batch_status; active[item] = false; }
        break;
      }
      csr_symbolic_reuse_count_ += symbolic_preexisting ? slots.size()
          : (slots.empty() ? 0 : slots.size() - 1);
      const ProfileClock::time_point update_start = ProfileClock::now();
      double iteration_finalize_seconds = 0.0;
      for (size_t local = 0; local < slots.size(); ++local) {
        const size_t item = slots[local];
        if (infos[local].status != Status::kOk) { results[item].info = infos[local]; active[item] = false; continue; }
        std::vector<double> candidate = assemblies[local].boundary_values;
        for (size_t row = 0; row < free_solutions[local].size(); ++row)
          candidate[assemblies[local].free_nodes[row]] = free_solutions[local][row];
        double difference_sq = 0.0, candidate_sq = 0.0;
        for (size_t node = 0; node < candidate.size(); ++node) {
          const double difference = candidate[node] - states[item][node];
          difference_sq += difference * difference; candidate_sq += candidate[node] * candidate[node];
        }
        const double residual = std::sqrt(difference_sq)
            / std::max(std::sqrt(candidate_sq), std::numeric_limits<double>::min());
        results[item].residual_history.push_back(residual);
        results[item].info.iterations = iteration + 1; results[item].info.residual_l2 = residual;
        if (!std::isfinite(residual)) { results[item].info.status = Status::kNumericalNonfinite; active[item] = false; continue; }
        if (iteration > 5) {
          if (residual > previous_residuals[item] && relaxations[item] > 0.125) relaxations[item] *= 0.5;
          else relaxations[item] += 0.1 * (1.0 - relaxations[item]);
        }
        for (size_t node = 0; node < states[item].size(); ++node)
          states[item][node] += relaxations[item] * (candidate[node] - states[item][node]);
        if (iteration > 0 && residual < 100.0 * options.relative_tolerance) {
          results[item].info.status = Status::kOk; results[item].a_wb_per_m = std::move(states[item]);
          const ProfileClock::time_point finalize_start = ProfileClock::now();
          results[item] = FinalizeResult(&results[item]); active[item] = false;
          iteration_finalize_seconds += ProfileSecondsSince(finalize_start);
        } else previous_residuals[item] = residual;
      }
      if (timing != nullptr) {
        timing->finalize_seconds += iteration_finalize_seconds;
        timing->state_update_seconds +=
            std::max(0.0, ProfileSecondsSince(update_start) - iteration_finalize_seconds);
      }
    }
    for (size_t item = 0; item < results.size(); ++item) if (active[item]) {
      results[item].info.status = Status::kNonlinearSolveNotConverged;
      results[item].a_wb_per_m = std::move(states[item]);
      const ProfileClock::time_point finalize_start = ProfileClock::now();
      results[item] = FinalizeResult(&results[item]);
      if (timing != nullptr) timing->finalize_seconds += ProfileSecondsSince(finalize_start);
    }
    return results;
  }

  private:
  GpuCsrSolver csr_solver_;
  bool csr_initialized_ = false;
  size_t csr_symbolic_reuse_count_ = 0;
  NonlinearSolveResult FinalizeResult(NonlinearSolveResult* result) const
  {
    result->bx_t.assign(model_.triangles.size(), 0.0);
    result->by_t.assign(model_.triangles.size(), 0.0);
    result->circuit_flux_linkage_wb.assign(static_cast<size_t>(model_.circuit_count), 0.0);
    for (size_t element = 0; element < model_.triangles.size(); ++element) {
      const NonlinearTriangle& triangle = model_.triangles[element];
      Node p[3] = { model_.nodes[triangle.node[0]], model_.nodes[triangle.node[1]],
        model_.nodes[triangle.node[2]] };
      NonlinearElementTerms terms;
      if (!BuildElementTerms(p, &terms)) {
        result->info.status = Status::kMeshInvalid;
        return *result;
      }
      for (int i = 0; i < 3; ++i) {
        result->bx_t[element] += result->a_wb_per_m[triangle.node[i]] * terms.b_x[i];
        result->by_t[element] += result->a_wb_per_m[triangle.node[i]] * terms.b_y[i];
      }
      const NonlinearMaterial& material = model_.materials[triangle.material];
      if (material.circuit_index >= 0) {
        const size_t circuit = static_cast<size_t>(material.circuit_index);
        result->circuit_flux_linkage_wb[circuit] += terms.area_m2 * material.source_j_per_a / 3.0
            * (result->a_wb_per_m[triangle.node[0]] + result->a_wb_per_m[triangle.node[1]]
                + result->a_wb_per_m[triangle.node[2]]);
      }
    }
    for (double& flux : result->circuit_flux_linkage_wb) {
      flux *= model_.depth_m;
      if (!std::isfinite(flux))
        result->info.status = Status::kNumericalNonfinite;
    }
    if (model_.circuit_count == 1)
      result->flux_linkage_wb = result->circuit_flux_linkage_wb[0];
    return *result;
  }

  bool initialized_ = false;
  NonlinearModel model_;
};

// Strict, dependency-free reader for the MATLAB-owned gpu_femm_mesh_v1
// interchange.  This intentionally accepts only the narrow JSON subset used
// by the artifact writer; duplicate object keys, trailing data, missing
// fields, and unsupported physics fail before a model reaches the solver.
struct StrictJson {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
  Type type = Type::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<StrictJson> array;
  std::map<std::string, StrictJson> object;
};

class StrictJsonParser {
 public:
  explicit StrictJsonParser(const std::string& text) : text_(text) {}

  bool Parse(StrictJson* value, std::string* error)
  {
    if (value == nullptr || error == nullptr)
      return false;
    Skip();
    if (!Value(value, error))
      return false;
    Skip();
    if (position_ != text_.size()) {
      *error = "trailing JSON data";
      return false;
    }
    return true;
  }

 private:
  void Skip()
  {
    while (position_ < text_.size()
        && std::isspace(static_cast<unsigned char>(text_[position_])))
      ++position_;
  }

  bool Value(StrictJson* value, std::string* error)
  {
    if (position_ >= text_.size()) {
      *error = "unexpected JSON end";
      return false;
    }
    const char token = text_[position_];
    if (token == '{') return Object(value, error);
    if (token == '[') return Array(value, error);
    if (token == '"') {
      value->type = StrictJson::Type::kString;
      return String(&value->string, error);
    }
    if (text_.compare(position_, 4, "true") == 0) {
      position_ += 4; value->type = StrictJson::Type::kBool; value->boolean = true; return true;
    }
    if (text_.compare(position_, 5, "false") == 0) {
      position_ += 5; value->type = StrictJson::Type::kBool; value->boolean = false; return true;
    }
    if (text_.compare(position_, 4, "null") == 0) {
      position_ += 4; value->type = StrictJson::Type::kNull; return true;
    }
    return Number(value, error);
  }

  bool String(std::string* output, std::string* error)
  {
    if (position_ >= text_.size() || text_[position_++] != '"')
      return false;
    output->clear();
    while (position_ < text_.size()) {
      const unsigned char c = static_cast<unsigned char>(text_[position_++]);
      if (c == '"') return true;
      if (c < 0x20) { *error = "control character in JSON string"; return false; }
      if (c != '\\') { output->push_back(static_cast<char>(c)); continue; }
      if (position_ >= text_.size()) { *error = "truncated JSON escape"; return false; }
      const char escaped = text_[position_++];
      switch (escaped) {
      case '"': output->push_back('"'); break;
      case '\\': output->push_back('\\'); break;
      case '/': output->push_back('/'); break;
      case 'b': output->push_back('\b'); break;
      case 'f': output->push_back('\f'); break;
      case 'n': output->push_back('\n'); break;
      case 'r': output->push_back('\r'); break;
      case 't': output->push_back('\t'); break;
      default: *error = "unsupported JSON string escape"; return false;
      }
    }
    *error = "unterminated JSON string";
    return false;
  }

  bool Number(StrictJson* value, std::string* error)
  {
    const size_t begin = position_;
    if (position_ < text_.size() && text_[position_] == '-') ++position_;
    if (position_ >= text_.size()) { *error = "invalid JSON number"; return false; }
    if (text_[position_] == '0') ++position_;
    else if (text_[position_] >= '1' && text_[position_] <= '9')
      while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
    else { *error = "invalid JSON number"; return false; }
    if (position_ < text_.size() && text_[position_] == '.') {
      ++position_; const size_t fraction = position_;
      while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
      if (fraction == position_) { *error = "invalid JSON fraction"; return false; }
    }
    if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
      ++position_; if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
      const size_t exponent = position_;
      while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
      if (exponent == position_) { *error = "invalid JSON exponent"; return false; }
    }
    try {
      value->number = std::stod(text_.substr(begin, position_ - begin));
    } catch (...) { *error = "JSON number conversion failed"; return false; }
    if (!std::isfinite(value->number)) { *error = "nonfinite JSON number"; return false; }
    value->type = StrictJson::Type::kNumber;
    return true;
  }

  bool Array(StrictJson* value, std::string* error)
  {
    ++position_; value->type = StrictJson::Type::kArray; value->array.clear(); Skip();
    if (position_ < text_.size() && text_[position_] == ']') { ++position_; return true; }
    while (true) {
      StrictJson child;
      if (!Value(&child, error)) return false;
      value->array.push_back(std::move(child)); Skip();
      if (position_ >= text_.size()) { *error = "unterminated JSON array"; return false; }
      if (text_[position_] == ']') { ++position_; return true; }
      if (text_[position_++] != ',') { *error = "JSON array comma expected"; return false; }
      Skip();
    }
  }

  bool Object(StrictJson* value, std::string* error)
  {
    ++position_; value->type = StrictJson::Type::kObject; value->object.clear(); Skip();
    if (position_ < text_.size() && text_[position_] == '}') { ++position_; return true; }
    while (true) {
      std::string key;
      if (!String(&key, error)) { *error = "JSON object key expected"; return false; }
      Skip();
      if (position_ >= text_.size() || text_[position_++] != ':') { *error = "JSON colon expected"; return false; }
      Skip(); StrictJson child;
      if (!Value(&child, error)) return false;
      if (!value->object.emplace(std::move(key), std::move(child)).second) {
        *error = "duplicate JSON object key"; return false;
      }
      Skip();
      if (position_ >= text_.size()) { *error = "unterminated JSON object"; return false; }
      if (text_[position_] == '}') { ++position_; return true; }
      if (text_[position_++] != ',') { *error = "JSON object comma expected"; return false; }
      Skip();
    }
  }

  const std::string& text_;
  size_t position_ = 0;
};

const StrictJson* JsonMember(const StrictJson& object, const char* name,
    StrictJson::Type type, std::string* error)
{
  if (object.type != StrictJson::Type::kObject) { *error = "expected JSON object"; return nullptr; }
  const auto found = object.object.find(name);
  if (found == object.object.end()) { *error = std::string("missing JSON field: ") + name; return nullptr; }
  if (found->second.type != type) { *error = std::string("wrong JSON type: ") + name; return nullptr; }
  return &found->second;
}

bool ExactObject(const StrictJson& value, std::initializer_list<const char*> names,
    std::string* error)
{
  if (value.type != StrictJson::Type::kObject || value.object.size() != names.size()) {
    *error = "unexpected JSON object fields"; return false;
  }
  for (const char* name : names) if (value.object.find(name) == value.object.end()) {
    *error = std::string("missing JSON field: ") + name; return false;
  }
  return true;
}

bool JsonInteger(const StrictJson& value, int32_t* result)
{
  if (value.type != StrictJson::Type::kNumber || value.number != std::floor(value.number)
      || value.number < static_cast<double>(std::numeric_limits<int32_t>::min())
      || value.number > static_cast<double>(std::numeric_limits<int32_t>::max())) return false;
  *result = static_cast<int32_t>(value.number); return true;
}

bool JsonSha256(const std::string& value)
{
  return value.size() == 64 && std::all_of(value.begin(), value.end(),
      [](unsigned char c) { return std::isxdigit(c); });
}

// Small dependency-free SHA-256 implementation.  Artifact fingerprints bind
// the exact bytes on disk, not a parsed/re-serialized JSON representation.
class Sha256 {
 public:
  Sha256() { Reset(); }
  void Update(const unsigned char* data, size_t size)
  {
    total_bits_ += static_cast<uint64_t>(size) * 8;
    while (size > 0) {
        const size_t count = std::min(size, static_cast<size_t>(sizeof(buffer_) - buffered_));
      std::memcpy(buffer_ + buffered_, data, count);
      buffered_ += count; data += count; size -= count;
      if (buffered_ == sizeof(buffer_)) { Transform(buffer_); buffered_ = 0; }
    }
  }
  std::string FinalHex()
  {
    const uint64_t bits = total_bits_;
    const unsigned char one = 0x80;
    Update(&one, 1);
    const unsigned char zero = 0;
    while (buffered_ != 56) Update(&zero, 1);
    unsigned char length[8];
    for (int i = 0; i < 8; ++i) length[7 - i] = static_cast<unsigned char>(bits >> (8 * i));
    Update(length, sizeof(length));
    std::ostringstream output; output << std::hex << std::setfill('0');
    for (uint32_t word : state_)
      for (int shift = 24; shift >= 0; shift -= 8) output << std::setw(2) << ((word >> shift) & 0xffU);
    return output.str();
  }
 private:
  static uint32_t RotateRight(uint32_t value, int shift) { return (value >> shift) | (value << (32 - shift)); }
  void Reset()
  {
    state_[0]=0x6a09e667U; state_[1]=0xbb67ae85U; state_[2]=0x3c6ef372U; state_[3]=0xa54ff53aU;
    state_[4]=0x510e527fU; state_[5]=0x9b05688cU; state_[6]=0x1f83d9abU; state_[7]=0x5be0cd19U;
    total_bits_ = 0; buffered_ = 0;
  }
  void Transform(const unsigned char block[64])
  {
    static const uint32_t k[64] = { 0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U };
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = (static_cast<uint32_t>(block[4*i]) << 24) | (static_cast<uint32_t>(block[4*i+1]) << 16) | (static_cast<uint32_t>(block[4*i+2]) << 8) | block[4*i+3];
    for (int i = 16; i < 64; ++i) { const uint32_t s0=RotateRight(w[i-15],7)^RotateRight(w[i-15],18)^(w[i-15]>>3); const uint32_t s1=RotateRight(w[i-2],17)^RotateRight(w[i-2],19)^(w[i-2]>>10); w[i]=w[i-16]+s0+w[i-7]+s1; }
    uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
    for (int i = 0; i < 64; ++i) { const uint32_t s1=RotateRight(e,6)^RotateRight(e,11)^RotateRight(e,25); const uint32_t ch=(e&f)^((~e)&g); const uint32_t t1=h+s1+ch+k[i]+w[i]; const uint32_t s0=RotateRight(a,2)^RotateRight(a,13)^RotateRight(a,22); const uint32_t maj=(a&b)^(a&c)^(b&c); h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+s0+maj; }
    state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
  }
  uint32_t state_[8] = {}; unsigned char buffer_[64] = {}; size_t buffered_ = 0; uint64_t total_bits_ = 0;
};

std::string Sha256Hex(const std::string& bytes)
{
  Sha256 sha; sha.Update(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size()); return sha.FinalHex();
}

struct GpuFemmMeshArtifact {
  NonlinearModel model;
  std::vector<double> circuit_currents_a;
  std::vector<int32_t> triangle_group_numbers;
  std::string pose_fem_sha256;
  std::string base_motor_fem_sha256;
  std::string canonical_identity_sha256;
  bool has_sliding_band = false;
  double rotor_angle_deg = 0.0;
  double displacement_mm[2] = {};
};

bool ParseGpuFemmMeshArtifactJson(const std::string& text, GpuFemmMeshArtifact* artifact,
    std::string* error)
{
  if (artifact == nullptr || error == nullptr) return false;
  StrictJson root;
  StrictJsonParser parser(text);
  if (!parser.Parse(&root, error)
      || !ExactObject(root, { "schema_version", "base_motor_fem_sha256", "source_fem_sha256",
        "canonical_identity_sha256", "resolved" }, error)) return false;
  const StrictJson* schema = JsonMember(root, "schema_version", StrictJson::Type::kString, error);
  const StrictJson* root_base_sha = JsonMember(root, "base_motor_fem_sha256", StrictJson::Type::kString, error);
  const StrictJson* source_sha = JsonMember(root, "source_fem_sha256", StrictJson::Type::kString, error);
  const StrictJson* identity = JsonMember(root, "canonical_identity_sha256", StrictJson::Type::kString, error);
  const StrictJson* resolved = JsonMember(root, "resolved", StrictJson::Type::kObject, error);
  if (schema == nullptr || root_base_sha == nullptr || source_sha == nullptr || identity == nullptr || resolved == nullptr
      || (schema->string != "gpu_femm_mesh_v1" && schema->string != "gpu_femm_mesh_v2") || !JsonSha256(root_base_sha->string) || !JsonSha256(source_sha->string)
      || !JsonSha256(identity->string)
      || !(schema->string == "gpu_femm_mesh_v1"
          ? ExactObject(*resolved, { "source_fem_sha256", "base_motor_fem_sha256", "model", "pose", "nodes_mm", "triangles",
            "regions", "materials", "circuits", "outer_dirichlet" }, error)
          : ExactObject(*resolved, { "source_fem_sha256", "base_motor_fem_sha256", "model", "pose", "nodes_mm", "triangles",
            "regions", "materials", "circuits", "outer_dirichlet", "air_gap_elements" }, error))) {
    if (error->empty()) *error = "invalid gpu_femm_mesh_v1 header";
    return false;
  }
  const StrictJson* resolved_sha = JsonMember(*resolved, "source_fem_sha256", StrictJson::Type::kString, error);
  const StrictJson* base_sha = JsonMember(*resolved, "base_motor_fem_sha256", StrictJson::Type::kString, error);
  const StrictJson* model = JsonMember(*resolved, "model", StrictJson::Type::kObject, error);
  const StrictJson* pose = JsonMember(*resolved, "pose", StrictJson::Type::kObject, error);
  const StrictJson* nodes = JsonMember(*resolved, "nodes_mm", StrictJson::Type::kArray, error);
  const StrictJson* triangles = JsonMember(*resolved, "triangles", StrictJson::Type::kObject, error);
  const StrictJson* regions = JsonMember(*resolved, "regions", StrictJson::Type::kArray, error);
  const StrictJson* materials = JsonMember(*resolved, "materials", StrictJson::Type::kArray, error);
  const StrictJson* circuits = JsonMember(*resolved, "circuits", StrictJson::Type::kArray, error);
  const StrictJson* boundary = JsonMember(*resolved, "outer_dirichlet", StrictJson::Type::kObject, error);
  if (resolved_sha == nullptr || base_sha == nullptr || model == nullptr || pose == nullptr || nodes == nullptr || triangles == nullptr || regions == nullptr
      || materials == nullptr || circuits == nullptr || boundary == nullptr
      || !JsonSha256(resolved_sha->string) || !JsonSha256(base_sha->string)
      || resolved_sha->string != source_sha->string || base_sha->string != root_base_sha->string) {
    if (error->empty()) *error = "source FEM SHA mismatch";
    return false;
  }
  if (!ExactObject(*model, { "depth_mm", "problem_type", "frequency_hz" }, error)) return false;
  const StrictJson* depth = JsonMember(*model, "depth_mm", StrictJson::Type::kNumber, error);
  const StrictJson* type = JsonMember(*model, "problem_type", StrictJson::Type::kString, error);
  const StrictJson* frequency = JsonMember(*model, "frequency_hz", StrictJson::Type::kNumber, error);
  if (depth == nullptr || type == nullptr || frequency == nullptr || !(depth->number > 0.0)
      || type->string != "planar" || frequency->number != 0.0) { *error = "unsupported model physics"; return false; }
  if (!ExactObject(*pose, { "rotor_angle_deg", "displacement_mm" }, error)) return false;
  const StrictJson* rotor_angle = JsonMember(*pose, "rotor_angle_deg", StrictJson::Type::kNumber, error);
  const StrictJson* displacement = JsonMember(*pose, "displacement_mm", StrictJson::Type::kArray, error);
  if (rotor_angle == nullptr || displacement == nullptr || displacement->array.size() != 2
      || displacement->array[0].type != StrictJson::Type::kNumber
      || displacement->array[1].type != StrictJson::Type::kNumber) { *error = "invalid pose"; return false; }
  if (nodes->array.size() < 3 || circuits->array.empty() || materials->array.empty()
      || regions->array.empty()) { *error = "empty mesh/material/circuit input"; return false; }

  GpuFemmMeshArtifact parsed;
  const bool is_sliding_band_v2 = schema->string == "gpu_femm_mesh_v2";
  parsed.has_sliding_band = is_sliding_band_v2;
  parsed.pose_fem_sha256 = source_sha->string;
  parsed.base_motor_fem_sha256 = base_sha->string;
  parsed.canonical_identity_sha256 = identity->string;
  parsed.rotor_angle_deg = rotor_angle->number;
  parsed.displacement_mm[0] = displacement->array[0].number;
  parsed.displacement_mm[1] = displacement->array[1].number;
  parsed.model.depth_m = depth->number * 1e-3;
  for (const StrictJson& pair : nodes->array) {
    if (pair.type != StrictJson::Type::kArray || pair.array.size() != 2
        || pair.array[0].type != StrictJson::Type::kNumber || pair.array[1].type != StrictJson::Type::kNumber) {
      *error = "invalid node coordinate"; return false;
    }
    parsed.model.nodes.push_back({ pair.array[0].number * 1e-3, pair.array[1].number * 1e-3 });
  }
  std::map<int32_t, int32_t> material_lookup;
  for (size_t index = 0; index < materials->array.size(); ++index) {
    const StrictJson& entry = materials->array[index];
    if (!ExactObject(entry, { "id", "name", "mu_x", "mu_y", "H_c_A_per_m", "B_T", "H_A_per_m",
          "lam_type", "lam_fill" }, error)) return false;
    const StrictJson* id = JsonMember(entry, "id", StrictJson::Type::kNumber, error);
    const StrictJson* material_name = JsonMember(entry, "name", StrictJson::Type::kString, error);
    const StrictJson* mux = JsonMember(entry, "mu_x", StrictJson::Type::kNumber, error);
    const StrictJson* muy = JsonMember(entry, "mu_y", StrictJson::Type::kNumber, error);
    const StrictJson* hc = JsonMember(entry, "H_c_A_per_m", StrictJson::Type::kNumber, error);
    const StrictJson* b = JsonMember(entry, "B_T", StrictJson::Type::kArray, error);
    const StrictJson* h = JsonMember(entry, "H_A_per_m", StrictJson::Type::kArray, error);
    const StrictJson* lam_type = JsonMember(entry, "lam_type", StrictJson::Type::kNumber, error);
    const StrictJson* lam_fill = JsonMember(entry, "lam_fill", StrictJson::Type::kNumber, error);
    int32_t material_id = -1;
    if (id == nullptr || material_name == nullptr || mux == nullptr || muy == nullptr || hc == nullptr || b == nullptr || h == nullptr
        || lam_type == nullptr || lam_fill == nullptr || !JsonInteger(*id, &material_id)
        || material_id < 0 || material_name->string.empty() || material_lookup.count(material_id) != 0 || !(mux->number > 0.0)
        || mux->number != muy->number || hc->number < 0.0 || lam_type->number != 0.0
        || lam_fill->number != 1.0 || b->array.size() != h->array.size()) {
      *error = "invalid material definition"; return false;
    }
    NonlinearMaterial material;
    material.reluctivity_zero_m_per_h = 1.0 / (kMu0 * mux->number);
    material.h_c_a_per_m = hc->number;
    if (b->array.size() > 4096) { *error = "B-H curve exceeds point cap"; return false; }
    if (!b->array.empty()) {
      if (b->array.size() < 2) { *error = "B-H curve needs two points"; return false; }
      NonlinearBhCurve curve;
      for (size_t point = 0; point < b->array.size(); ++point) {
        if (b->array[point].type != StrictJson::Type::kNumber || h->array[point].type != StrictJson::Type::kNumber
            || b->array[point].number < 0.0 || h->array[point].number < 0.0
            || (point > 0 && (b->array[point].number < b->array[point - 1].number
                || h->array[point].number < h->array[point - 1].number))) { *error = "invalid B-H point"; return false; }
        curve.b_t.push_back(b->array[point].number); curve.h_a_per_m.push_back(h->array[point].number);
      }
      if (!BuildNonlinearBhSpline(&curve)) { *error = "invalid B-H spline"; return false; }
      material.bh_curve_index = static_cast<int32_t>(parsed.model.bh_curves.size());
      parsed.model.bh_curves.push_back(std::move(curve));
    }
    material_lookup.emplace(material_id, static_cast<int32_t>(parsed.model.materials.size()));
    parsed.model.materials.push_back(material);
  }
  // Regions are expanded into per-label material entries so PM angle, FEMM
  // group, and coil source are never accidentally shared across labels.
  std::map<int32_t, int32_t> circuit_lookup;
  for (size_t index = 0; index < circuits->array.size(); ++index) {
    const StrictJson& entry = circuits->array[index];
    if (!ExactObject(entry, { "index", "name", "type", "current_A" }, error)) return false;
    const StrictJson* id = JsonMember(entry, "index", StrictJson::Type::kNumber, error);
    const StrictJson* name = JsonMember(entry, "name", StrictJson::Type::kString, error);
    const StrictJson* circuit_type = JsonMember(entry, "type", StrictJson::Type::kString, error);
    const StrictJson* current = JsonMember(entry, "current_A", StrictJson::Type::kNumber, error);
    int32_t circuit = -1;
    if (id == nullptr || name == nullptr || circuit_type == nullptr || current == nullptr || !JsonInteger(*id, &circuit)
        || circuit != static_cast<int32_t>(index) || name->string.empty() || circuit_type->string != "series") {
      *error = "circuits must be ordered contiguous current-driven series circuits"; return false;
    }
    circuit_lookup.emplace(circuit, circuit);
    parsed.circuit_currents_a.push_back(current->number);
  }
  parsed.model.circuit_count = static_cast<int32_t>(parsed.circuit_currents_a.size());
  std::map<int32_t, int32_t> region_lookup;
  std::vector<double> region_area_m2;
  struct RegionRaw { int32_t material; int32_t circuit; double turns; double angle; int32_t group; };
  std::vector<RegionRaw> raw_regions;
  for (const StrictJson& entry : regions->array) {
    if (!ExactObject(entry, { "id", "group_number", "material_id", "circuit_index", "turns", "pm_magnetization_deg" }, error)) return false;
    const StrictJson* id = JsonMember(entry, "id", StrictJson::Type::kNumber, error);
    const StrictJson* group = JsonMember(entry, "group_number", StrictJson::Type::kNumber, error);
    const StrictJson* material = JsonMember(entry, "material_id", StrictJson::Type::kNumber, error);
    const StrictJson* circuit = JsonMember(entry, "circuit_index", StrictJson::Type::kNumber, error);
    const StrictJson* turns = JsonMember(entry, "turns", StrictJson::Type::kNumber, error);
    const StrictJson* angle = JsonMember(entry, "pm_magnetization_deg", StrictJson::Type::kNumber, error);
    int32_t region_id = -1, group_id = -1, material_id = -1, circuit_id = -2;
    if (id == nullptr || group == nullptr || material == nullptr || circuit == nullptr || turns == nullptr || angle == nullptr
        || !JsonInteger(*id, &region_id) || !JsonInteger(*group, &group_id) || !JsonInteger(*material, &material_id)
        || !JsonInteger(*circuit, &circuit_id) || region_id < 0 || group_id < 0 || material_lookup.count(material_id) == 0
        || region_lookup.count(region_id) != 0 || circuit_id < -1
        || (circuit_id == -1 && turns->number != 0.0)
        || (circuit_id >= 0 && (circuit_lookup.count(circuit_id) == 0 || turns->number == 0.0))) {
      *error = "invalid region definition"; return false;
    }
    region_lookup.emplace(region_id, static_cast<int32_t>(raw_regions.size()));
    raw_regions.push_back({ material_lookup[material_id], circuit_id, turns->number, angle->number, group_id });
    region_area_m2.push_back(0.0);
  }
  if (!ExactObject(*triangles, { "node_indices", "region_ids" }, error)) return false;
  const StrictJson* faces = JsonMember(*triangles, "node_indices", StrictJson::Type::kArray, error);
  const StrictJson* triangle_regions = JsonMember(*triangles, "region_ids", StrictJson::Type::kArray, error);
  if (faces == nullptr || triangle_regions == nullptr || faces->array.empty() || faces->array.size() != triangle_regions->array.size()) {
    *error = "invalid triangle arrays"; return false;
  }
  std::vector<int32_t> mapped_regions;
  for (size_t element = 0; element < faces->array.size(); ++element) {
    if (faces->array[element].type != StrictJson::Type::kArray || faces->array[element].array.size() != 3) {
      *error = "triangle must have three node indices"; return false;
    }
    int32_t node[3], region_id = -1;
    if (triangle_regions->array[element].type != StrictJson::Type::kNumber || !JsonInteger(triangle_regions->array[element], &region_id)
        || region_lookup.count(region_id) == 0) { *error = "triangle references unknown region"; return false; }
    for (int local = 0; local < 3; ++local) {
      if (!JsonInteger(faces->array[element].array[local], &node[local]) || node[local] < 0
          || static_cast<size_t>(node[local]) >= parsed.model.nodes.size()) { *error = "triangle node out of range"; return false; }
    }
    Node p[3] = { parsed.model.nodes[node[0]], parsed.model.nodes[node[1]], parsed.model.nodes[node[2]] };
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms)) { *error = "triangle is not CCW"; return false; }
    const int32_t mapped = region_lookup[region_id];
    region_area_m2[mapped] += terms.area_m2;
    mapped_regions.push_back(mapped);
    parsed.model.triangles.push_back({ { node[0], node[1], node[2] }, mapped });
  }
  for (size_t region = 0; region < raw_regions.size(); ++region) {
    const RegionRaw& raw = raw_regions[region];
    if (!(region_area_m2[region] > 0.0)) { *error = "region has no triangle area"; return false; }
    const NonlinearMaterial base = parsed.model.materials[raw.material];
    NonlinearMaterial label = base;
    label.group_number = raw.group;
    label.circuit_index = raw.circuit;
    label.magnetization_deg = raw.angle;
    label.source_j_per_a = raw.circuit < 0 ? 0.0 : raw.turns / region_area_m2[region];
    parsed.model.materials.push_back(label);
  }
  // Replace temporary region indices by the expanded per-label material IDs.
  const int32_t labels_begin = static_cast<int32_t>(parsed.model.materials.size() - raw_regions.size());
  for (NonlinearTriangle& triangle : parsed.model.triangles) {
    triangle.material = labels_begin + triangle.material;
    parsed.triangle_group_numbers.push_back(parsed.model.materials[triangle.material].group_number);
  }
  if (!ExactObject(*boundary, { "node_indices", "A_Wb_per_m" }, error)) return false;
  const StrictJson* boundary_nodes = JsonMember(*boundary, "node_indices", StrictJson::Type::kArray, error);
  const StrictJson* boundary_values = JsonMember(*boundary, "A_Wb_per_m", StrictJson::Type::kArray, error);
  if (boundary_nodes == nullptr || boundary_values == nullptr || boundary_nodes->array.empty()
      || boundary_nodes->array.size() != boundary_values->array.size()) { *error = "invalid outer Dirichlet"; return false; }
  for (size_t index = 0; index < boundary_nodes->array.size(); ++index) {
    int32_t node = -1;
    if (!JsonInteger(boundary_nodes->array[index], &node) || node < 0
        || static_cast<size_t>(node) >= parsed.model.nodes.size()
        || boundary_values->array[index].type != StrictJson::Type::kNumber
        || boundary_values->array[index].number != 0.0) { *error = "outer Dirichlet must be zero and in range"; return false; }
    parsed.model.dirichlet_nodes.push_back(node);
    parsed.model.dirichlet_a_wb_per_m.push_back(0.0);
  }
  if (is_sliding_band_v2) {
    const auto ages_it = resolved->object.find("air_gap_elements");
    if (ages_it == resolved->object.end() || (ages_it->second.type != StrictJson::Type::kArray
            && ages_it->second.type != StrictJson::Type::kObject)
        || parsed.displacement_mm[0] != 0.0
        || parsed.displacement_mm[1] != 0.0 || parsed.rotor_angle_deg != 0.0) {
      if (error->empty()) *error = "sliding-band artifacts require a centered zero-degree reference mesh";
      return false;
    }
    // MATLAB's jsonencode represents a scalar struct as an object rather
    // than a one-element array.  Accept that canonical single-AGE spelling
    // while retaining the array form for multi-AGE artifacts.
    std::vector<const StrictJson*> age_entries;
    if (ages_it->second.type == StrictJson::Type::kArray) {
      if (ages_it->second.array.empty()) { *error = "sliding-band requires an air-gap element"; return false; }
      for (const StrictJson& entry : ages_it->second.array) age_entries.push_back(&entry);
    } else {
      age_entries.push_back(&ages_it->second);
    }
    for (const StrictJson* entry_ptr : age_entries) {
      const StrictJson& entry = *entry_ptr;
      if (!ExactObject(entry, { "name", "periodicity", "center_mm", "ri_mm", "ro_mm", "arc_length_deg",
            "sector_count", "inner_shift", "outer_shift", "quad_points" }, error)) return false;
      const StrictJson* name = JsonMember(entry, "name", StrictJson::Type::kString, error);
      const StrictJson* periodicity = JsonMember(entry, "periodicity", StrictJson::Type::kString, error);
      const StrictJson* center = JsonMember(entry, "center_mm", StrictJson::Type::kArray, error);
      const StrictJson* ri = JsonMember(entry, "ri_mm", StrictJson::Type::kNumber, error);
      const StrictJson* ro = JsonMember(entry, "ro_mm", StrictJson::Type::kNumber, error);
      const StrictJson* arc = JsonMember(entry, "arc_length_deg", StrictJson::Type::kNumber, error);
      const StrictJson* sectors = JsonMember(entry, "sector_count", StrictJson::Type::kNumber, error);
      const StrictJson* inner_shift = JsonMember(entry, "inner_shift", StrictJson::Type::kNumber, error);
      const StrictJson* outer_shift = JsonMember(entry, "outer_shift", StrictJson::Type::kNumber, error);
      const StrictJson* points = JsonMember(entry, "quad_points", StrictJson::Type::kArray, error);
      int32_t sector_count = 0;
      if (name == nullptr || name->string.empty() || periodicity == nullptr || center == nullptr || ri == nullptr || ro == nullptr
          || arc == nullptr || sectors == nullptr || inner_shift == nullptr || outer_shift == nullptr || points == nullptr
          || center->array.size() != 2 || center->array[0].type != StrictJson::Type::kNumber
          || center->array[1].type != StrictJson::Type::kNumber || !JsonInteger(*sectors, &sector_count)
          || sector_count < 2 || points->array.size() != static_cast<size_t>(sector_count + 1)
          || (periodicity->string != "periodic" && periodicity->string != "antiperiodic")) {
        *error = "invalid sliding-band air-gap element"; return false;
      }
      AirGapElement age;
      age.antiperiodic = periodicity->string == "antiperiodic";
      age.center_x_m = center->array[0].number * 1e-3; age.center_y_m = center->array[1].number * 1e-3;
      age.inner_radius_m = ri->number * 1e-3; age.outer_radius_m = ro->number * 1e-3;
      age.arc_length_deg = arc->number; age.inner_shift = inner_shift->number; age.outer_shift = outer_shift->number;
      for (const StrictJson& point : points->array) {
        if (!ExactObject(point, { "n0", "w0", "n1", "w1", "n2", "w2", "n3", "w3" }, error)) return false;
        AirGapQuadPoint parsed_point;
        for (int local = 0; local < 4; ++local) {
          const std::string node_name = std::string("n") + char('0' + local);
          const std::string weight_name = std::string("w") + char('0' + local);
          const StrictJson* node = JsonMember(point, node_name.c_str(), StrictJson::Type::kNumber, error);
          const StrictJson* weight = JsonMember(point, weight_name.c_str(), StrictJson::Type::kNumber, error);
          if (node == nullptr || weight == nullptr || !JsonInteger(*node, &parsed_point.node[local])) {
            *error = "invalid sliding-band quadrature point"; return false;
          }
          parsed_point.weight[local] = weight->number;
        }
        age.quad_points.push_back(parsed_point);
      }
      parsed.model.air_gap_elements.push_back(std::move(age));
    }
  }
  if (ValidateNonlinearModel(parsed.model) != Status::kOk) { *error = "mapped mesh model validation failed"; return false; }
  *artifact = std::move(parsed);
  return true;
}

bool ReadGpuFemmMeshArtifact(const std::string& path, GpuFemmMeshArtifact* artifact,
    std::string* error)
{
  std::ifstream input(path);
  std::stringstream contents; contents << input.rdbuf();
  if (!input) { *error = "cannot open mesh artifact"; return false; }
  return ParseGpuFemmMeshArtifactJson(contents.str(), artifact, error);
}

int GpuFemmMeshArtifactTest(const std::string& path)
{
  GpuFemmMeshArtifact artifact;
  std::string error;
  if (!ReadGpuFemmMeshArtifact(path, &artifact, &error)) {
    std::cerr << "FAIL gpu_femm_mesh_v1: " << error << '\n';
    return 1;
  }
  std::cout << "PASS gpu_femm_mesh_v1\n"
            << "  nodes=" << artifact.model.nodes.size()
            << " triangles=" << artifact.model.triangles.size()
            << " circuits=" << artifact.model.circuit_count
            << " pose_fem_sha256=" << artifact.pose_fem_sha256
            << " base_motor_fem_sha256=" << artifact.base_motor_fem_sha256 << '\n';
  return 0;
}

// fkn assembles magnetics in centimetres and writes A = (100 * mu0) V.
constexpr double kFemmInternalToPhysicalA = 100.0 * kMu0;

// Package 3B frozen-scope postprocessing.  The production MATLAB path uses
// mo_blockintegral(18/19/22) over the rotor and mo_getb() in the air gap.
// Keep this host-side until its FEMM parity is established; it consumes the
// same P1 solution that the CUDA linear solve produced.
struct AirgapSample {
  double angle_rad = 0.0;
  double radial_b_t = std::numeric_limits<double>::quiet_NaN();
};

struct FrozenPostprocessOptions {
  // Fixture mapping is explicit: label 1 is PM/selected rotor, label 3 is
  // default air.  This avoids silently treating an unassigned material as air.
  int32_t selected_material_label = 1;
  int32_t air_material_label = 3;
  // Artifact mode selects complete FEMM label groups; -1 retains frozen
  // material-label semantics above.
  int32_t selected_group_number = -1;
  int32_t air_group_number = -1;
  double airgap_radius_m = 0.0;
  std::vector<double> airgap_angles_rad;
  int max_mask_iterations = 256;
  double mask_relative_tolerance = 1e-12;
};

bool IsSelectedPostprocessMaterial(const NonlinearMaterial& material,
    const FrozenPostprocessOptions& options, int32_t material_index)
{
  return options.selected_group_number >= 0
      ? material.group_number == options.selected_group_number
      : material_index == options.selected_material_label;
}

bool IsAirPostprocessMaterial(const NonlinearMaterial& material,
    const FrozenPostprocessOptions& options, int32_t material_index)
{
  return options.air_group_number >= 0
      ? material.group_number == options.air_group_number
      : material_index == options.air_material_label;
}

struct FrozenPostprocessResult {
  Status status = Status::kInternalError;
  double force_x_n = std::numeric_limits<double>::quiet_NaN();
  double force_y_n = std::numeric_limits<double>::quiet_NaN();
  double torque_nm = std::numeric_limits<double>::quiet_NaN();
  std::vector<AirgapSample> airgap_samples;
};

using EdgeKey = std::pair<int32_t, int32_t>;

EdgeKey CanonicalEdge(int32_t first, int32_t second)
{
  return { std::min(first, second), std::max(first, second) };
}

bool TriangleBarycentric(const Node p[3], double x_m, double y_m, double lambda[3])
{
  const double determinant = (p[1].x_m - p[0].x_m) * (p[2].y_m - p[0].y_m)
      - (p[2].x_m - p[0].x_m) * (p[1].y_m - p[0].y_m);
  if (!(determinant > 0.0) || !std::isfinite(determinant))
    return false;
  lambda[1] = ((x_m - p[0].x_m) * (p[2].y_m - p[0].y_m)
      - (p[2].x_m - p[0].x_m) * (y_m - p[0].y_m)) / determinant;
  lambda[2] = ((p[1].x_m - p[0].x_m) * (y_m - p[0].y_m)
      - (x_m - p[0].x_m) * (p[1].y_m - p[0].y_m)) / determinant;
  lambda[0] = 1.0 - lambda[1] - lambda[2];
  return std::isfinite(lambda[0]) && std::isfinite(lambda[1]) && std::isfinite(lambda[2]);
}

Status BuildWeightedStressMask(const NonlinearModel& model,
    const FrozenPostprocessOptions& options, std::vector<double>* mask)
{
  if (mask == nullptr || ((options.selected_group_number < 0)
          && (options.selected_material_label < 0 || static_cast<size_t>(options.selected_material_label) >= model.materials.size()))
      || ((options.air_group_number < 0)
          && (options.air_material_label < 0 || static_cast<size_t>(options.air_material_label) >= model.materials.size()))
      || (options.selected_group_number >= 0 && options.selected_group_number == options.air_group_number)
      || (options.selected_group_number < 0 && options.air_group_number < 0
          && options.selected_material_label == options.air_material_label)
      || !(options.mask_relative_tolerance > 0.0) || !std::isfinite(options.mask_relative_tolerance)
      || options.max_mask_iterations <= 0)
    return Status::kInvalidArgument;

  const Status validation = ValidateNonlinearModel(model);
  if (validation != Status::kOk)
    return validation;
  const size_t node_count = model.nodes.size();
  std::map<EdgeKey, std::vector<int32_t>> edge_elements;
  bool selected_seen = false;
  for (size_t element = 0; element < model.triangles.size(); ++element) {
    const NonlinearTriangle& triangle = model.triangles[element];
    selected_seen = selected_seen || IsSelectedPostprocessMaterial(
        model.materials[triangle.material], options, triangle.material);
    for (int local = 0; local < 3; ++local) {
      edge_elements[CanonicalEdge(triangle.node[local], triangle.node[(local + 1) % 3])]
          .push_back(static_cast<int32_t>(element));
    }
  }
  if (!selected_seen)
    return Status::kInvalidMaterial;
  // FEMM's MakeMask rejects selections that touch a non-free-space region.
  for (const auto& edge : edge_elements) {
    const std::vector<int32_t>& elements = edge.second;
    bool selected = false;
    bool other_non_air = false;
    for (const int32_t element : elements) {
      const int32_t material = model.triangles[element].material;
      const NonlinearMaterial& properties = model.materials[material];
      selected = selected || IsSelectedPostprocessMaterial(properties, options, material);
      other_non_air = other_non_air || (!IsSelectedPostprocessMaterial(properties, options, material)
          && !IsAirPostprocessMaterial(properties, options, material));
    }
    if (selected && other_non_air)
      return Status::kInvalidMaterial;
  }

  std::vector<double> fixed(node_count, std::numeric_limits<double>::quiet_NaN());
  for (const int32_t boundary : model.dirichlet_nodes)
    fixed[boundary] = 0.0;
  for (const NonlinearTriangle& triangle : model.triangles) {
    const NonlinearMaterial& material = model.materials[triangle.material];
    if (IsSelectedPostprocessMaterial(material, options, triangle.material)) {
      for (const int32_t node : triangle.node) {
        if (std::isfinite(fixed[node]) && fixed[node] != 1.0)
          return Status::kBoundaryInvalid;
        fixed[node] = 1.0;
      }
    } else if (!IsAirPostprocessMaterial(material, options, triangle.material)) {
      for (const int32_t node : triangle.node) {
        if (!std::isfinite(fixed[node]))
          fixed[node] = 0.0;
      }
    }
  }

  SparseRows stiffness(node_count);
  for (const NonlinearTriangle& triangle : model.triangles) {
    Node p[3] = { model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
      model.nodes[triangle.node[2]] };
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms))
      return Status::kMeshInvalid;
    // FEMM default WeightingScheme=0 uses sqrt(label.MaxArea) when supplied,
    // otherwise sqrt(element area).  The frozen fixture has MaxArea=-1 for
    // every label; changing mm to SI only introduces a common factor and
    // therefore leaves the Dirichlet mask solution unchanged.
    const double mask_weight = std::sqrt(terms.area_m2);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        const Status added = AddSparseEntry(&stiffness, triangle.node[row], triangle.node[column],
            mask_weight * terms.area_m2 * (terms.b_x[row] * terms.b_x[column]
                + terms.b_y[row] * terms.b_y[column]));
        if (added != Status::kOk)
          return added;
      }
    }
  }
  const Status sparse_validation = ValidateSparseRows(stiffness);
  if (sparse_validation != Status::kOk)
    return sparse_validation;
  Assembly assembly;
  assembly.boundary_values.assign(node_count, 0.0);
  for (size_t node = 0; node < node_count; ++node) {
    if (std::isfinite(fixed[node]))
      assembly.boundary_values[node] = fixed[node];
    else
      assembly.free_nodes.push_back(static_cast<int32_t>(node));
  }
  if (assembly.free_nodes.empty()) {
    mask->assign(node_count, 0.0);
    for (size_t node = 0; node < node_count; ++node)
      (*mask)[node] = fixed[node] > 0.5 ? 1.0 : 0.0;
    return Status::kOk;
  }
  const size_t free_count = assembly.free_nodes.size();
  std::vector<int32_t> free_index(node_count, -1);
  for (size_t row = 0; row < free_count; ++row)
    free_index[assembly.free_nodes[row]] = static_cast<int32_t>(row);
  assembly.row_offsets.assign(free_count + 1, 0);
  assembly.diagonal.assign(free_count, 0.0);
  std::vector<double> rhs(free_count, 0.0);
  for (size_t row = 0; row < free_count; ++row) {
    const int32_t global_row = assembly.free_nodes[row];
    assembly.row_offsets[row] = static_cast<int32_t>(assembly.values.size());
    for (const auto& entry : stiffness[global_row]) {
      const int32_t global_column = entry.first;
      const double value = entry.second;
      if (std::isfinite(fixed[global_column])) {
        rhs[row] -= value * fixed[global_column];
      } else if (value != 0.0) {
        const int32_t column = free_index[global_column];
        if (column < 0)
          return Status::kAssemblyFailed;
        assembly.column_indices.push_back(column);
        assembly.values.push_back(value);
        if (static_cast<size_t>(column) == row)
          assembly.diagonal[row] = value;
      }
    }
    if (!std::isfinite(rhs[row]))
      return Status::kNumericalNonfinite;
    if (!(assembly.diagonal[row] > 0.0) || !std::isfinite(assembly.diagonal[row]))
      return Status::kAssemblyFailed;
  }
  assembly.row_offsets[free_count] = static_cast<int32_t>(assembly.values.size());
  GpuCsrSolver solver;
  const Status initialize = solver.Initialize(assembly);
  if (initialize != Status::kOk)
    return initialize;
  std::vector<double> free_mask;
  const SolveInfo info = solver.Solve(rhs, options.mask_relative_tolerance,
      options.max_mask_iterations, &free_mask);
  if (info.status != Status::kOk)
    return info.status;
  mask->assign(node_count, 0.0);
  for (size_t node = 0; node < node_count; ++node)
    (*mask)[node] = std::isfinite(fixed[node]) ? fixed[node] : 0.0;
  for (size_t row = 0; row < free_count; ++row)
    (*mask)[assembly.free_nodes[row]] = free_mask[row];
  // FEMM's default WeightingScheme=0 does not use the continuous harmonic
  // mask directly: it thresholds every solved node at V>0.5.
  for (double& value : *mask)
    value = value > 0.5 ? 1.0 : 0.0;
  return Status::kOk;
}

// Exact DC subset of FEMM GetNodalB: first use inverse-centroid-distance
// averaging in an equivalent-material patch.  Otherwise traverse CCW/CW from
// the element around the node, preserve Bn from A and Bt from the source-side
// element at each interface, then use FEMM's sharp-corner fallback.
bool SmoothedElementB(const NonlinearModel& model, const std::vector<double>& a,
    const std::vector<double>& bx, const std::vector<double>& by,
    const std::vector<std::vector<int32_t>>& incident,
    const std::map<EdgeKey, std::vector<int32_t>>& edge_elements, int32_t element,
    double nodal_bx[3], double nodal_by[3])
{
  if (element < 0 || static_cast<size_t>(element) >= model.triangles.size()
      || a.size() != model.nodes.size() || bx.size() != model.triangles.size()
      || by.size() != model.triangles.size())
    return false;
  (void)edge_elements;
  const NonlinearTriangle& triangle = model.triangles[element];
  for (int local = 0; local < 3; ++local) {
    const int32_t node = triangle.node[local];
    bool homogeneous = true;
    for (const int32_t near : incident[node])
      // FEMM switches to its interface traversal at a block-label boundary,
      // even when the two labels happen to reference equivalent materials.
      homogeneous = homogeneous
          && model.triangles[element].material == model.triangles[near].material;
    if (homogeneous) {
      double weight_sum = 0.0;
      nodal_bx[local] = 0.0;
      nodal_by[local] = 0.0;
      for (const int32_t near : incident[node]) {
        const NonlinearTriangle& candidate = model.triangles[near];
        const Node& p0 = model.nodes[candidate.node[0]];
        const Node& p1 = model.nodes[candidate.node[1]];
        const Node& p2 = model.nodes[candidate.node[2]];
        const double cx = (p0.x_m + p1.x_m + p2.x_m) / 3.0;
        const double cy = (p0.y_m + p1.y_m + p2.y_m) / 3.0;
        const double weight = 1.0 / std::hypot(model.nodes[node].x_m - cx,
            model.nodes[node].y_m - cy);
        if (!(weight > 0.0) || !std::isfinite(weight))
          return false;
        weight_sum += weight;
        nodal_bx[local] += weight * bx[near];
        nodal_by[local] += weight * by[near];
      }
      nodal_bx[local] /= weight_sum;
      nodal_by[local] /= weight_sum;
      continue;
    }

    auto local_index = [&model, node](int32_t candidate) {
      for (int index = 0; index < 3; ++index)
        if (model.triangles[candidate].node[index] == node)
          return index;
      return -1;
    };
    auto next_across = [&model, &incident, node](int32_t current, int32_t other) {
      for (const int32_t candidate : incident[node]) {
        if (candidate == current)
          continue;
        for (int index = 0; index < 3; ++index)
          if (model.triangles[candidate].node[index] == other)
            return candidate;
      }
      return static_cast<int32_t>(-1);
    };
    double weight_sum = 0.0;
    nodal_bx[local] = 0.0;
    nodal_by[local] = 0.0;
    double v1x = 0.0, v1y = 0.0, v2x = 0.0, v2y = 0.0;
    bool special_case = false;
    auto add_interface = [&](int32_t source, int32_t other) {
      const double dx = model.nodes[other].x_m - model.nodes[node].x_m;
      const double dy = model.nodes[other].y_m - model.nodes[node].y_m;
      const double length = std::hypot(dx, dy);
      if (!(length > 0.0))
        return false;
      const double tx = dx / length;
      const double ty = dy / length;
      const double bt = bx[source] * tx + by[source] * ty;
      const double bn = (a[other] - a[node]) / length;
      const double weight = 0.5 / length;
      nodal_bx[local] += weight * (tx * bt + ty * bn);
      nodal_by[local] += weight * (ty * bt - tx * bn);
      weight_sum += weight;
      return true;
    };
    int32_t current = element;
    for (size_t scan = 0; scan < incident[node].size(); ++scan) {
      const int index = local_index(current);
      if (index < 0)
        return false;
      const int32_t other = model.triangles[current].node[(index + 2) % 3];
      const int32_t next = next_across(current, other);
      if (next < 0) {
        nodal_bx[local] = bx[current];
        nodal_by[local] = by[current];
        special_case = true;
        break;
      }
      if (model.triangles[element].material != model.triangles[next].material) {
        if (!add_interface(current, other))
          return false;
        const double length = std::hypot(model.nodes[other].x_m - model.nodes[node].x_m,
            model.nodes[other].y_m - model.nodes[node].y_m);
        v1x = (model.nodes[other].x_m - model.nodes[node].x_m) / length;
        v1y = (model.nodes[other].y_m - model.nodes[node].y_m) / length;
        break;
      }
      current = next;
    }
    if (!special_case) {
      current = element;
      for (size_t scan = 0; scan < incident[node].size(); ++scan) {
        const int index = local_index(current);
        if (index < 0)
          return false;
        const int32_t other = model.triangles[current].node[(index + 1) % 3];
        const int32_t next = next_across(current, other);
        if (next < 0) {
          nodal_bx[local] = bx[current];
          nodal_by[local] = by[current];
          special_case = true;
          break;
        }
        if (model.triangles[element].material != model.triangles[next].material) {
          if (!add_interface(current, other))
            return false;
          const double length = std::hypot(model.nodes[other].x_m - model.nodes[node].x_m,
              model.nodes[other].y_m - model.nodes[node].y_m);
          v2x = (model.nodes[other].x_m - model.nodes[node].x_m) / length;
          v2y = (model.nodes[other].y_m - model.nodes[node].y_m) / length;
          break;
        }
        current = next;
      }
    }
    if (!special_case) {
      if (!(weight_sum > 0.0))
        return false;
      nodal_bx[local] /= weight_sum;
      nodal_by[local] /= weight_sum;
      const bool simple_corner = std::hypot(v1x, v1y) < 0.9 || std::hypot(v2x, v2y) < 0.9
          || (-v1x * v2x - v1y * v2y) > 0.985;
      if (!simple_corner) {
        double magnitude = 0.0;
        for (const int32_t near : incident[node]) {
          if (model.triangles[near].material == triangle.material)
            magnitude = std::max(magnitude, std::hypot(bx[near], by[near]));
        }
        const double own_magnitude = std::hypot(bx[element], by[element]);
        if (own_magnitude > 0.0) {
          nodal_bx[local] = magnitude * bx[element] / own_magnitude;
          nodal_by[local] = magnitude * by[element] / own_magnitude;
        } else {
          nodal_bx[local] = 0.0;
          nodal_by[local] = 0.0;
        }
      }
    }
  }
  return true;
}

FrozenPostprocessResult ComputeFrozenPostprocess(const NonlinearModel& model,
    const NonlinearSolveResult& solution, const FrozenPostprocessOptions& options)
{
  FrozenPostprocessResult result;
  result.force_x_n = 0.0;
  result.force_y_n = 0.0;
  result.torque_nm = 0.0;
  if (solution.a_wb_per_m.size() != model.nodes.size()
      || solution.bx_t.size() != model.triangles.size()
      || solution.by_t.size() != model.triangles.size() || !(model.depth_m > 0.0)) {
    result.status = Status::kInvalidArgument;
    return result;
  }
  std::vector<double> mask;
  result.status = BuildWeightedStressMask(model, options, &mask);
  if (result.status != Status::kOk)
    return result;
  std::map<EdgeKey, std::vector<int32_t>> edge_elements;
  std::vector<std::vector<int32_t>> incident(model.nodes.size());
  for (size_t element = 0; element < model.triangles.size(); ++element) {
    const NonlinearTriangle& triangle = model.triangles[element];
    for (int local = 0; local < 3; ++local) {
      incident[triangle.node[local]].push_back(static_cast<int32_t>(element));
      edge_elements[CanonicalEdge(triangle.node[local], triangle.node[(local + 1) % 3])]
          .push_back(static_cast<int32_t>(element));
    }
  }
  for (size_t element = 0; element < model.triangles.size(); ++element) {
    const NonlinearTriangle& triangle = model.triangles[element];
    Node p[3] = { model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
      model.nodes[triangle.node[2]] };
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms)) {
      result.status = Status::kMeshInvalid;
      return result;
    }
    double hx = 0.0, hy = 0.0;
    for (int local = 0; local < 3; ++local) {
      // HenrotteVector is -grad(mask).  b=(dN/dy,-dN/dx), hence
      // -grad(m)=(sum m*b_y, -sum m*b_x).
      hx += mask[triangle.node[local]] * terms.b_y[local];
      hy -= mask[triangle.node[local]] * terms.b_x[local];
    }
    const double bx = solution.bx_t[element], by = solution.by_t[element];
    const double fx_density = ((bx * bx - by * by) * hx + 2.0 * bx * by * hy)
        / (2.0 * kMu0);
    const double fy_density = (2.0 * bx * by * hx + (by * by - bx * bx) * hy)
        / (2.0 * kMu0);
    const double weight = terms.area_m2 * model.depth_m;
    result.force_x_n += weight * fx_density;
    result.force_y_n += weight * fy_density;
    const double cx = (p[0].x_m + p[1].x_m + p[2].x_m) / 3.0;
    const double cy = (p[0].y_m + p[1].y_m + p[2].y_m) / 3.0;
    result.torque_nm += weight * (cx * fy_density - cy * fx_density);
  }
  for (const double angle : options.airgap_angles_rad) {
    if (!std::isfinite(angle) || !(options.airgap_radius_m >= 0.0)
        || !std::isfinite(options.airgap_radius_m)) {
      result.status = Status::kInvalidArgument;
      return result;
    }
    const double x = options.airgap_radius_m * std::cos(angle);
    const double y = options.airgap_radius_m * std::sin(angle);
    int32_t containing = -1;
    double lambda[3] = {};
    for (size_t element = 0; element < model.triangles.size(); ++element) {
      const NonlinearTriangle& triangle = model.triangles[element];
      Node p[3] = { model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
        model.nodes[triangle.node[2]] };
      double candidate[3] = {};
      if (TriangleBarycentric(p, x, y, candidate) && candidate[0] >= -1e-12
          && candidate[1] >= -1e-12 && candidate[2] >= -1e-12) {
        containing = static_cast<int32_t>(element);
        std::copy(candidate, candidate + 3, lambda);
        break;
      }
    }
    if (containing < 0 || !IsAirPostprocessMaterial(
            model.materials[model.triangles[containing].material], options,
            model.triangles[containing].material)) {
      result.status = Status::kInvalidArgument;
      return result;
    }
    double nodal_bx[3] = {}, nodal_by[3] = {};
    if (!SmoothedElementB(model, solution.a_wb_per_m, solution.bx_t, solution.by_t,
            incident, edge_elements, containing, nodal_bx, nodal_by)) {
      result.status = Status::kMeshInvalid;
      return result;
    }
    double bx = 0.0, by = 0.0;
    for (int local = 0; local < 3; ++local) {
      bx += lambda[local] * nodal_bx[local];
      by += lambda[local] * nodal_by[local];
    }
    result.airgap_samples.push_back({ angle, bx * std::cos(angle) + by * std::sin(angle) });
  }
  if (!std::isfinite(result.force_x_n) || !std::isfinite(result.force_y_n)
      || !std::isfinite(result.torque_nm))
    result.status = Status::kNumericalNonfinite;
  return result;
}

bool BuildAirGapStencil(const AirGapElement& age, size_t k, int32_t node[10], double weight[10])
{
  const size_t elements = age.quad_points.size() - 1;
  if (elements < 2 || k >= elements) return false;
  const size_t previous = k == 0 ? elements - 1 : k - 1;
  const size_t next = k + 1;
  const size_t next2 = k + 2 > elements ? 1 : k + 2;
  const AirGapQuadPoint& before = age.quad_points[previous];
  const AirGapQuadPoint& current = age.quad_points[k];
  const AirGapQuadPoint& after = age.quad_points[next];
  const AirGapQuadPoint& after2 = age.quad_points[next2];
  const int local_node[10] = { before.node[0], current.node[0], current.node[1], after.node[1], after2.node[1],
    before.node[2], current.node[2], current.node[3], after.node[3], after2.node[3] };
  const double local_weight[10] = { before.weight[0], current.weight[0], current.weight[1], after.weight[1], after2.weight[1],
    before.weight[2], current.weight[2], current.weight[3], after.weight[3], after2.weight[3] };
  std::copy(local_node, local_node + 10, node); std::copy(local_weight, local_weight + 10, weight);
  if (age.antiperiodic && k == 0) { weight[0] = -weight[0]; weight[5] = -weight[5]; }
  if (age.antiperiodic && k + 1 == elements) { weight[4] = -weight[4]; weight[9] = -weight[9]; }
  return true;
}

FrozenPostprocessResult ComputeAirGapElementPostprocess(const NonlinearModel& model,
    const NonlinearSolveResult& solution, const FrozenPostprocessOptions& options)
{
  FrozenPostprocessResult result;
  result.status = Status::kOk;
  result.force_x_n = 0.0;
  result.force_y_n = 0.0;
  result.torque_nm = 0.0;
  if (model.air_gap_elements.empty() || solution.a_wb_per_m.size() != model.nodes.size()) {
    result.status = Status::kInvalidArgument; return result;
  }
  struct AirGapField {
    const AirGapElement* age = nullptr;
    std::vector<double> br, bt;
    std::vector<int32_t> harmonic_order;
    std::vector<double> br_cos, br_sin;
  };
  std::vector<AirGapField> fields;
  for (const AirGapElement& age : model.air_gap_elements) {
    const size_t elements = age.quad_points.size() - 1;
    const double dt = (3.141592653589793238462643383279502884 / 180.0)
        * age.arc_length_deg / static_cast<double>(elements);
    const double radius = 0.5 * (age.inner_radius_m + age.outer_radius_m);
    const double dr = age.outer_radius_m - age.inner_radius_m;
    if (!(dt > 0.0) || !(radius > 0.0) || !(dr > 0.0)) { result.status = Status::kMeshInvalid; return result; }
    AirGapField field; field.age = &age; field.br.resize(elements); field.bt.resize(elements);
    for (size_t k = 0; k < elements; ++k) {
      int32_t node[10]; double weight[10], a[10];
      if (!BuildAirGapStencil(age, k, node, weight)) { result.status = Status::kMeshInvalid; return result; }
      for (int local = 0; local < 10; ++local) a[local] = solution.a_wb_per_m[node[local]] * weight[local];
      const double ci = age.inner_shift, co = age.outer_shift;
      field.br[k] = (-(ci * a[1]) - 2 * a[2] + 2 * a[3] + ci * (a[2] + a[3] - a[4])
          - ci * ci * ci * (a[0] - 4 * a[1] + 6 * a[2] - 4 * a[3] + a[4])
          + ci * ci * (a[0] - 5 * a[1] + 9 * a[2] - 7 * a[3] + 2 * a[4]) - 2 * a[7] + 2 * a[8]
          + co * (-a[6] + a[7] + a[8] - a[9]) - co * co * co * (a[5] - 4 * a[6] + 6 * a[7] - 4 * a[8] + a[9])
          + co * co * (a[5] - 5 * a[6] + 9 * a[7] - 7 * a[8] + 2 * a[9])) / (4 * dt * radius);
      field.bt[k] = (ci * a[1] + 2 * a[2] + 2 * a[3] - ci * ci * (a[0] - 3 * a[1] + a[2] + 3 * a[3] - 2 * a[4])
          + ci * (a[2] - a[3] - a[4]) + ci * ci * ci * (a[0] - 2 * a[1] + 2 * a[3] - a[4])
          - co * a[6] + (-2 + co) * (1 + co) * a[7] - 2 * a[8]
          + co * (a[8] + co * (a[5] - 3 * a[6] + 3 * a[8] - 2 * a[9]) + a[9]
              + co * co * (-a[5] + 2 * a[6] - 2 * a[8] + a[9]))) / (4 * dr);
      if (!std::isfinite(field.br[k]) || !std::isfinite(field.bt[k])) {
        result.status = Status::kNumericalNonfinite; return result;
      }
      const double theta = (static_cast<double>(k) + 0.5) * dt;
      const double normal = field.br[k] * field.br[k] - field.bt[k] * field.bt[k];
      const double shear = 2.0 * field.br[k] * field.bt[k];
      const double scale = model.depth_m * radius * dt / (2.0 * kMu0);
      result.force_x_n += scale * (normal * std::cos(theta) - shear * std::sin(theta));
      result.force_y_n += scale * (normal * std::sin(theta) + shear * std::cos(theta));
      result.torque_nm += model.depth_m * radius * radius * dt * field.br[k] * field.bt[k] / kMu0;
    }
    // FEMM's mo_getgapb does not return the nearest AGE element value.  It
    // reconstructs the radial field from the same discrete Fourier series
    // built by femmviewDoc when it reads an .ans file.
    const size_t harmonic_count = age.antiperiodic ? (elements + 1) / 2 : elements / 2 + 1;
    const int32_t harmonic_base = static_cast<int32_t>(std::lround(
        (age.antiperiodic ? 180.0 : 360.0) / age.arc_length_deg));
    field.harmonic_order.resize(harmonic_count);
    field.br_cos.assign(harmonic_count, 0.0);
    field.br_sin.assign(harmonic_count, 0.0);
    for (size_t harmonic = 0; harmonic < harmonic_count; ++harmonic) {
      const int32_t order = age.antiperiodic
          ? harmonic_base * static_cast<int32_t>(2 * harmonic + 1)
          : harmonic_base * static_cast<int32_t>(harmonic);
      field.harmonic_order[harmonic] = order;
      for (size_t k = 0; k < elements; ++k) {
        const double phase = (static_cast<double>(k) + 0.5) * dt * static_cast<double>(order);
        field.br_cos[harmonic] += field.br[k] * std::cos(phase);
        field.br_sin[harmonic] += field.br[k] * std::sin(phase);
      }
      const bool dc_or_nyquist = order == 0
          || (!age.antiperiodic && harmonic + 1 == harmonic_count && (elements % 2) == 0);
      const double normalization = dc_or_nyquist ? static_cast<double>(elements)
                                                 : static_cast<double>(elements) / 2.0;
      field.br_cos[harmonic] /= normalization;
      field.br_sin[harmonic] /= normalization;
    }
    fields.push_back(std::move(field));
  }
  for (const double angle : options.airgap_angles_rad) {
    if (!std::isfinite(angle)) { result.status = Status::kInvalidArgument; return result; }
    const AirGapField& field = fields.front();
    double radial_b_t = 0.0;
    for (size_t harmonic = 0; harmonic < field.harmonic_order.size(); ++harmonic) {
      const double phase = static_cast<double>(field.harmonic_order[harmonic]) * angle;
      radial_b_t += field.br_cos[harmonic] * std::cos(phase)
          + field.br_sin[harmonic] * std::sin(phase);
    }
    result.airgap_samples.push_back({ angle, radial_b_t });
  }
  if (!std::isfinite(result.force_x_n) || !std::isfinite(result.force_y_n) || !std::isfinite(result.torque_nm)) {
    result.status = Status::kNumericalNonfinite;
  }
  return result;
}

struct FemmReference {
  Model model;
  std::vector<double> expected_a;
  std::vector<double> expected_bx;
  std::vector<double> expected_by;
  double current_a = 0.0;
  double flux_linkage_wb = 0.0;
  Assembly dump;
  std::vector<double> dump_rhs;
};

bool NearMixed(double actual, double expected, double absolute_tolerance,
    double relative_tolerance)
{
  return std::isfinite(actual) && std::isfinite(expected)
      && std::abs(actual - expected)
          <= absolute_tolerance + relative_tolerance * std::abs(expected);
}

bool ReadFemmAns(const std::string& path, FemmReference* reference,
    std::string* error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::string line;
  bool found_solution = false;
  while (std::getline(input, line)) {
    if (line == "[Solution]" || line == "[Solution]\r") {
      found_solution = true;
      break;
    }
  }
  int node_count = 0;
  if (!found_solution || !(input >> node_count) || node_count <= 0) {
    *error = "invalid [Solution] node count";
    return false;
  }
  reference->model.nodes.resize(node_count);
  reference->expected_a.resize(node_count);
  for (int node = 0; node < node_count; ++node) {
    int boundary_code = 0;
    if (!(input >> reference->model.nodes[node].x_m
              >> reference->model.nodes[node].y_m
              >> reference->expected_a[node] >> boundary_code)
        || !std::isfinite(reference->expected_a[node])) {
      *error = "invalid FEMM solution node";
      return false;
    }
  }
  int element_count = 0;
  if (!(input >> element_count) || element_count <= 0) {
    *error = "invalid FEMM solution element count";
    return false;
  }
  reference->model.triangles.resize(element_count);
  std::vector<bool> boundary(node_count, false);
  for (int element = 0; element < element_count; ++element) {
    int label = 0;
    int edge[3] = { -1, -1, -1 };
    double element_current = 0.0;
    Triangle& triangle = reference->model.triangles[element];
    if (!(input >> triangle.node[0] >> triangle.node[1] >> triangle.node[2]
              >> label >> edge[0] >> edge[1] >> edge[2] >> element_current)
        || label != 0 || !std::isfinite(element_current)) {
      *error = "invalid or unsupported FEMM solution element";
      return false;
    }
    triangle.reluctivity_m_per_h = 1.0 / kMu0;
    triangle.source_j_per_a = 1.0;
    for (int side = 0; side < 3; ++side) {
      if (edge[side] >= 0) {
        const int first = triangle.node[side];
        const int second = triangle.node[(side + 1) % 3];
        if (first < 0 || first >= node_count || second < 0 || second >= node_count) {
          *error = "FEMM boundary node out of range";
          return false;
        }
        boundary[first] = true;
        boundary[second] = true;
      }
    }
  }
  for (int node = 0; node < node_count; ++node) {
    if (boundary[node]) {
      if (reference->expected_a[node] != 0.0) {
        *error = "fixture requires zero-A outer boundary";
        return false;
      }
      reference->model.dirichlet_nodes.push_back(node);
      reference->model.dirichlet_a_wb_per_m.push_back(0.0);
    }
  }
  reference->model.depth_m = 1.0;
  if (ValidateModel(reference->model) != Status::kOk) {
    *error = "FEMM mesh failed frozen-scope validation";
    return false;
  }

  reference->expected_bx.resize(element_count);
  reference->expected_by.resize(element_count);
  double area_sum = 0.0;
  for (int element = 0; element < element_count; ++element) {
    const Triangle& triangle = reference->model.triangles[element];
    const Node p0 = reference->model.nodes[triangle.node[0]];
    const Node p1 = reference->model.nodes[triangle.node[1]];
    const Node p2 = reference->model.nodes[triangle.node[2]];
    const double determinant = (p1.x_m - p0.x_m) * (p2.y_m - p0.y_m)
        - (p2.x_m - p0.x_m) * (p1.y_m - p0.y_m);
    area_sum += 0.5 * determinant;
    const double a0 = reference->expected_a[triangle.node[0]];
    const double a1 = reference->expected_a[triangle.node[1]];
    const double a2 = reference->expected_a[triangle.node[2]];
    reference->expected_bx[element] = (a0 * (p2.x_m - p1.x_m)
        + a1 * (p0.x_m - p2.x_m) + a2 * (p1.x_m - p0.x_m)) / determinant;
    reference->expected_by[element] = -(a0 * (p1.y_m - p2.y_m)
        + a1 * (p2.y_m - p0.y_m) + a2 * (p0.y_m - p1.y_m)) / determinant;
  }
  if (!NearMixed(area_sum, 1.0, 1e-12, 1e-12)) {
    *error = "fixture source normalization requires one square metre";
    return false;
  }
  return true;
}

bool ReadCircuitReference(const std::string& path, FemmReference* reference,
    std::string* error)
{
  std::ifstream input(path);
  std::string current_name;
  std::string voltage_name;
  std::string flux_name;
  double voltage = 0.0;
  if (!input || !(input >> current_name >> reference->current_a
                    >> voltage_name >> voltage
                    >> flux_name >> reference->flux_linkage_wb)
      || current_name != "current_A" || voltage_name != "voltage_drop_V"
      || flux_name != "flux_linkage_Wb" || reference->current_a != 12.0
      || voltage != 0.0 || !std::isfinite(reference->flux_linkage_wb)) {
    *error = "invalid frozen circuit reference";
    return false;
  }
  return true;
}

struct NonlinearFemmReference {
  NonlinearModel model;
  std::vector<double> expected_a;
  std::vector<double> expected_bx;
  std::vector<double> expected_by;
  double current_a = 0.0;
  double flux_linkage_wb = 0.0;
};

struct FrozenPostprocessReference {
  FrozenPostprocessOptions options;
  int32_t expected_node_count = 0;
  int32_t expected_element_count = 0;
  double expected_current_a = std::numeric_limits<double>::quiet_NaN();
  double expected_flux_linkage_wb = std::numeric_limits<double>::quiet_NaN();
  double expected_force_x_n = std::numeric_limits<double>::quiet_NaN();
  double expected_force_y_n = std::numeric_limits<double>::quiet_NaN();
  double expected_torque_nm = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> expected_radial_b_t;
};

bool ParseKeyValueDouble(const std::string& line, const std::string& key, double* value)
{
  const size_t position = line.find(key + "=");
  if (position == std::string::npos)
    return false;
  try {
    size_t parsed = 0;
    *value = std::stod(line.substr(position + key.size() + 1), &parsed);
    return parsed > 0 && std::isfinite(*value);
  } catch (...) {
    return false;
  }
}

bool ReadFrozenPostprocessReference(const std::string& stem,
    FrozenPostprocessReference* reference, std::string* error)
{
  std::ifstream input(stem + ".postprocess.txt");
  if (!input) {
    *error = "cannot open " + stem + ".postprocess.txt";
    return false;
  }
  std::string line;
  bool have_selected = false, have_nodes = false, have_elements = false;
  bool have_current = false, have_flux = false, have_fx = false, have_fy = false;
  bool have_torque = false, have_radius = false;
  while (std::getline(input, line)) {
    double value = 0.0;
    if (ParseKeyValueDouble(line, "selected_material_label", &value)) {
      const int32_t label = static_cast<int32_t>(value);
      if (value != static_cast<double>(label)) {
        *error = "non-integral selected material label";
        return false;
      }
      reference->options.selected_material_label = label;
      have_selected = true;
    } else if (ParseKeyValueDouble(line, "node_count", &value)) {
      reference->expected_node_count = static_cast<int32_t>(value);
      have_nodes = value == static_cast<double>(reference->expected_node_count)
          && reference->expected_node_count > 0;
    } else if (ParseKeyValueDouble(line, "element_count", &value)) {
      reference->expected_element_count = static_cast<int32_t>(value);
      have_elements = value == static_cast<double>(reference->expected_element_count)
          && reference->expected_element_count > 0;
    } else if (ParseKeyValueDouble(line, "circuit_current_A", &value)) {
      reference->expected_current_a = value;
      have_current = true;
    } else if (ParseKeyValueDouble(line, "flux_linkage_Wb", &value)) {
      reference->expected_flux_linkage_wb = value;
      have_flux = true;
    } else if (ParseKeyValueDouble(line, "air_material_label", &value)) {
      const int32_t label = static_cast<int32_t>(value);
      if (value != static_cast<double>(label)) {
        *error = "non-integral air material label";
        return false;
      }
      reference->options.air_material_label = label;
    } else if (ParseKeyValueDouble(line, "Fx_N", &value)) {
      reference->expected_force_x_n = value;
      have_fx = true;
    } else if (ParseKeyValueDouble(line, "Fy_N", &value)) {
      reference->expected_force_y_n = value;
      have_fy = true;
    } else if (ParseKeyValueDouble(line, "torque_Nm", &value)) {
      reference->expected_torque_nm = value;
      have_torque = true;
    } else if (ParseKeyValueDouble(line, "airgap_radius_mm", &value)) {
      reference->options.airgap_radius_m = value * 1e-3;
      have_radius = true;
    } else if (line.find("airgap_sample_deg=") != std::string::npos) {
      double angle_deg = 0.0, radial_b = 0.0;
      if (!ParseKeyValueDouble(line, "airgap_sample_deg", &angle_deg)
          || !ParseKeyValueDouble(line, "radial_B_T", &radial_b)) {
        *error = "invalid airgap sample line";
        return false;
      }
      reference->options.airgap_angles_rad.push_back(angle_deg * 3.141592653589793238462643383279502884 / 180.0);
      reference->expected_radial_b_t.push_back(radial_b);
    }
  }
  if (!have_selected || !have_nodes || !have_elements || !have_current || !have_flux
      || !have_fx || !have_fy || !have_torque || !have_radius
      || reference->options.airgap_angles_rad.empty()
      || reference->options.airgap_angles_rad.size() != reference->expected_radial_b_t.size()) {
    *error = "incomplete frozen postprocess reference";
    return false;
  }
  return true;
}

bool ReadNonlinearFemmReference(const std::string& stem, const std::string& curve_directory,
    NonlinearFemmReference* reference, std::string* error)
{
  std::ifstream input(stem + ".ans");
  std::string line;
  while (std::getline(input, line) && line != "[Solution]") {}
  int node_count = 0;
  if (!input || !(input >> node_count) || node_count <= 0) {
    *error = "invalid nonlinear [Solution] node count";
    return false;
  }
  reference->model.nodes.resize(node_count);
  reference->expected_a.resize(node_count);
  for (int node = 0; node < node_count; ++node) {
    int ignored = 0;
    if (!(input >> reference->model.nodes[node].x_m >> reference->model.nodes[node].y_m
              >> reference->expected_a[node] >> ignored)) {
      *error = "invalid nonlinear solution node";
      return false;
    }
    reference->model.nodes[node].x_m *= 1e-3;
    reference->model.nodes[node].y_m *= 1e-3;
  }
  int element_count = 0;
  if (!(input >> element_count) || element_count <= 0) {
    *error = "invalid nonlinear solution element count";
    return false;
  }
  reference->model.triangles.resize(element_count);
  std::vector<int> label(element_count, -1);
  bool label_seen[4] = { false, false, false, false };
  std::vector<bool> boundary(node_count, false);
  for (int element = 0; element < element_count; ++element) {
    int edge[3] = { -1, -1, -1 };
    double element_current = 0.0;
    NonlinearTriangle& triangle = reference->model.triangles[element];
    if (!(input >> triangle.node[0] >> triangle.node[1] >> triangle.node[2] >> label[element]
              >> edge[0] >> edge[1] >> edge[2] >> element_current)
        || label[element] < 0 || label[element] > 3 || !std::isfinite(element_current)) {
      *error = "missing or out-of-range nonlinear block label";
      return false;
    }
    triangle.material = label[element];
    label_seen[label[element]] = true;
    for (int side = 0; side < 3; ++side) {
      if (edge[side] >= 0) {
        const int first = triangle.node[side];
        const int second = triangle.node[(side + 1) % 3];
        if (first < 0 || first >= node_count || second < 0 || second >= node_count) {
          *error = "nonlinear boundary node out of range";
          return false;
        }
        boundary[first] = boundary[second] = true;
      }
    }
  }
  for (bool seen : label_seen) {
    if (!seen) {
      *error = "nonlinear fixture is missing an expected material region";
      return false;
    }
  }
  double coil_area = 0.0;
  for (int element = 0; element < element_count; ++element) {
    const NonlinearTriangle& triangle = reference->model.triangles[element];
    Node p[3] = { reference->model.nodes[triangle.node[0]], reference->model.nodes[triangle.node[1]],
      reference->model.nodes[triangle.node[2]] };
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms)) {
      *error = "invalid nonlinear element geometry";
      return false;
    }
    if (label[element] == 2)
      coil_area += terms.area_m2;
  }
  if (!(coil_area > 0.0)) {
    *error = "nonlinear fixture has no coil material";
    return false;
  }
  NonlinearBhCurve steel_curve;
  if (!ReadNonlinearBhTable(curve_directory + "/35PN230.tab", &steel_curve)) {
    *error = "cannot read 35PN230 B-H table";
    return false;
  }
  reference->model.bh_curves = { std::move(steel_curve) };
  reference->model.materials = {
    { 1.0, 0.0, 0.0, 0.0, 0.0, 0 }, // label 0: Steel35PN230
    { 1.0 / (kMu0 * 1.05), 0.0, 0.0, 900000.0, 0.0 }, // label 1: PM
    { 1.0 / kMu0, 0.0, 50.0 / coil_area, 0.0, 0.0, -1, 0 }, // label 2: driven coil-air
    { 1.0 / kMu0, 0.0, 0.0, 0.0, 0.0 }, // label 3: default air
  };
  reference->model.depth_m = 0.020;
  for (int node = 0; node < node_count; ++node) {
    if (boundary[node]) {
      if (reference->expected_a[node] != 0.0) {
        *error = "fixture requires zero-A outer boundary";
        return false;
      }
      reference->model.dirichlet_nodes.push_back(node);
      reference->model.dirichlet_a_wb_per_m.push_back(0.0);
    }
  }
  if (ValidateNonlinearModel(reference->model) != Status::kOk) {
    *error = "nonlinear frozen mesh validation failed";
    return false;
  }
  std::ifstream summary(stem + ".summary.txt");
  std::string key;
  while (summary >> key) {
    if (key == "circuit_current_A=")
      summary >> reference->current_a;
    else if (key == "flux_linkage_Wb=")
      summary >> reference->flux_linkage_wb;
    else
      summary.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  // Summary is key=value, not whitespace-separated; parse it linewise when needed.
  if (!std::isfinite(reference->current_a) || reference->current_a == 0.0
      || !std::isfinite(reference->flux_linkage_wb) || reference->flux_linkage_wb == 0.0) {
    summary.clear();
    summary.seekg(0);
    while (std::getline(summary, line)) {
      const size_t equals = line.find('=');
      if (equals == std::string::npos)
        continue;
      if (line.substr(0, equals) == "circuit_current_A")
        reference->current_a = std::stod(line.substr(equals + 1));
      if (line.substr(0, equals) == "flux_linkage_Wb")
        reference->flux_linkage_wb = std::stod(line.substr(equals + 1));
    }
  }
  if (!std::isfinite(reference->current_a) || !std::isfinite(reference->flux_linkage_wb)) {
    *error = "invalid nonlinear fixture summary";
    return false;
  }
  reference->expected_bx.resize(element_count);
  reference->expected_by.resize(element_count);
  for (int element = 0; element < element_count; ++element) {
    const NonlinearTriangle& triangle = reference->model.triangles[element];
    Node p[3] = { reference->model.nodes[triangle.node[0]], reference->model.nodes[triangle.node[1]],
      reference->model.nodes[triangle.node[2]] };
    NonlinearElementTerms terms;
    BuildElementTerms(p, &terms);
    for (int i = 0; i < 3; ++i) {
      reference->expected_bx[element] += reference->expected_a[triangle.node[i]] * terms.b_x[i];
      reference->expected_by[element] += reference->expected_a[triangle.node[i]] * terms.b_y[i];
    }
  }
  return true;
}

bool ReadDumpRhs(const std::string& path, std::vector<double>* rhs,
    std::string* error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::string line;
  bool reading = false;
  while (std::getline(input, line)) {
    if (!reading) {
      const size_t start = line.find("b = [");
      if (start == std::string::npos)
        continue;
      line = line.substr(start + 5);
      reading = true;
    }
    const bool end = line.find("];") != std::string::npos;
    line.erase(std::remove(line.begin(), line.end(), ';'), line.end());
    line.erase(std::remove(line.begin(), line.end(), ']'), line.end());
    std::istringstream value_input(line);
    double value = 0.0;
    if (value_input >> value)
      rhs->push_back(value);
    if (end)
      break;
  }
  if (!reading || rhs->empty()) {
    *error = "invalid FEMM dump RHS";
    return false;
  }
  return true;
}

bool ReadDumpMatrix(const std::string& path, size_t dimension, Assembly* dump,
    std::string* error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::map<std::pair<int, int>, std::vector<double>> entries;
  int one_based_row = 0;
  int one_based_column = 0;
  double value = 0.0;
  while (input >> one_based_row >> one_based_column >> value) {
    const int row = one_based_row - 1;
    const int column = one_based_column - 1;
    if (row < 0 || column < 0 || static_cast<size_t>(row) >= dimension
        || static_cast<size_t>(column) >= dimension || !std::isfinite(value)) {
      *error = "invalid FEMM dump matrix entry";
      return false;
    }
    entries[{ std::min(row, column), std::max(row, column) }].push_back(value);
  }
  std::vector<std::map<int, double>> rows(dimension);
  for (const auto& item : entries) {
    const std::vector<double>& duplicates = item.second;
    if (duplicates.size() != 2 || duplicates[0] != duplicates[1]) {
      *error = "FEMM dump duplicate/mirror invariant failed";
      return false;
    }
    const int row = item.first.first;
    const int column = item.first.second;
    rows[row][column] = duplicates[0];
    if (row != column)
      rows[column][row] = duplicates[0];
  }
  dump->row_offsets.assign(dimension + 1, 0);
  dump->diagonal.assign(dimension, 0.0);
  dump->free_nodes.resize(dimension);
  for (size_t row = 0; row < dimension; ++row) {
    dump->free_nodes[row] = static_cast<int32_t>(row);
    dump->row_offsets[row] = static_cast<int32_t>(dump->values.size());
    for (const auto& entry : rows[row]) {
      dump->column_indices.push_back(entry.first);
      dump->values.push_back(entry.second);
      if (entry.first == static_cast<int>(row))
        dump->diagonal[row] = entry.second;
    }
    if (!(dump->diagonal[row] > 0.0)) {
      *error = "FEMM dump missing positive diagonal";
      return false;
    }
  }
  dump->row_offsets[dimension] = static_cast<int32_t>(dump->values.size());
  return true;
}

bool LoadFemmReference(const std::string& stem, FemmReference* reference,
    std::string* error)
{
  return ReadFemmAns(stem + ".ans", reference, error)
      && ReadCircuitReference(stem + ".circuit.txt", reference, error)
      && ReadDumpRhs(stem + ".m", &reference->dump_rhs, error)
      && reference->dump_rhs.size() == reference->expected_a.size()
      && ReadDumpMatrix(stem + ".dat", reference->dump_rhs.size(),
          &reference->dump, error);
}

int FemmReferenceTest(const std::string& stem)
{
  FemmReference reference;
  std::string error;
  if (!LoadFemmReference(stem, &reference, &error)) {
    std::cerr << "FAIL: " << (error.empty() ? "reference dimension mismatch" : error) << '\n';
    return 1;
  }
  int failures = 0;
  double max_a_error = 0.0;
  double max_b_error = 0.0;
  auto expect = [&failures](bool condition, const std::string& message) {
    if (!condition) {
      std::cerr << "FAIL: " << message << '\n';
      ++failures;
    }
  };

  LinearP1FixtureSolver physical_solver;
  const Status initialize_status = physical_solver.Initialize(reference.model);
  expect(initialize_status == Status::kOk,
      std::string("FEMM physical fixture initialization: ") + StatusName(initialize_status));
  if (initialize_status == Status::kOk) {
    const SolveResult result = physical_solver.Solve(reference.current_a, 1e-13, 256);
    expect(result.info.status == Status::kOk,
        std::string("FEMM physical fixture solve: ") + StatusName(result.info.status));
    expect(result.a_wb_per_m.size() == reference.expected_a.size(), "FEMM nodal A dimension");
    expect(result.bx_t.size() == reference.expected_bx.size(), "FEMM Bx dimension");
    expect(result.by_t.size() == reference.expected_by.size(), "FEMM By dimension");
    if (result.a_wb_per_m.size() == reference.expected_a.size()) {
      for (size_t i = 0; i < result.a_wb_per_m.size(); ++i) {
        max_a_error = std::max(max_a_error,
            std::abs(result.a_wb_per_m[i] - reference.expected_a[i]));
        expect(NearMixed(result.a_wb_per_m[i], reference.expected_a[i], 1e-12, 1e-4),
            "FEMM nodal A parity");
      }
    }
    if (result.bx_t.size() == reference.expected_bx.size()
        && result.by_t.size() == reference.expected_by.size()) {
      for (size_t i = 0; i < result.bx_t.size(); ++i) {
        max_b_error = std::max(max_b_error,
            std::max(std::abs(result.bx_t[i] - reference.expected_bx[i]),
                std::abs(result.by_t[i] - reference.expected_by[i])));
        expect(NearMixed(result.bx_t[i], reference.expected_bx[i], 1e-9, 1e-4),
            "FEMM Bx parity");
        expect(NearMixed(result.by_t[i], reference.expected_by[i], 1e-9, 1e-4),
            "FEMM By parity");
      }
    }
    expect(NearMixed(result.flux_linkage_wb, reference.flux_linkage_wb, 1e-12, 1e-4),
        "FEMM circuit flux-linkage parity");
  }

  GpuCsrSolver dump_solver;
  const Status dump_status = dump_solver.Initialize(reference.dump);
  expect(dump_status == Status::kOk,
      std::string("FEMM dump initialization: ") + StatusName(dump_status));
  if (dump_status == Status::kOk) {
    std::vector<double> internal_solution;
    const SolveInfo info = dump_solver.Solve(reference.dump_rhs, 1e-13, 256,
        &internal_solution);
    expect(info.status == Status::kOk,
        std::string("FEMM dump solve: ") + StatusName(info.status));
    expect(internal_solution.size() == reference.expected_a.size(), "FEMM dump dimension");
    if (internal_solution.size() == reference.expected_a.size()) {
      for (size_t i = 0; i < internal_solution.size(); ++i) {
        const double physical_a = kFemmInternalToPhysicalA * internal_solution[i];
        expect(NearMixed(physical_a, reference.expected_a[i], 1e-12, 1e-4),
            "FEMM dump-to-ans A parity");
      }
    }
  }

  if (failures == 0) {
    std::cout << "PASS gpu_linear_p1_femm_reference\n"
              << "  nodes=" << reference.model.nodes.size()
              << " triangles=" << reference.model.triangles.size()
              << " max_A_error=" << max_a_error
              << " max_B_error=" << max_b_error << '\n'
              << "  flux_reference_Wb=" << reference.flux_linkage_wb << '\n';
  }
  return failures == 0 ? 0 : 1;
}

int NonlinearFemmReferenceTest(const std::string& stem, const std::string& curve_directory)
{
  NonlinearFemmReference reference;
  std::string error;
  if (!ReadNonlinearFemmReference(stem, curve_directory, &reference, &error)) {
    std::cerr << "FAIL nonlinear FEMM reference input: " << error << '\n';
    return 1;
  }
  NonlinearP1FixtureSolver solver;
  const Status initialize = solver.Initialize(reference.model);
  if (initialize != Status::kOk) {
    std::cerr << "FAIL nonlinear FEMM initialization: " << StatusName(initialize) << '\n';
    return 1;
  }
  NonlinearOptions options;
  options.relative_tolerance = 1e-8;
  options.max_newton_iterations = 128;
  const NonlinearSolveResult actual = solver.Solve(reference.current_a, options);
  if (actual.info.status != Status::kOk) {
    std::cerr << "FAIL nonlinear FEMM solve: " << StatusName(actual.info.status) << '\n';
    return 1;
  }
  double max_a_error = 0.0, max_b_error = 0.0;
  for (size_t i = 0; i < actual.a_wb_per_m.size(); ++i)
    max_a_error = std::max(max_a_error, std::abs(actual.a_wb_per_m[i] - reference.expected_a[i]));
  for (size_t i = 0; i < actual.bx_t.size(); ++i) {
    max_b_error = std::max(max_b_error, std::abs(actual.bx_t[i] - reference.expected_bx[i]));
    max_b_error = std::max(max_b_error, std::abs(actual.by_t[i] - reference.expected_by[i]));
  }
  const bool parity = NearMixed(actual.flux_linkage_wb, reference.flux_linkage_wb, 1e-10, 1e-6)
      && max_a_error <= 1e-9 && max_b_error <= 1e-6;
  if (!parity) {
    std::cerr << "FAIL nonlinear FEMM parity A=" << max_a_error << " B=" << max_b_error
              << " flux=" << actual.flux_linkage_wb << " expected_flux="
              << reference.flux_linkage_wb << '\n';
    return 1;
  }
  std::cout << "PASS gpu_nonlinear_p1_femm_reference\n"
            << "  nodes=" << reference.model.nodes.size() << " triangles="
            << reference.model.triangles.size() << " iterations=" << actual.info.iterations
            << " max_A_error=" << max_a_error << " max_B_error=" << max_b_error
            << " flux_Wb=" << actual.flux_linkage_wb << '\n';
  return 0;
}

int FrozenPostprocessReferenceTest(const std::string& stem, const std::string& curve_directory)
{
  NonlinearFemmReference nonlinear_reference;
  FrozenPostprocessReference postprocess_reference;
  std::string error;
  if (!ReadNonlinearFemmReference(stem, curve_directory, &nonlinear_reference, &error)
      || !ReadFrozenPostprocessReference(stem, &postprocess_reference, &error)) {
    std::cerr << "FAIL postprocess FEMM reference input: " << error << '\n';
    return 1;
  }
  NonlinearP1FixtureSolver solver;
  const Status initialize = solver.Initialize(nonlinear_reference.model);
  if (initialize != Status::kOk) {
    std::cerr << "FAIL postprocess FEMM initialization: " << StatusName(initialize) << '\n';
    return 1;
  }
  NonlinearOptions nonlinear_options;
  nonlinear_options.relative_tolerance = 1e-8;
  nonlinear_options.max_newton_iterations = 128;
  const NonlinearSolveResult solution = solver.Solve(nonlinear_reference.current_a, nonlinear_options);
  if (solution.info.status != Status::kOk) {
    std::cerr << "FAIL postprocess FEMM solve: " << StatusName(solution.info.status) << '\n';
    return 1;
  }
  const FrozenPostprocessResult actual = ComputeFrozenPostprocess(
      nonlinear_reference.model, solution, postprocess_reference.options);
  if (actual.status != Status::kOk) {
    std::cerr << "FAIL postprocess calculation: " << StatusName(actual.status) << '\n';
    return 1;
  }
  bool pass = NearMixed(actual.force_x_n, postprocess_reference.expected_force_x_n, 1e-8, 1e-5)
      && NearMixed(actual.force_y_n, postprocess_reference.expected_force_y_n, 1e-8, 1e-5)
      && NearMixed(actual.torque_nm, postprocess_reference.expected_torque_nm, 1e-10, 1e-5)
      && nonlinear_reference.model.nodes.size()
          == static_cast<size_t>(postprocess_reference.expected_node_count)
      && nonlinear_reference.model.triangles.size()
          == static_cast<size_t>(postprocess_reference.expected_element_count)
      && NearMixed(nonlinear_reference.current_a,
          postprocess_reference.expected_current_a, 1e-12, 1e-12)
      && NearMixed(solution.flux_linkage_wb,
          postprocess_reference.expected_flux_linkage_wb, 1e-10, 1e-6)
      && actual.airgap_samples.size() == postprocess_reference.expected_radial_b_t.size();
  double max_airgap_error = 0.0;
  if (actual.airgap_samples.size() == postprocess_reference.expected_radial_b_t.size()) {
    for (size_t index = 0; index < actual.airgap_samples.size(); ++index) {
      max_airgap_error = std::max(max_airgap_error,
          std::abs(actual.airgap_samples[index].radial_b_t
              - postprocess_reference.expected_radial_b_t[index]));
      pass = pass && NearMixed(actual.airgap_samples[index].radial_b_t,
          postprocess_reference.expected_radial_b_t[index], 1e-9, 1e-5);
    }
  }
  if (!pass) {
    std::cerr << "FAIL postprocess FEMM parity Fx=" << actual.force_x_n
              << " Fy=" << actual.force_y_n << " T=" << actual.torque_nm
              << " max_airgap_B=" << max_airgap_error << '\n';
    return 1;
  }
  std::cout << "PASS gpu_nonlinear_p1_postprocess_reference\n"
            << "  Fx_N=" << actual.force_x_n << " Fy_N=" << actual.force_y_n
            << " torque_Nm=" << actual.torque_nm
            << " max_airgap_B_error=" << max_airgap_error << '\n';
  return 0;
}

// Deliberately small JSON protocol for the MATLAB adapter.  It only accepts
// the fixed single-circuit fixture schema; batch/cache policy remains owned by
// MATLAB and is not inferred here.
bool JsonValueStart(const std::string& json, const std::string& key, size_t* position)
{
  const std::string quoted = "\"" + key + "\"";
  const size_t key_position = json.find(quoted);
  if (key_position == std::string::npos)
    return false;
  const size_t colon = json.find(':', key_position + quoted.size());
  if (colon == std::string::npos)
    return false;
  *position = colon + 1;
  while (*position < json.size() && std::isspace(static_cast<unsigned char>(json[*position])))
    ++*position;
  return *position < json.size();
}

bool JsonString(const std::string& json, const std::string& key, std::string* value)
{
  size_t position = 0;
  if (!JsonValueStart(json, key, &position) || json[position] != '\"')
    return false;
  const size_t end = json.find('\"', position + 1);
  if (end == std::string::npos)
    return false;
  *value = json.substr(position + 1, end - position - 1);
  return value->find('\\') == std::string::npos;
}

bool JsonDouble(const std::string& json, const std::string& key, double* value)
{
  size_t position = 0;
  if (!JsonValueStart(json, key, &position))
    return false;
  try {
    size_t parsed = 0;
    *value = std::stod(json.substr(position), &parsed);
    return parsed > 0 && std::isfinite(*value);
  } catch (...) {
    return false;
  }
}

bool JsonDoubleArray(const std::string& json, const std::string& key, std::vector<double>* values)
{
  size_t position = 0;
  if (!JsonValueStart(json, key, &position) || json[position] != '[')
    return false;
  ++position;
  values->clear();
  while (position < json.size()) {
    while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position])))
      ++position;
    if (position >= json.size())
      return false;
    if (json[position] == ']')
      return true;
    try {
      size_t parsed = 0;
      const double value = std::stod(json.substr(position), &parsed);
      if (parsed == 0 || !std::isfinite(value))
        return false;
      values->push_back(value);
      position += parsed;
    } catch (...) {
      return false;
    }
    while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position])))
      ++position;
    if (position >= json.size() || (json[position] != ',' && json[position] != ']'))
      return false;
    if (json[position] == ']')
      return true;
    ++position;
  }
  return false;
}

struct SingleSampleRequest {
  std::string stem;
  std::string curve_directory;
  std::string source_motor_fem_sha256;
  double current_a = 0.0;
  int32_t selected_material_label = -1;
  double airgap_radius_mm = 0.0;
  std::vector<double> airgap_angles_deg;
};

bool ReadSingleSampleRequest(const std::string& path, SingleSampleRequest* request, std::string* error)
{
  std::ifstream input(path);
  std::stringstream contents;
  contents << input.rdbuf();
  const std::string json = contents.str();
  std::string protocol;
  double selected = 0.0;
  if (!input || !JsonString(json, "protocol", &protocol)
      || protocol != "gpu_femm_single_sample_v1" || !JsonString(json, "stem", &request->stem)
      || !JsonString(json, "curve_directory", &request->curve_directory)
      || !JsonString(json, "source_motor_fem_sha256", &request->source_motor_fem_sha256)
      || !JsonDouble(json, "current_A", &request->current_a)
      || !JsonDouble(json, "selected_material_label", &selected)
      || !JsonDouble(json, "airgap_radius_mm", &request->airgap_radius_mm)
      || !JsonDoubleArray(json, "airgap_angles_deg", &request->airgap_angles_deg)
      || request->stem.empty() || request->curve_directory.empty()
      || request->source_motor_fem_sha256.size() != 64
      || !(request->airgap_radius_mm >= 0.0)
      || (!request->airgap_angles_deg.empty() && !(request->airgap_radius_mm > 0.0))) {
    *error = "invalid gpu_femm_single_sample_v1 request";
    return false;
  }
  request->selected_material_label = static_cast<int32_t>(selected);
  if (selected != static_cast<double>(request->selected_material_label)
      || request->selected_material_label != 1) {
    *error = "frozen adapter requires selected_material_label=1";
    return false;
  }
  if (!std::all_of(request->source_motor_fem_sha256.begin(),
          request->source_motor_fem_sha256.end(), [](unsigned char c) { return std::isxdigit(c); })) {
    *error = "source_motor_fem_sha256 must be hexadecimal";
    return false;
  }
  return true;
}

struct MotorSampleRequest {
  std::string mesh_artifact_path;
  std::string mesh_artifact_sha256;
  std::string base_motor_fem_sha256;
  std::string source_fem_sha256;
  std::vector<double> circuit_currents_a;
  int32_t selected_group_number = -1;
  int32_t air_group_number = -1;
  double airgap_radius_mm = 0.0;
  std::vector<double> airgap_angles_deg;
  double rotor_angle_deg = 0.0;
  double displacement_mm[2] = {};
};

bool StrictNumberArray(const StrictJson& value, std::vector<double>* output)
{
  if (value.type != StrictJson::Type::kArray || output == nullptr) return false;
  output->clear();
  for (const StrictJson& member : value.array) {
    if (member.type != StrictJson::Type::kNumber) return false;
    output->push_back(member.number);
  }
  return true;
}

bool ReadMotorSampleRequestValue(const StrictJson& root, MotorSampleRequest* request,
    std::string* error)
{
  if (request == nullptr || error == nullptr
      || !ExactObject(root, { "protocol", "mesh_artifact_path", "mesh_artifact_sha256",
        "base_motor_fem_sha256", "source_fem_sha256", "circuit_currents_A",
        "selected_group_number", "air_group_number", "airgap_radius_mm", "airgap_angles_deg",
        "rotor_angle_deg", "displacement_mm" }, error)) return false;
  const StrictJson* protocol = JsonMember(root, "protocol", StrictJson::Type::kString, error);
  const StrictJson* path = JsonMember(root, "mesh_artifact_path", StrictJson::Type::kString, error);
  const StrictJson* artifact_sha = JsonMember(root, "mesh_artifact_sha256", StrictJson::Type::kString, error);
  const StrictJson* base_sha = JsonMember(root, "base_motor_fem_sha256", StrictJson::Type::kString, error);
  const StrictJson* source_sha = JsonMember(root, "source_fem_sha256", StrictJson::Type::kString, error);
  const StrictJson* currents = JsonMember(root, "circuit_currents_A", StrictJson::Type::kArray, error);
  const StrictJson* selected = JsonMember(root, "selected_group_number", StrictJson::Type::kNumber, error);
  const StrictJson* air = JsonMember(root, "air_group_number", StrictJson::Type::kNumber, error);
  const StrictJson* radius = JsonMember(root, "airgap_radius_mm", StrictJson::Type::kNumber, error);
  const StrictJson* angles = JsonMember(root, "airgap_angles_deg", StrictJson::Type::kArray, error);
  const StrictJson* rotor = JsonMember(root, "rotor_angle_deg", StrictJson::Type::kNumber, error);
  const StrictJson* displacement = JsonMember(root, "displacement_mm", StrictJson::Type::kArray, error);
  int32_t selected_group = -1, air_group = -1;
  if (protocol == nullptr || path == nullptr || artifact_sha == nullptr || base_sha == nullptr || source_sha == nullptr
      || currents == nullptr || selected == nullptr || air == nullptr || radius == nullptr
      || angles == nullptr || rotor == nullptr || displacement == nullptr || protocol->string != "gpu_femm_motor_sample_v1"
      || path->string.empty() || !JsonSha256(artifact_sha->string) || !JsonSha256(base_sha->string)
      || !JsonSha256(source_sha->string)
      || !JsonInteger(*selected, &selected_group) || !JsonInteger(*air, &air_group)
      || selected_group < 0 || air_group < 0 || selected_group == air_group || !(radius->number >= 0.0)
      || displacement->array.size() != 2 || displacement->array[0].type != StrictJson::Type::kNumber
      || displacement->array[1].type != StrictJson::Type::kNumber
      || !StrictNumberArray(*currents, &request->circuit_currents_a)
      || !StrictNumberArray(*angles, &request->airgap_angles_deg)
      || request->circuit_currents_a.empty() || (!request->airgap_angles_deg.empty() && !(radius->number > 0.0))) {
    if (error->empty()) *error = "invalid gpu_femm_motor_sample_v1 request";
    return false;
  }
  request->mesh_artifact_path = path->string; request->mesh_artifact_sha256 = artifact_sha->string;
  request->base_motor_fem_sha256 = base_sha->string; request->source_fem_sha256 = source_sha->string;
  request->selected_group_number = selected_group;
  request->air_group_number = air_group; request->airgap_radius_mm = radius->number;
  request->rotor_angle_deg = rotor->number; request->displacement_mm[0] = displacement->array[0].number;
  request->displacement_mm[1] = displacement->array[1].number;
  return true;
}

bool ReadMotorSampleRequestJson(const std::string& json, MotorSampleRequest* request,
    std::string* error)
{
  StrictJson root;
  StrictJsonParser parser(json);
  return request != nullptr && error != nullptr && parser.Parse(&root, error)
      && ReadMotorSampleRequestValue(root, request, error);
}

bool RequestMatchesArtifact(const MotorSampleRequest& request, const GpuFemmMeshArtifact& artifact)
{
  const bool matching_pose = artifact.has_sliding_band
      ? request.displacement_mm[0] == 0.0 && request.displacement_mm[1] == 0.0
      : request.rotor_angle_deg == artifact.rotor_angle_deg
          && request.displacement_mm[0] == artifact.displacement_mm[0]
          && request.displacement_mm[1] == artifact.displacement_mm[1];
  bool matching_sliding_postprocess = true;
  if (artifact.has_sliding_band) {
    const auto has_group = [&artifact](int32_t group) {
      return std::any_of(artifact.model.materials.begin(), artifact.model.materials.end(),
          [group](const NonlinearMaterial& material) { return material.group_number == group; });
    };
    matching_sliding_postprocess = has_group(request.selected_group_number)
        && has_group(request.air_group_number);
    if (matching_sliding_postprocess && !request.airgap_angles_deg.empty()) {
      matching_sliding_postprocess = !artifact.model.air_gap_elements.empty();
      for (const AirGapElement& age : artifact.model.air_gap_elements) {
        const double inner_radius_mm = 1000.0 * age.inner_radius_m;
        const double outer_radius_mm = 1000.0 * age.outer_radius_m;
        const double tolerance_mm = 1e-9 * std::max(1.0, std::abs(outer_radius_mm));
        matching_sliding_postprocess = matching_sliding_postprocess
            && request.airgap_radius_mm >= inner_radius_mm - tolerance_mm
            && request.airgap_radius_mm <= outer_radius_mm + tolerance_mm;
      }
    }
  }
  return request.base_motor_fem_sha256 == artifact.base_motor_fem_sha256
      && request.source_fem_sha256 == artifact.pose_fem_sha256
      && matching_pose && matching_sliding_postprocess
      && request.circuit_currents_a.size() == static_cast<size_t>(artifact.model.circuit_count);
}

bool ApplySlidingBandRotorAngle(const MotorSampleRequest& request, const GpuFemmMeshArtifact& artifact,
    NonlinearModel* model)
{
  if (model == nullptr) return false;
  *model = artifact.model;
  if (!artifact.has_sliding_band) return true;
  const double delta_deg = request.rotor_angle_deg - artifact.rotor_angle_deg;
  if (!std::isfinite(delta_deg)) return false;
  for (AirGapElement& age : model->air_gap_elements) {
    const size_t sectors = age.quad_points.size() - 1;
    if (sectors == 0 || !(age.arc_length_deg > 0.0)) return false;
    // FEMM regenerates AGE point records for every rotor angle.  Its fractional
    // shift is only part of that operation: crossing a ring cell also rotates
    // the inner-side node/weight sequence.  Keep the stator-side records fixed
    // and reproduce that cyclic inner-side remap from the reference artifact.
    const double shifted = age.inner_shift
        + delta_deg * static_cast<double>(sectors) / age.arc_length_deg;
    const int64_t integer_cells = static_cast<int64_t>(std::floor(shifted));
    age.inner_shift = shifted - static_cast<double>(integer_cells);
    std::vector<AirGapQuadPoint> source = age.quad_points;
    const int64_t count = static_cast<int64_t>(sectors);
    for (size_t k = 0; k < sectors; ++k) {
      int64_t source_index = static_cast<int64_t>(k) - integer_cells;
      source_index %= count;
      if (source_index < 0) source_index += count;
      for (int local = 0; local < 2; ++local) {
        age.quad_points[k].node[local] = source[static_cast<size_t>(source_index)].node[local];
        age.quad_points[k].weight[local] = source[static_cast<size_t>(source_index)].weight[local];
      }
    }
    // The final point duplicates the first point on the inner ring while its
    // outer-side interpolation stays the original terminal record.
    for (int local = 0; local < 2; ++local) {
      age.quad_points[sectors].node[local] = age.quad_points[0].node[local];
      age.quad_points[sectors].weight[local] = age.quad_points[0].weight[local];
    }
  }
  return true;
}

Status SolveMeshArtifactSingleSample(const MotorSampleRequest& request,
    const GpuFemmMeshArtifact& artifact, NonlinearSolveResult* solution,
    FrozenPostprocessResult* postprocess)
{
  if (solution == nullptr || postprocess == nullptr || !RequestMatchesArtifact(request, artifact))
    return Status::kInvalidArgument;
  NonlinearModel model;
  if (!ApplySlidingBandRotorAngle(request, artifact, &model)) return Status::kInvalidArgument;
  NonlinearP1FixtureSolver solver;
  Status status = solver.Initialize(model);
  if (status != Status::kOk) return status;
  NonlinearOptions nonlinear_options;
  nonlinear_options.relative_tolerance = 1e-8;
  nonlinear_options.max_newton_iterations = 128;
  nonlinear_options.linear_relative_tolerance = kMotorLinearRelativeTolerance;
  nonlinear_options.max_linear_iterations = kMotorMaxLinearIterations;
  *solution = solver.Solve(request.circuit_currents_a, nonlinear_options);
  if (solution->info.status != Status::kOk) return solution->info.status;
  FrozenPostprocessOptions postprocess_options;
  postprocess_options.selected_group_number = request.selected_group_number;
  postprocess_options.air_group_number = request.air_group_number;
  postprocess_options.max_mask_iterations = 4096;
  postprocess_options.airgap_radius_m = request.airgap_radius_mm * 1e-3;
  for (double angle : request.airgap_angles_deg)
    postprocess_options.airgap_angles_rad.push_back(angle * 3.141592653589793238462643383279502884 / 180.0);
  *postprocess = artifact.has_sliding_band
      ? ComputeAirGapElementPostprocess(model, *solution, postprocess_options)
      : ComputeFrozenPostprocess(model, *solution, postprocess_options);
  return postprocess->status;
}

void WriteMotorSampleResponseJson(std::ostream& output, Status status, const MotorSampleRequest* request,
    const NonlinearSolveResult* solution, const FrozenPostprocessResult* postprocess)
{
  const auto field = [request](const std::string MotorSampleRequest::*member) { return request == nullptr ? "" : request->*member; };
  output << std::setprecision(17) << "{\n  \"protocol\": \"gpu_femm_motor_sample_v1\",\n"
         << "  \"mesh_artifact_sha256\": \"" << field(&MotorSampleRequest::mesh_artifact_sha256) << "\",\n"
         << "  \"base_motor_fem_sha256\": \"" << field(&MotorSampleRequest::base_motor_fem_sha256) << "\",\n"
         << "  \"source_fem_sha256\": \"" << field(&MotorSampleRequest::source_fem_sha256) << "\",\n"
         << "  \"status\": \"" << (status == Status::kOk ? "PASS" : "FAIL") << "\",\n"
         << "  \"solve_status\": \"" << StatusName(status) << "\",\n"
         << "  \"error_identifier\": \"" << (status == Status::kOk ? "" : std::string("GPU_FEMM_") + StatusName(status)) << "\",\n"
         << "  \"error_message\": \"" << (status == Status::kOk ? "" : StatusName(status)) << "\",\n";
  if (status == Status::kOk && request != nullptr && solution != nullptr && postprocess != nullptr) {
    output << "  \"Fx_N\": " << postprocess->force_x_n << ",\n  \"Fy_N\": " << postprocess->force_y_n
           << ",\n  \"torque_Nm\": " << postprocess->torque_nm << ",\n  \"actual_circuit_currents_A\": [";
    for (size_t i = 0; i < solution->circuit_currents_a.size(); ++i) output << (i ? ", " : "") << solution->circuit_currents_a[i];
    output << "],\n  \"circuit_flux_linkage_Wb\": [";
    for (size_t i = 0; i < solution->circuit_flux_linkage_wb.size(); ++i) output << (i ? ", " : "") << solution->circuit_flux_linkage_wb[i];
    output << "],\n  \"airgap_sample_angles_deg\": [";
    for (size_t i = 0; i < request->airgap_angles_deg.size(); ++i) output << (i ? ", " : "") << request->airgap_angles_deg[i];
    output << "],\n  \"airgap_radial_flux_density_T\": [";
    for (size_t i = 0; i < postprocess->airgap_samples.size(); ++i) output << (i ? ", " : "") << postprocess->airgap_samples[i].radial_b_t;
    output << "],\n  \"mesh_element_count\": " << solution->bx_t.size() << ",\n  \"convergence\": {\"iterations\": "
           << solution->info.iterations << ", \"residual_l2\": " << solution->info.residual_l2 << "}\n";
  } else {
    output << "  \"Fx_N\": null,\n  \"Fy_N\": null,\n  \"torque_Nm\": null,\n"
           << "  \"actual_circuit_currents_A\": [],\n  \"circuit_flux_linkage_Wb\": [],\n"
           << "  \"airgap_sample_angles_deg\": [],\n  \"airgap_radial_flux_density_T\": [],\n"
           << "  \"mesh_element_count\": 0,\n  \"convergence\": {\"iterations\": "
           << (solution == nullptr ? 0 : solution->info.iterations) << ", \"residual_l2\": ";
    if (solution != nullptr && std::isfinite(solution->info.residual_l2))
      output << solution->info.residual_l2;
    else
      output << "null";
    output << "}\n";
  }
  output << "}";
}

bool WriteMotorSampleResponse(const std::string& path, Status status, const MotorSampleRequest* request,
    const NonlinearSolveResult* solution, const FrozenPostprocessResult* postprocess)
{
  std::ofstream output(path, std::ios::trunc);
  if (!output) return false;
  WriteMotorSampleResponseJson(output, status, request, solution, postprocess);
  output << '\n';
  return static_cast<bool>(output);
}

int MotorSingleSampleAdapter(const std::string& request_path, const std::string& response_path)
{
  std::ifstream request_file(request_path); std::stringstream bytes; bytes << request_file.rdbuf();
  MotorSampleRequest request; std::string error;
  if (!request_file || !ReadMotorSampleRequestJson(bytes.str(), &request, &error)) {
    WriteMotorSampleResponse(response_path, Status::kInvalidArgument, nullptr, nullptr, nullptr);
    std::cerr << "FAIL motor single-sample request: " << error << '\n'; return 1;
  }
  std::ifstream artifact_file(request.mesh_artifact_path, std::ios::binary); std::stringstream artifact_bytes; artifact_bytes << artifact_file.rdbuf();
  GpuFemmMeshArtifact artifact;
  Status status = Status::kOk;
  if (!artifact_file) { status = Status::kInputIo; error = "cannot open mesh artifact"; }
  else if (Sha256Hex(artifact_bytes.str()) != request.mesh_artifact_sha256) { status = Status::kInvalidArgument; error = "mesh artifact SHA mismatch"; }
  else if (!ParseGpuFemmMeshArtifactJson(artifact_bytes.str(), &artifact, &error)) status = Status::kInvalidArgument;
  else if (!RequestMatchesArtifact(request, artifact)) { status = Status::kInvalidArgument; error = "artifact identity, pose, or circuit-count mismatch"; }
  NonlinearSolveResult solution; FrozenPostprocessResult postprocess;
  const bool solve_attempted = status == Status::kOk;
  if (solve_attempted)
    status = SolveMeshArtifactSingleSample(request, artifact, &solution, &postprocess);
  if (!WriteMotorSampleResponse(response_path, status, &request,
          solve_attempted ? &solution : nullptr,
          status == Status::kOk ? &postprocess : nullptr)) return 1;
  if (status != Status::kOk) { std::cerr << "FAIL motor single-sample: " << error << (error.empty() ? StatusName(status) : "") << '\n'; return 1; }
  return 0;
}

// Phase 4 keeps the existing single-sample request as the item identity.  The
// batch envelope is deliberately narrow: one geometry group per process, in
// caller order, with no implicit regrouping or best-effort identity repair.
struct MotorBatchItem {
  std::string task_id;
  MotorSampleRequest request;
};

struct MotorBatchRequest {
  int32_t max_items_per_chunk = 1;
  std::vector<MotorBatchItem> items;
};

bool BatchTaskIdValid(const std::string& id)
{
  return !id.empty() && id.size() <= 256 && std::all_of(id.begin(), id.end(),
      [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-' || c == '.'; });
}

bool ReadMotorBatchRequestJson(const std::string& json, MotorBatchRequest* request,
    std::string* error)
{
  StrictJson root;
  StrictJsonParser parser(json);
  if (request == nullptr || error == nullptr || !parser.Parse(&root, error)
      || !ExactObject(root, { "protocol", "max_items_per_chunk", "items" }, error)) return false;
  const StrictJson* protocol = JsonMember(root, "protocol", StrictJson::Type::kString, error);
  const StrictJson* chunk = JsonMember(root, "max_items_per_chunk", StrictJson::Type::kNumber, error);
  const StrictJson* items = JsonMember(root, "items", StrictJson::Type::kArray, error);
  int32_t max_items = 0;
  if (protocol == nullptr || chunk == nullptr || items == nullptr
      || protocol->string != "gpu_femm_motor_batch_v1" || !JsonInteger(*chunk, &max_items)
      || max_items < 1 || max_items > 4096 || items->array.empty() || items->array.size() > 4096) {
    if (error->empty()) *error = "invalid gpu_femm_motor_batch_v1 request";
    return false;
  }
  request->max_items_per_chunk = max_items;
  request->items.clear();
  std::map<std::string, bool> ids;
  for (const StrictJson& item : items->array) {
    if (!ExactObject(item, { "task_id", "request" }, error)) return false;
    const StrictJson* id = JsonMember(item, "task_id", StrictJson::Type::kString, error);
    const StrictJson* sample = JsonMember(item, "request", StrictJson::Type::kObject, error);
    MotorBatchItem parsed;
    if (id == nullptr || sample == nullptr || !BatchTaskIdValid(id->string)
        || ids.count(id->string) != 0 || !ReadMotorSampleRequestValue(*sample, &parsed.request, error)) {
      if (error->empty()) *error = "invalid or duplicate batch task_id";
      return false;
    }
    parsed.task_id = id->string;
    ids.emplace(parsed.task_id, true);
    request->items.push_back(std::move(parsed));
  }
  return true;
}

std::string NonlinearModelFingerprint(const NonlinearModel& model)
{
  std::ostringstream bytes;
  bytes << std::setprecision(17) << model.depth_m << '|' << model.circuit_count << '|';
  for (const Node& node : model.nodes) bytes << node.x_m << ',' << node.y_m << ';';
  for (const NonlinearMaterial& material : model.materials)
    bytes << material.reluctivity_zero_m_per_h << ',' << material.alpha_per_t2 << ','
          << material.source_j_per_a << ',' << material.h_c_a_per_m << ','
          << material.magnetization_deg << ',' << material.bh_curve_index << ','
          << material.circuit_index << ',' << material.group_number << ';';
  for (const NonlinearBhCurve& curve : model.bh_curves) {
    for (double value : curve.h_a_per_m) bytes << value << ',';
    bytes << ':';
    for (double value : curve.b_t) bytes << value << ',';
    bytes << ';';
  }
  for (const NonlinearTriangle& triangle : model.triangles)
    bytes << triangle.node[0] << ',' << triangle.node[1] << ',' << triangle.node[2] << ',' << triangle.material << ';';
  for (const AirGapElement& age : model.air_gap_elements) {
    bytes << 'A' << age.antiperiodic << ',' << age.center_x_m << ',' << age.center_y_m << ','
          << age.inner_radius_m << ',' << age.outer_radius_m << ',' << age.arc_length_deg << ','
          << age.inner_shift << ',' << age.outer_shift << ';';
    for (const AirGapQuadPoint& point : age.quad_points)
      for (int local = 0; local < 4; ++local) bytes << point.node[local] << ',' << point.weight[local] << ',';
    bytes << ';';
  }
  for (int32_t node : model.dirichlet_nodes) bytes << node << ',';
  bytes << ':';
  for (double value : model.dirichlet_a_wb_per_m) bytes << value << ',';
  return Sha256Hex(bytes.str());
}

// Keep artifact parsing and SHA verification tied to a resolved filesystem
// identity, rather than to the spelling used by an individual request.  The
// original request path is retained as the read path so failed opens preserve
// the established per-request INPUT_IO behavior.
std::string CanonicalArtifactPathKey(const std::string& path)
{
  std::error_code error;
  const std::filesystem::path resolved =
      std::filesystem::weakly_canonical(std::filesystem::path(path), error);
  std::string key = error ? path : resolved.generic_string();
#ifdef _WIN32
  std::transform(key.begin(), key.end(), key.begin(),
      [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
#endif
  return key;
}

struct MotorBatchArtifactLoadPlan {
  std::vector<std::string> canonical_paths;
  std::vector<std::string> representative_paths;
  std::vector<size_t> item_slots;
  std::vector<size_t> slot_use_counts;
};

MotorBatchArtifactLoadPlan MakeMotorBatchArtifactLoadPlan(const MotorBatchRequest& request)
{
  MotorBatchArtifactLoadPlan plan;
  plan.item_slots.reserve(request.items.size());
  std::map<std::string, size_t> slot_by_path;
  for (const MotorBatchItem& item : request.items) {
    const std::string canonical_path = CanonicalArtifactPathKey(item.request.mesh_artifact_path);
    const auto found = slot_by_path.find(canonical_path);
    if (found != slot_by_path.end()) {
      plan.item_slots.push_back(found->second);
      ++plan.slot_use_counts[found->second];
      continue;
    }
    const size_t slot = plan.canonical_paths.size();
    slot_by_path.emplace(canonical_path, slot);
    plan.canonical_paths.push_back(canonical_path);
    plan.representative_paths.push_back(item.request.mesh_artifact_path);
    plan.slot_use_counts.push_back(1);
    plan.item_slots.push_back(slot);
  }
  return plan;
}

struct MotorBatchArtifactLoad {
  Status status = Status::kInternalError;
  std::string artifact_sha256;
  std::string fingerprint;
  GpuFemmMeshArtifact artifact;
};

void LoadMotorBatchArtifact(const std::string& path, MotorBatchArtifactLoad* load)
{
  std::ifstream artifact_file(path, std::ios::binary);
  std::stringstream artifact_bytes; artifact_bytes << artifact_file.rdbuf();
  if (!artifact_file) {
    load->status = Status::kInputIo;
    return;
  }
  const std::string bytes = artifact_bytes.str();
  load->artifact_sha256 = Sha256Hex(bytes);
  std::string error;
  if (!ParseGpuFemmMeshArtifactJson(bytes, &load->artifact, &error)) {
    load->status = Status::kInvalidArgument;
    return;
  }
  load->fingerprint = NonlinearModelFingerprint(load->artifact.model);
  load->status = Status::kOk;
}

// Keep only protocol output after each item/chunk.  In particular, do not
// retain the nodal A or per-element B vectors for a whole batch.
struct MotorBatchResponseDto {
  Status status = Status::kInvalidArgument;
  double force_x_n = 0.0;
  double force_y_n = 0.0;
  double torque_nm = 0.0;
  std::vector<double> circuit_currents_a;
  std::vector<double> circuit_flux_linkage_wb;
  std::vector<double> airgap_angles_deg;
  std::vector<double> airgap_radial_flux_density_t;
  size_t mesh_element_count = 0;
  int iterations = 0;
  double residual_l2 = std::numeric_limits<double>::infinity();
};

struct MotorBatchTiming {
  double request_read_parse_seconds = 0.0;
  double preflight_seconds = 0.0;
  double artifact_read_validate_seconds = 0.0;
  double solver_initialize_seconds = 0.0;
  NonlinearBatchTiming nonlinear;
  double postprocess_seconds = 0.0;
  double measured_before_response_write_seconds = 0.0;
};

MotorBatchResponseDto MakeMotorBatchResponseDto(Status status, const MotorSampleRequest& request,
    const NonlinearSolveResult* solution, const FrozenPostprocessResult* postprocess)
{
  MotorBatchResponseDto dto;
  dto.status = status;
  if (solution != nullptr) {
    dto.circuit_currents_a = solution->circuit_currents_a;
    dto.iterations = solution->info.iterations;
    dto.residual_l2 = solution->info.residual_l2;
  }
  if (status == Status::kOk && solution != nullptr && postprocess != nullptr) {
    dto.force_x_n = postprocess->force_x_n; dto.force_y_n = postprocess->force_y_n;
    dto.torque_nm = postprocess->torque_nm;
    dto.circuit_flux_linkage_wb = solution->circuit_flux_linkage_wb;
    dto.airgap_angles_deg = request.airgap_angles_deg;
    for (const AirgapSample& sample : postprocess->airgap_samples)
      dto.airgap_radial_flux_density_t.push_back(sample.radial_b_t);
    dto.mesh_element_count = solution->bx_t.size();
  }
  return dto;
}

void WriteMotorBatchItemResponseJson(std::ostream& output, const MotorSampleRequest& request,
    const MotorBatchResponseDto& dto)
{
  output << std::setprecision(17) << "{\n  \"protocol\": \"gpu_femm_motor_sample_v1\",\n"
         << "  \"mesh_artifact_sha256\": \"" << request.mesh_artifact_sha256 << "\",\n"
         << "  \"base_motor_fem_sha256\": \"" << request.base_motor_fem_sha256 << "\",\n"
         << "  \"source_fem_sha256\": \"" << request.source_fem_sha256 << "\",\n"
         << "  \"status\": \"" << (dto.status == Status::kOk ? "PASS" : "FAIL") << "\",\n"
         << "  \"solve_status\": \"" << StatusName(dto.status) << "\",\n"
         << "  \"error_identifier\": \"" << (dto.status == Status::kOk ? "" : std::string("GPU_FEMM_") + StatusName(dto.status)) << "\",\n"
         << "  \"error_message\": \"" << (dto.status == Status::kOk ? "" : StatusName(dto.status)) << "\",\n";
  if (dto.status == Status::kOk) {
    output << "  \"Fx_N\": " << dto.force_x_n << ",\n  \"Fy_N\": " << dto.force_y_n
           << ",\n  \"torque_Nm\": " << dto.torque_nm << ",\n  \"actual_circuit_currents_A\": [";
    for (size_t i = 0; i < dto.circuit_currents_a.size(); ++i) output << (i ? ", " : "") << dto.circuit_currents_a[i];
    output << "],\n  \"circuit_flux_linkage_Wb\": [";
    for (size_t i = 0; i < dto.circuit_flux_linkage_wb.size(); ++i) output << (i ? ", " : "") << dto.circuit_flux_linkage_wb[i];
    output << "],\n  \"airgap_sample_angles_deg\": [";
    for (size_t i = 0; i < dto.airgap_angles_deg.size(); ++i) output << (i ? ", " : "") << dto.airgap_angles_deg[i];
    output << "],\n  \"airgap_radial_flux_density_T\": [";
    for (size_t i = 0; i < dto.airgap_radial_flux_density_t.size(); ++i) output << (i ? ", " : "") << dto.airgap_radial_flux_density_t[i];
    output << "],\n  \"mesh_element_count\": " << dto.mesh_element_count << ",\n  \"convergence\": {\"iterations\": "
           << dto.iterations << ", \"residual_l2\": " << dto.residual_l2 << "}\n";
  } else {
    output << "  \"Fx_N\": null,\n  \"Fy_N\": null,\n  \"torque_Nm\": null,\n"
           << "  \"actual_circuit_currents_A\": [],\n  \"circuit_flux_linkage_Wb\": [],\n"
           << "  \"airgap_sample_angles_deg\": [],\n  \"airgap_radial_flux_density_T\": [],\n"
           << "  \"mesh_element_count\": 0,\n  \"convergence\": {\"iterations\": " << dto.iterations
           << ", \"residual_l2\": ";
    if (std::isfinite(dto.residual_l2)) output << dto.residual_l2; else output << "null";
    output << "}\n";
  }
  output << "}";
}

bool WriteMotorBatchResponse(const std::string& path, const MotorBatchRequest* request,
    const std::vector<MotorBatchResponseDto>& responses, int32_t requested_chunk,
    int32_t effective_chunk, int32_t cache_hits, int32_t chunk_count, size_t csr_symbolic_cache_hits,
    int actual_parallel_width, int batched_pcg_launches, const MotorBatchTiming* timing)
{
  std::ofstream output(path, std::ios::trunc);
  if (!output) return false;
  bool pass = request != nullptr && request->items.size() == responses.size();
  for (const MotorBatchResponseDto& response : responses) pass = pass && response.status == Status::kOk;
  output << std::setprecision(17) << "{\n  \"protocol\": \"gpu_femm_motor_batch_v1\",\n"
         << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
         << "  \"solve_status\": \"" << (pass ? "PASS" : "PARTIAL_FAILURE") << "\",\n"
         << "  \"error_identifier\": \"" << (pass ? "" : "GPU_FEMM_BATCH_PARTIAL_FAILURE") << "\",\n"
         << "  \"error_message\": \"" << (pass ? "" : "one or more batch items failed") << "\",\n"
         << "  \"cache\": {\"mesh_cache_hits\": " << cache_hits
         << ", \"chunk_count\": " << chunk_count << ", \"requested_chunk_size\": " << requested_chunk
         << ", \"effective_chunk_size\": " << effective_chunk
         << ", \"actual_parallel_width\": " << actual_parallel_width
         << ", \"csr_symbolic_cache_hits\": " << csr_symbolic_cache_hits
         << ", \"batched_pcg_launches\": " << batched_pcg_launches << "},\n";
  if (timing != nullptr) {
    output << "  \"timing\": {\"schema_version\": \"gpu_femm_motor_batch_timing_v1\""
           << ", \"request_read_parse_seconds\": " << timing->request_read_parse_seconds
           << ", \"preflight_seconds\": " << timing->preflight_seconds
           << ", \"artifact_read_validate_seconds\": " << timing->artifact_read_validate_seconds
           << ", \"solver_initialize_seconds\": " << timing->solver_initialize_seconds
           << ", \"host_assembly_seconds\": " << timing->nonlinear.host_assembly_seconds
           << ", \"gpu_upload_seconds\": " << timing->nonlinear.gpu_upload_seconds
           << ", \"gpu_kernel_sync_seconds\": " << timing->nonlinear.gpu_kernel_sync_seconds
           << ", \"gpu_download_seconds\": " << timing->nonlinear.gpu_download_seconds
           << ", \"state_update_seconds\": " << timing->nonlinear.state_update_seconds
           << ", \"finalize_seconds\": " << timing->nonlinear.finalize_seconds
           << ", \"postprocess_seconds\": " << timing->postprocess_seconds
           << ", \"measured_before_response_write_seconds\": "
           << timing->measured_before_response_write_seconds << "},\n";
  }
  output << "  \"items\": [\n";
  if (request != nullptr) for (size_t index = 0; index < request->items.size(); ++index) {
    output << "    {\"task_id\": \"" << request->items[index].task_id << "\", \"response\": ";
    WriteMotorBatchItemResponseJson(output, request->items[index].request, responses[index]);
    output << "}" << (index + 1 == request->items.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  return static_cast<bool>(output);
}

int MotorBatchAdapter(const std::string& request_path, const std::string& response_path,
    bool include_timing = false)
{
  const ProfileClock::time_point total_start = ProfileClock::now();
  MotorBatchTiming timing;
  std::ifstream input(request_path); std::stringstream bytes; bytes << input.rdbuf();
  MotorBatchRequest request; std::string error;
  if (!input || !ReadMotorBatchRequestJson(bytes.str(), &request, &error)) {
    std::cerr << "FAIL motor batch request: " << error << '\n';
    return 1; // No trusted item identity exists to write a resumable response.
  }
  timing.request_read_parse_seconds = ProfileSecondsSince(total_start);
  const ProfileClock::time_point preflight_start = ProfileClock::now();
  const MotorBatchArtifactLoadPlan artifact_load_plan =
      MakeMotorBatchArtifactLoadPlan(request);
  std::vector<std::unique_ptr<MotorBatchArtifactLoad>> cached_artifact_loads(
      artifact_load_plan.canonical_paths.size());
  const size_t preflight_slot = artifact_load_plan.item_slots.front();
  cached_artifact_loads[preflight_slot] = std::make_unique<MotorBatchArtifactLoad>();
  LoadMotorBatchArtifact(artifact_load_plan.representative_paths[preflight_slot],
      cached_artifact_loads[preflight_slot].get());
  // Derive the batch cap from the real CSR footprint, not a sample-count
  // heuristic.  The preflight is read-only and repeats the normal strict
  // request validation before any CUDA allocation, while reusing the parsed
  // artifact during the actual batch execution.
  size_t values_per_item = 0;
  size_t nodes_per_item = 0;
  {
    const MotorSampleRequest& first = request.items.front().request;
    const MotorBatchArtifactLoad& load = *cached_artifact_loads[preflight_slot];
    if (load.status == Status::kOk
        && load.artifact_sha256 == first.mesh_artifact_sha256
        && RequestMatchesArtifact(first, load.artifact)) {
      NonlinearModel model;
      if (!ApplySlidingBandRotorAngle(first, load.artifact, &model)) model.nodes.clear();
      std::vector<double> initial_a(model.nodes.size(), 0.0);
      for (size_t boundary = 0; boundary < model.dirichlet_nodes.size(); ++boundary)
        initial_a[model.dirichlet_nodes[boundary]] = model.dirichlet_a_wb_per_m[boundary];
      Assembly estimate;
      if (AssembleNonlinearNewton(model, initial_a, first.circuit_currents_a, false, &estimate) == Status::kOk) {
        values_per_item = estimate.values.size();
        nodes_per_item = estimate.free_nodes.size();
      }
    }
  }
  timing.preflight_seconds = ProfileSecondsSince(preflight_start);
  size_t free_bytes = 0, total_bytes = 0;
  const bool memory_known = cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
  constexpr size_t kMotorBatchSafetyReserveBytes = 512ULL * 1024ULL * 1024ULL;
  constexpr int32_t kMotorBatchMaxParallelWidth = 32;
  // values plus original/inverse diagonal, RHS, and five PCG state vectors.
  // The inverse is extra workspace; keep the original diagonal for validation.
  // Reserve the largest small-batch cooperative reduction/control workspace
  // as well, even though B>4 remains on the regular one-block path.
  const size_t bytes_per_item = values_per_item == 0 || nodes_per_item == 0 ? 0
      : sizeof(double) * (values_per_item + 8 * nodes_per_item
          + kPcgCooperativeShardsPerItem + 3)
          + sizeof(int) * 5;
  const int32_t vram_limit = memory_known && bytes_per_item > 0
      ? static_cast<int32_t>(std::max<size_t>(1, std::min<size_t>(kMotorBatchMaxParallelWidth,
          free_bytes > kMotorBatchSafetyReserveBytes
              ? (free_bytes - kMotorBatchSafetyReserveBytes) / bytes_per_item : 1))) : 1;
  const bool contains_sliding_band = cached_artifact_loads[preflight_slot] != nullptr
      && cached_artifact_loads[preflight_slot]->status == Status::kOk
      && cached_artifact_loads[preflight_slot]->artifact.has_sliding_band;
  // A v2 operator changes with mechanical angle.  Until an angle-keyed CUDA
  // symbolic cache is introduced, keep each v2 item isolated rather than
  // accidentally applying one angle's AGE matrix to another angle's solve.
  const int32_t effective_chunk = contains_sliding_band ? 1 : std::max(1, std::min({ request.max_items_per_chunk, vram_limit,
      kMotorBatchMaxParallelWidth }));
  // Only paths that recur across chunks (plus the already-required preflight
  // path) remain resident.  A mixed request with thousands of distinct
  // artifacts therefore retains bounded host-memory behavior, while one
  // geometry artifact is parsed once for the whole batch regardless of CUDA
  // chunk width.
  std::vector<MotorBatchResponseDto> responses(request.items.size());
  NonlinearP1FixtureSolver cached_solver;
  std::string cached_fingerprint;
  int32_t cache_hits = 0;
  int32_t chunk_count = 0;
  int actual_parallel_width = 0;
  int batched_pcg_launches = 0;
  for (size_t first = 0; first < request.items.size(); first += static_cast<size_t>(effective_chunk)) {
    ++chunk_count;
    const size_t end = std::min(request.items.size(), first + static_cast<size_t>(effective_chunk));
    std::vector<size_t> chunk_slots;
    std::vector<const GpuFemmMeshArtifact*> chunk_artifacts;
    std::vector<NonlinearModel> chunk_models;
    std::vector<std::vector<double>> chunk_currents;
    const ProfileClock::time_point artifact_start = ProfileClock::now();
    const size_t candidate_count = end - first;
    std::vector<MotorBatchArtifactLoad*> artifact_by_slot(
        artifact_load_plan.canonical_paths.size(), nullptr);
    std::vector<size_t> load_slots;
    std::vector<size_t> load_index_by_slot(artifact_load_plan.canonical_paths.size(),
        std::numeric_limits<size_t>::max());
    for (size_t local = 0; local < candidate_count; ++local) {
      const size_t slot = artifact_load_plan.item_slots[first + local];
      if (cached_artifact_loads[slot] != nullptr) {
        artifact_by_slot[slot] = cached_artifact_loads[slot].get();
      } else if (load_index_by_slot[slot] == std::numeric_limits<size_t>::max()) {
        load_index_by_slot[slot] = load_slots.size();
        load_slots.push_back(slot);
      }
    }
    std::vector<std::unique_ptr<MotorBatchArtifactLoad>> loaded_artifacts;
    loaded_artifacts.reserve(load_slots.size());
    for (size_t ignored : load_slots) {
      (void)ignored;
      loaded_artifacts.push_back(std::make_unique<MotorBatchArtifactLoad>());
    }
    std::atomic<size_t> next_artifact { 0 };
    const size_t artifact_worker_count = std::min<size_t>(load_slots.size(),
        std::min<size_t>(6, std::max(1u, std::thread::hardware_concurrency())));
    auto load_artifact = [&]() {
      for (;;) {
        const size_t load_index = next_artifact.fetch_add(1, std::memory_order_relaxed);
        if (load_index >= load_slots.size()) return;
        const size_t slot = load_slots[load_index];
        LoadMotorBatchArtifact(artifact_load_plan.representative_paths[slot],
            loaded_artifacts[load_index].get());
      }
    };
    std::vector<std::thread> artifact_workers;
    artifact_workers.reserve(artifact_worker_count > 0 ? artifact_worker_count - 1 : 0);
    for (size_t worker = 1; worker < artifact_worker_count; ++worker) {
      try { artifact_workers.emplace_back(load_artifact); }
      catch (const std::system_error&) { break; }
    }
    load_artifact();
    for (std::thread& worker : artifact_workers) worker.join();
    for (size_t load_index = 0; load_index < load_slots.size(); ++load_index) {
      const size_t slot = load_slots[load_index];
      if (artifact_load_plan.slot_use_counts[slot] > 1) {
        cached_artifact_loads[slot] = std::move(loaded_artifacts[load_index]);
        artifact_by_slot[slot] = cached_artifact_loads[slot].get();
      } else {
        artifact_by_slot[slot] = loaded_artifacts[load_index].get();
      }
    }
    timing.artifact_read_validate_seconds += ProfileSecondsSince(artifact_start);
    for (size_t local = 0; local < candidate_count; ++local) {
      const size_t index = first + local;
      const MotorSampleRequest& item = request.items[index].request;
      MotorBatchArtifactLoad* load = artifact_by_slot[
          artifact_load_plan.item_slots[index]];
      responses[index].status = load == nullptr ? Status::kInternalError : load->status;
      if (responses[index].status != Status::kOk)
        continue;
      if (load->artifact_sha256 != item.mesh_artifact_sha256
          || !RequestMatchesArtifact(item, load->artifact)) {
        responses[index].status = Status::kInvalidArgument;
        continue;
      }
      const GpuFemmMeshArtifact& artifact = load->artifact;
      NonlinearModel model;
      if (!ApplySlidingBandRotorAngle(item, artifact, &model)) {
        responses[index].status = Status::kInvalidArgument; continue;
      }
      const std::string fingerprint = NonlinearModelFingerprint(model);
      if (cached_fingerprint.empty()) {
        const ProfileClock::time_point initialize_start = ProfileClock::now();
        responses[index].status = cached_solver.Initialize(model);
        timing.solver_initialize_seconds += ProfileSecondsSince(initialize_start);
        if (responses[index].status != Status::kOk) continue;
        cached_fingerprint = fingerprint;
      } else if (fingerprint != cached_fingerprint) {
        const ProfileClock::time_point initialize_start = ProfileClock::now();
        responses[index].status = cached_solver.Initialize(model);
        timing.solver_initialize_seconds += ProfileSecondsSince(initialize_start);
        if (responses[index].status != Status::kOk) continue;
        cached_fingerprint = fingerprint;
      } else {
        ++cache_hits;
      }
      chunk_slots.push_back(index);
      chunk_currents.push_back(item.circuit_currents_a);
      chunk_artifacts.push_back(&artifact);
      chunk_models.push_back(std::move(model));
    }
    if (!chunk_slots.empty()) {
      actual_parallel_width = std::max(actual_parallel_width, static_cast<int>(chunk_slots.size()));
      NonlinearOptions options; options.relative_tolerance = 1e-8; options.max_newton_iterations = 128;
      options.linear_relative_tolerance = kMotorLinearRelativeTolerance;
      options.max_linear_iterations = kMotorMaxLinearIterations;
      NonlinearBatchTiming chunk_timing;
      std::vector<NonlinearSolveResult> chunk_solutions = cached_solver.SolveBatch(
          chunk_currents, options, &batched_pcg_launches, &chunk_timing);
      timing.nonlinear.host_assembly_seconds += chunk_timing.host_assembly_seconds;
      timing.nonlinear.gpu_upload_seconds += chunk_timing.gpu_upload_seconds;
      timing.nonlinear.gpu_kernel_sync_seconds += chunk_timing.gpu_kernel_sync_seconds;
      timing.nonlinear.gpu_download_seconds += chunk_timing.gpu_download_seconds;
      timing.nonlinear.state_update_seconds += chunk_timing.state_update_seconds;
      timing.nonlinear.finalize_seconds += chunk_timing.finalize_seconds;
      for (size_t local = 0; local < chunk_slots.size(); ++local) {
        const size_t index = chunk_slots[local];
        const MotorSampleRequest& item = request.items[index].request;
        NonlinearSolveResult& solution = chunk_solutions[local];
        if (solution.info.status != Status::kOk) {
          responses[index] = MakeMotorBatchResponseDto(solution.info.status, item, &solution, nullptr);
          continue;
        }
      FrozenPostprocessOptions post_options;
      post_options.selected_group_number = item.selected_group_number; post_options.air_group_number = item.air_group_number;
      post_options.max_mask_iterations = 4096; post_options.airgap_radius_m = item.airgap_radius_mm * 1e-3;
      for (double angle : item.airgap_angles_deg)
        post_options.airgap_angles_rad.push_back(angle * 3.141592653589793238462643383279502884 / 180.0);
        const ProfileClock::time_point postprocess_start = ProfileClock::now();
        FrozenPostprocessResult postprocess = chunk_artifacts[local]->has_sliding_band
            ? ComputeAirGapElementPostprocess(chunk_models[local], solution, post_options)
            : ComputeFrozenPostprocess(chunk_models[local], solution, post_options);
      timing.postprocess_seconds += ProfileSecondsSince(postprocess_start);
      responses[index] = MakeMotorBatchResponseDto(postprocess.status, item, &solution,
          postprocess.status == Status::kOk ? &postprocess : nullptr);
      }
    }
  }
  timing.measured_before_response_write_seconds = ProfileSecondsSince(total_start);
  if (!WriteMotorBatchResponse(response_path, &request, responses,
          request.max_items_per_chunk, effective_chunk, cache_hits, chunk_count,
          cached_solver.csr_symbolic_reuse_count(), actual_parallel_width, batched_pcg_launches,
          include_timing ? &timing : nullptr)) return 1;
  return std::all_of(responses.begin(), responses.end(),
      [](const MotorBatchResponseDto& response) { return response.status == Status::kOk; }) ? 0 : 1;
}

bool WriteSingleSampleResponse(const std::string& path, Status status,
    const SingleSampleRequest* request, const NonlinearSolveResult* solution,
    const FrozenPostprocessResult* postprocess)
{
  std::ofstream output(path, std::ios::trunc);
  if (!output)
    return false;
  output << std::setprecision(17) << "{\n"
         << "  \"protocol\": \"gpu_femm_single_sample_v1\",\n"
         << "  \"source_motor_fem_sha256\": \""
         << (request == nullptr ? "" : request->source_motor_fem_sha256) << "\",\n"
         << "  \"status\": \"" << (status == Status::kOk ? "PASS" : "FAIL") << "\",\n"
         << "  \"solve_status\": \"" << (status == Status::kOk ? "PASS" : "FAIL") << "\",\n"
         << "  \"error_identifier\": \""
         << (status == Status::kOk ? "" : std::string("GPU_FEMM_") + StatusName(status)) << "\",\n"
         << "  \"error_message\": \"" << (status == Status::kOk ? "" : StatusName(status)) << "\",\n";
  if (status == Status::kOk && request != nullptr && solution != nullptr && postprocess != nullptr) {
    output << "  \"Fx_N\": " << postprocess->force_x_n << ",\n"
           << "  \"Fy_N\": " << postprocess->force_y_n << ",\n"
           << "  \"torque_Nm\": " << postprocess->torque_nm << ",\n"
           << "  \"actual_circuit_currents_A\": [" << request->current_a << "],\n"
           << "  \"circuit_flux_linkage_Wb\": [" << solution->flux_linkage_wb << "],\n"
           << "  \"airgap_sample_angles_deg\": [";
    for (size_t index = 0; index < request->airgap_angles_deg.size(); ++index)
      output << (index == 0 ? "" : ", ") << request->airgap_angles_deg[index];
    output << "],\n  \"airgap_radial_flux_density_T\": [";
    for (size_t index = 0; index < postprocess->airgap_samples.size(); ++index)
      output << (index == 0 ? "" : ", ") << postprocess->airgap_samples[index].radial_b_t;
    output << "],\n  \"mesh_element_count\": " << solution->bx_t.size() << ",\n"
           << "  \"convergence\": {\"iterations\": " << solution->info.iterations
           << ", \"residual_l2\": " << solution->info.residual_l2 << "}\n";
  } else {
    output << "  \"Fx_N\": null,\n  \"Fy_N\": null,\n  \"torque_Nm\": null,\n"
           << "  \"actual_circuit_currents_A\": [],\n"
           << "  \"circuit_flux_linkage_Wb\": [],\n"
           << "  \"airgap_sample_angles_deg\": [],\n"
           << "  \"airgap_radial_flux_density_T\": [],\n"
           << "  \"mesh_element_count\": 0,\n"
           << "  \"convergence\": {\"iterations\": 0, \"residual_l2\": null}\n";
  }
  output << "}\n";
  return static_cast<bool>(output);
}

int SingleSampleAdapter(const std::string& request_path, const std::string& response_path)
{
  std::ifstream protocol_file(request_path);
  std::stringstream protocol_bytes; protocol_bytes << protocol_file.rdbuf();
  StrictJson protocol_root; std::string protocol_error;
  if (protocol_file && StrictJsonParser(protocol_bytes.str()).Parse(&protocol_root, &protocol_error)
      && protocol_root.type == StrictJson::Type::kObject) {
    const auto protocol = protocol_root.object.find("protocol");
    if (protocol != protocol_root.object.end() && protocol->second.type == StrictJson::Type::kString
        && protocol->second.string == "gpu_femm_motor_sample_v1")
      return MotorSingleSampleAdapter(request_path, response_path);
  }
  SingleSampleRequest request;
  std::string error;
  if (!ReadSingleSampleRequest(request_path, &request, &error)) {
    WriteSingleSampleResponse(response_path, Status::kInvalidArgument, nullptr, nullptr, nullptr);
    std::cerr << "FAIL single-sample request: " << error << '\n';
    return 1;
  }
  NonlinearFemmReference reference;
  if (!ReadNonlinearFemmReference(request.stem, request.curve_directory, &reference, &error)) {
    WriteSingleSampleResponse(response_path, Status::kInputIo, &request, nullptr, nullptr);
    std::cerr << "FAIL single-sample reference: " << error << '\n';
    return 1;
  }
  NonlinearP1FixtureSolver solver;
  Status status = solver.Initialize(reference.model);
  NonlinearSolveResult solution;
  if (status == Status::kOk) {
    NonlinearOptions options;
    options.relative_tolerance = 1e-8;
    options.max_newton_iterations = 128;
    solution = solver.Solve(request.current_a, options);
    status = solution.info.status;
  }
  FrozenPostprocessResult postprocess;
  if (status == Status::kOk) {
    FrozenPostprocessOptions options;
    options.selected_material_label = request.selected_material_label;
    options.airgap_radius_m = request.airgap_radius_mm * 1e-3;
    for (const double angle_deg : request.airgap_angles_deg)
      options.airgap_angles_rad.push_back(angle_deg * 3.141592653589793238462643383279502884 / 180.0);
    postprocess = ComputeFrozenPostprocess(reference.model, solution, options);
    status = postprocess.status;
  }
  if (!WriteSingleSampleResponse(response_path, status, &request,
          status == Status::kOk ? &solution : nullptr,
          status == Status::kOk ? &postprocess : nullptr)) {
    std::cerr << "FAIL single-sample response: cannot write " << response_path << '\n';
    return 1;
  }
  if (status != Status::kOk) {
    std::cerr << "FAIL single-sample: " << StatusName(status) << '\n';
    return 1;
  }
  return 0;
}


Model UnitSquareFixture()
{
  Model model;
  model.nodes = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }, { 0.0, 1.0 },
    { 0.5, 0.5 } };
  model.triangles = {
    { { 0, 1, 4 }, 1.0, 1.0 }, { { 1, 2, 4 }, 1.0, 1.0 },
    { { 2, 3, 4 }, 1.0, 1.0 }, { { 3, 0, 4 }, 1.0, 1.0 }
  };
  model.dirichlet_nodes = { 0, 1, 2, 3 };
  model.dirichlet_a_wb_per_m = { 0.0, 0.0, 0.0, 0.0 };
  model.depth_m = 1.0;
  return model;
}

Model LargeStructuredLinearFixture(int cells_per_side)
{
  Model model;
  const int nodes_per_side = cells_per_side + 1;
  model.nodes.reserve(static_cast<size_t>(nodes_per_side) * nodes_per_side);
  model.triangles.reserve(static_cast<size_t>(cells_per_side) * cells_per_side * 2);
  for (int y = 0; y < nodes_per_side; ++y) {
    for (int x = 0; x < nodes_per_side; ++x) {
      model.nodes.push_back({ static_cast<double>(x), static_cast<double>(y) });
      if (x == 0 || y == 0 || x == cells_per_side || y == cells_per_side) {
        model.dirichlet_nodes.push_back(static_cast<int32_t>(y * nodes_per_side + x));
        model.dirichlet_a_wb_per_m.push_back(0.0);
      }
    }
  }
  for (int y = 0; y < cells_per_side; ++y) {
    for (int x = 0; x < cells_per_side; ++x) {
      const int32_t lower_left = static_cast<int32_t>(y * nodes_per_side + x);
      const int32_t lower_right = lower_left + 1;
      const int32_t upper_left = lower_left + nodes_per_side;
      const int32_t upper_right = upper_left + 1;
      model.triangles.push_back({ { lower_left, lower_right, upper_right }, 1.0, 0.0 });
      model.triangles.push_back({ { lower_left, upper_right, upper_left }, 1.0, 0.0 });
    }
  }
  model.depth_m = 1.0;
  return model;
}

NonlinearModel LargeStructuredMaskFixture(int cells_per_side)
{
  NonlinearModel model;
  const int nodes_per_side = cells_per_side + 1;
  const int center = cells_per_side / 2;
  const int32_t selected_lower_left = static_cast<int32_t>(center * nodes_per_side + center);
  const int32_t selected_lower_right = selected_lower_left + 1;
  const int32_t selected_upper_right = selected_lower_right + nodes_per_side;
  model.nodes.reserve(static_cast<size_t>(nodes_per_side) * nodes_per_side);
  model.triangles.reserve(static_cast<size_t>(cells_per_side) * cells_per_side * 2);
  // Both regions have the same linear material law; their distinct labels
  // exercise WST selection without changing the analytic stencil.
  model.materials = { { 1.0, 0.0, 0.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0, 0.0, 0.0 } };
  for (int y = 0; y < nodes_per_side; ++y) {
    for (int x = 0; x < nodes_per_side; ++x) {
      const int32_t node = static_cast<int32_t>(y * nodes_per_side + x);
      model.nodes.push_back({ static_cast<double>(x), static_cast<double>(y) });
      if (node != selected_lower_left && node != selected_lower_right
          && node != selected_upper_right) {
        model.dirichlet_nodes.push_back(node);
        model.dirichlet_a_wb_per_m.push_back(0.0);
      }
    }
  }
  for (int y = 0; y < cells_per_side; ++y) {
    for (int x = 0; x < cells_per_side; ++x) {
      const int32_t lower_left = static_cast<int32_t>(y * nodes_per_side + x);
      const int32_t lower_right = lower_left + 1;
      const int32_t upper_left = lower_left + nodes_per_side;
      const int32_t upper_right = upper_left + 1;
      const int32_t material = lower_left == selected_lower_left ? 1 : 0;
      model.triangles.push_back({ { lower_left, lower_right, upper_right }, material });
      model.triangles.push_back({ { lower_left, upper_right, upper_left }, 0 });
    }
  }
  model.depth_m = 1.0;
  return model;
}

NonlinearModel NonlinearThreeRegionFixture()
{
  NonlinearModel model;
  model.nodes = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }, { 0.0, 1.0 },
    { 0.5, 0.5 } };
  // PM, driven coil, and nonlinear steel all occur in the same small model.
  model.materials = {
    { 1.0, 0.0, 0.0, 10.0, 0.0 },
    { 1.0, 0.0, 12.0, 0.0, 0.0, -1, 0 },
    { 0.5, 1.0, 0.0, 0.0, 0.0, 0 },
  };
  model.bh_curves = { { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 4.0 }, {} } };
  BuildNonlinearBhSpline(&model.bh_curves[0]);
  model.triangles = { { { 0, 1, 4 }, 0 }, { { 1, 2, 4 }, 1 },
    { { 2, 3, 4 }, 2 }, { { 3, 0, 4 }, 2 } };
  model.dirichlet_nodes = { 0, 1, 2, 3 };
  model.dirichlet_a_wb_per_m = { 0.0, 0.0, 0.0, 0.0 };
  model.depth_m = 1.0;
  return model;
}

NonlinearModel NonlinearTwoCircuitFixture()
{
  NonlinearModel model;
  model.nodes = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }, { 0.0, 1.0 },
    { 0.5, 0.5 } };
  model.materials = {
    { 1.0, 0.0, 0.0, 0.0, 0.0 },
    { 1.0, 0.0, 12.0, 0.0, 0.0, -1, 0 },
    { 1.0, 0.0, 6.0, 0.0, 0.0, -1, 1 },
  };
  model.triangles = { { { 0, 1, 4 }, 1 }, { { 1, 2, 4 }, 0 },
    { { 2, 3, 4 }, 2 }, { { 3, 0, 4 }, 0 } };
  model.dirichlet_nodes = { 0, 1, 2, 3 };
  model.dirichlet_a_wb_per_m = { 0.0, 0.0, 0.0, 0.0 };
  model.circuit_count = 2;
  model.depth_m = 1.0;
  return model;
}

NonlinearModel PostprocessRingFixture()
{
  NonlinearModel model;
  model.nodes = { { -1.0, -1.0 }, { 1.0, -1.0 }, { 1.0, 1.0 }, { -1.0, 1.0 },
    { -0.5, -0.5 }, { 0.5, -0.5 }, { 0.5, 0.5 }, { -0.5, 0.5 } };
  model.materials = {
    { 1.0 / kMu0, 0.0, 0.0, 0.0, 0.0 }, { 1.0 / kMu0, 0.0, 0.0, 0.0, 0.0 },
    { 1.0 / kMu0, 0.0, 0.0, 0.0, 0.0 }, { 1.0 / kMu0, 0.0, 0.0, 0.0, 0.0 }
  };
  model.triangles = { { { 0, 1, 5 }, 3 }, { { 0, 5, 4 }, 3 },
    { { 1, 2, 6 }, 3 }, { { 1, 6, 5 }, 3 }, { { 2, 3, 7 }, 3 },
    { { 2, 7, 6 }, 3 }, { { 3, 0, 4 }, 3 }, { { 3, 4, 7 }, 3 },
    { { 4, 5, 6 }, 1 }, { { 4, 6, 7 }, 1 } };
  model.dirichlet_nodes = { 0, 1, 2, 3 };
  model.dirichlet_a_wb_per_m = { -1.0, 1.0, 1.0, -1.0 };
  model.depth_m = 1.0;
  return model;
}

bool Near(double actual, double expected, double tolerance = 1e-12)
{
  return std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
}

bool SameBits(const SolveResult& left, const SolveResult& right)
{
  if (left.info.status != right.info.status || left.info.iterations != right.info.iterations || std::memcmp(&left.info.residual_l2, &right.info.residual_l2, sizeof(double)) != 0 || std::memcmp(&left.flux_linkage_wb, &right.flux_linkage_wb, sizeof(double)) != 0 || left.a_wb_per_m.size() != right.a_wb_per_m.size() || left.bx_t.size() != right.bx_t.size() || left.by_t.size() != right.by_t.size()) {
    return false;
  }
  return std::memcmp(left.a_wb_per_m.data(), right.a_wb_per_m.data(),
             left.a_wb_per_m.size() * sizeof(double))
      == 0
      && std::memcmp(left.bx_t.data(), right.bx_t.data(), left.bx_t.size() * sizeof(double)) == 0 && std::memcmp(left.by_t.data(), right.by_t.data(), left.by_t.size() * sizeof(double)) == 0;
}

int SelfTest()
{
  int failures = 0;
  auto expect = [&failures](bool condition, const std::string& message) {
    if (!condition) {
      std::cerr << "FAIL: " << message << '\n';
      ++failures;
    }
  };

  const std::string mesh_artifact = R"json({
    "schema_version":"gpu_femm_mesh_v1",
    "base_motor_fem_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "source_fem_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "canonical_identity_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "resolved":{
      "source_fem_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "base_motor_fem_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "model":{"depth_mm":20,"problem_type":"planar","frequency_hz":0},
      "pose":{"rotor_angle_deg":3,"displacement_mm":[0.1,0]},
      "nodes_mm":[[0,0],[1,0],[1,1],[0,1],[0.5,0.5]],
      "triangles":{"node_indices":[[0,1,4],[1,2,4],[2,3,4],[3,0,4]],"region_ids":[7,8,9,9]},
      "regions":[
        {"id":7,"group_number":20,"material_id":4,"circuit_index":-1,"turns":0,"pm_magnetization_deg":30},
        {"id":8,"group_number":50,"material_id":9,"circuit_index":0,"turns":50,"pm_magnetization_deg":0},
        {"id":9,"group_number":50,"material_id":9,"circuit_index":1,"turns":-50,"pm_magnetization_deg":0}],
      "materials":[
        {"id":4,"name":"PM","mu_x":1.05,"mu_y":1.05,"H_c_A_per_m":900000,"B_T":[],"H_A_per_m":[],"lam_type":0,"lam_fill":1},
        {"id":9,"name":"Air","mu_x":1,"mu_y":1,"H_c_A_per_m":0,"B_T":[],"H_A_per_m":[],"lam_type":0,"lam_fill":1}],
      "circuits":[{"index":0,"name":"coil_00","type":"series","current_A":2},{"index":1,"name":"coil_01","type":"series","current_A":-2}],
      "outer_dirichlet":{"node_indices":[0,1,2,3],"A_Wb_per_m":[0,0,0,0]}
    }
  })json";
  GpuFemmMeshArtifact parsed_artifact;
  std::string parser_error;
  expect(ParseGpuFemmMeshArtifactJson(mesh_artifact, &parsed_artifact, &parser_error),
      std::string("gpu_femm_mesh_v1 parser: ") + parser_error);
  expect(parsed_artifact.model.circuit_count == 2
          && parsed_artifact.circuit_currents_a.size() == 2
          && parsed_artifact.triangle_group_numbers.size() == 4
          && parsed_artifact.base_motor_fem_sha256.size() == 64
          && parsed_artifact.pose_fem_sha256.size() == 64,
      "gpu_femm_mesh_v1 maps multi-circuit labels and identities");
  double native_age_matrix[10][10] = {};
  expect(BuildNativeFemmAirGapMatrix(0.75, 1.0 / 0.75, 0.25, 0.75, native_age_matrix)
          && native_age_matrix[0][0] > 0.0 && native_age_matrix[7][7] > 0.0
          && Near(native_age_matrix[1][8], native_age_matrix[8][1], 1e-14),
      "native FEMM AGE matrix is finite, positive-diagonal, and symmetric");
  expect(Near(kNativeFemmAgeToSiReluctivity * kMu0, 1.0, 1e-15),
      "native FEMM AGE stiffness is converted to SI free-space reluctivity");
  std::string sliding_mesh_artifact = mesh_artifact;
  sliding_mesh_artifact.replace(sliding_mesh_artifact.find("gpu_femm_mesh_v1"), 16, "gpu_femm_mesh_v2");
  const std::string v1_pose = "\"pose\":{\"rotor_angle_deg\":3,\"displacement_mm\":[0.1,0]}";
  sliding_mesh_artifact.replace(sliding_mesh_artifact.find(v1_pose), v1_pose.size(),
      "\"pose\":{\"rotor_angle_deg\":0,\"displacement_mm\":[0,0]}");
  const std::string v1_boundary = "\"outer_dirichlet\":{\"node_indices\":[0,1,2,3],\"A_Wb_per_m\":[0,0,0,0]}";
  const std::string v2_boundary = v1_boundary + ",\"air_gap_elements\":[{\"name\":\"gap\",\"periodicity\":\"periodic\",\"center_mm\":[0,0],\"ri_mm\":0.4,\"ro_mm\":0.6,\"arc_length_deg\":360,\"sector_count\":2,\"inner_shift\":0,\"outer_shift\":0,\"quad_points\":[{\"n0\":0,\"w0\":1,\"n1\":1,\"w1\":1,\"n2\":2,\"w2\":1,\"n3\":3,\"w3\":1},{\"n0\":1,\"w0\":1,\"n1\":2,\"w1\":1,\"n2\":3,\"w2\":1,\"n3\":0,\"w3\":1},{\"n0\":2,\"w0\":1,\"n1\":3,\"w1\":1,\"n2\":0,\"w2\":1,\"n3\":1,\"w3\":1}]}]";
  sliding_mesh_artifact.replace(sliding_mesh_artifact.find(v1_boundary), v1_boundary.size(), v2_boundary);
  GpuFemmMeshArtifact sliding_artifact;
  expect(ParseGpuFemmMeshArtifactJson(sliding_mesh_artifact, &sliding_artifact, &parser_error)
          && sliding_artifact.has_sliding_band && sliding_artifact.model.air_gap_elements.size() == 1,
      std::string("gpu_femm_mesh_v2 sliding-band parser: ") + parser_error);
  std::string scalar_sliding_mesh_artifact = sliding_mesh_artifact;
  const std::string age_array_open = "\"air_gap_elements\":[{";
  const size_t age_array_open_at = scalar_sliding_mesh_artifact.find(age_array_open);
  if (age_array_open_at != std::string::npos)
    scalar_sliding_mesh_artifact.replace(age_array_open_at, age_array_open.size(), "\"air_gap_elements\":{");
  const size_t age_array_close_at = scalar_sliding_mesh_artifact.rfind("}]\n    }");
  if (age_array_close_at != std::string::npos) scalar_sliding_mesh_artifact.erase(age_array_close_at + 1, 1);
  GpuFemmMeshArtifact scalar_sliding_artifact;
  expect(ParseGpuFemmMeshArtifactJson(scalar_sliding_mesh_artifact, &scalar_sliding_artifact, &parser_error)
          && scalar_sliding_artifact.has_sliding_band && scalar_sliding_artifact.model.air_gap_elements.size() == 1,
      "gpu_femm_mesh_v2 parser accepts MATLAB scalar AGE object spelling");
  NonlinearSolveResult zero_age_solution;
  zero_age_solution.a_wb_per_m.assign(sliding_artifact.model.nodes.size(), 0.0);
  FrozenPostprocessOptions zero_age_options;
  zero_age_options.airgap_angles_rad = { 0.0 };
  const FrozenPostprocessResult zero_age_postprocess = ComputeAirGapElementPostprocess(
      sliding_artifact.model, zero_age_solution, zero_age_options);
  expect(zero_age_postprocess.status == Status::kOk && zero_age_postprocess.force_x_n == 0.0
          && zero_age_postprocess.force_y_n == 0.0 && zero_age_postprocess.torque_nm == 0.0
          && zero_age_postprocess.airgap_samples.size() == 1
          && zero_age_postprocess.airgap_samples[0].radial_b_t == 0.0,
      "sliding AGE postprocess initializes finite zero-field outputs");
  expect(Sha256Hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
      "dependency-free SHA-256 matches known vector");
  const std::string motor_request_json = R"json({
    "protocol":"gpu_femm_motor_sample_v1",
    "mesh_artifact_path":"fixture.json",
    "mesh_artifact_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    "base_motor_fem_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "source_fem_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "circuit_currents_A":[2,-2],"selected_group_number":20,"air_group_number":50,
    "airgap_radius_mm":0,"airgap_angles_deg":[],"rotor_angle_deg":3,"displacement_mm":[0.1,0]
  })json";
  MotorSampleRequest parsed_request;
  expect(ReadMotorSampleRequestJson(motor_request_json, &parsed_request, &parser_error)
          && RequestMatchesArtifact(parsed_request, parsed_artifact),
      "motor request binds artifact identities and pose");
  MotorSampleRequest sliding_request = parsed_request;
  sliding_request.rotor_angle_deg = 3.0; sliding_request.displacement_mm[0] = 0.0;
  expect(RequestMatchesArtifact(sliding_request, sliding_artifact),
      "v2 request permits a centered non-reference rotor angle");
  MotorSampleRequest invalid_sliding_postprocess = sliding_request;
  invalid_sliding_postprocess.airgap_angles_deg = { 0.0 };
  invalid_sliding_postprocess.airgap_radius_mm = 0.7;
  expect(!RequestMatchesArtifact(invalid_sliding_postprocess, sliding_artifact),
      "v2 request rejects sampling outside the native AGE annulus");
  invalid_sliding_postprocess = sliding_request;
  invalid_sliding_postprocess.selected_group_number = 999;
  expect(!RequestMatchesArtifact(invalid_sliding_postprocess, sliding_artifact),
      "v2 request rejects postprocess groups absent from the artifact");
  NonlinearModel shifted_sliding_model;
  expect(ApplySlidingBandRotorAngle(sliding_request, sliding_artifact, &shifted_sliding_model)
          && Near(shifted_sliding_model.air_gap_elements[0].inner_shift, 3.0 / 180.0, 1e-14)
          && shifted_sliding_model.air_gap_elements[0].quad_points[0].node[0]
              == sliding_artifact.model.air_gap_elements[0].quad_points[0].node[0],
      "v2 request applies FEMM's positive AGE shift and inner-ring remap for rotor angle");
  MotorSampleRequest cell_shift_request = sliding_request;
  cell_shift_request.rotor_angle_deg = 180.0;
  NonlinearModel cell_shift_model;
  expect(ApplySlidingBandRotorAngle(cell_shift_request, sliding_artifact, &cell_shift_model)
          && Near(cell_shift_model.air_gap_elements[0].inner_shift, 0.0, 1e-14)
          && cell_shift_model.air_gap_elements[0].quad_points[0].node[0]
              == sliding_artifact.model.air_gap_elements[0].quad_points[1].node[0],
      "v2 request cyclically remaps inner AGE records after a whole-cell shift");
  parsed_request.circuit_currents_a = { 3.0, -3.0 };
  expect(RequestMatchesArtifact(parsed_request, parsed_artifact),
      "motor request accepts a new circuit vector for one immutable geometry artifact");
  parsed_request.circuit_currents_a = { 3.0 };
  expect(!RequestMatchesArtifact(parsed_request, parsed_artifact),
      "motor request rejects a circuit-count mismatch for its geometry artifact");
  parsed_request.circuit_currents_a = { 3.0, -3.0 };
  parsed_request.base_motor_fem_sha256[0] = '0';
  expect(!RequestMatchesArtifact(parsed_request, parsed_artifact),
      "motor request rejects artifact identity hash mismatch");
  const std::string batch_request_json = std::string("{\"protocol\":\"gpu_femm_motor_batch_v1\",")
      + "\"max_items_per_chunk\":2,\"items\":[{\"task_id\":\"case_0001.op\",\"request\":"
      + motor_request_json + "}]}";
  MotorBatchRequest parsed_batch;
  expect(ReadMotorBatchRequestJson(batch_request_json, &parsed_batch, &parser_error)
          && parsed_batch.items.size() == 1 && parsed_batch.items[0].task_id == "case_0001.op"
          && RequestMatchesArtifact(parsed_batch.items[0].request, parsed_artifact),
      "motor batch preserves ordered single-sample identity");
  const std::string duplicate_batch_json = std::string("{\"protocol\":\"gpu_femm_motor_batch_v1\",")
      + "\"max_items_per_chunk\":1,\"items\":[{\"task_id\":\"duplicate\",\"request\":"
      + motor_request_json + "},{\"task_id\":\"duplicate\",\"request\":" + motor_request_json + "}]}";
  expect(!ReadMotorBatchRequestJson(duplicate_batch_json, &parsed_batch, &parser_error),
      "motor batch rejects duplicate task identities");
  // A tiny real adapter run guards the batch envelope contract, including the
  // top-level PASS spelling consumed by MATLAB and DTO response ordering.
  const std::string batch_artifact_path = "gpu_femm_batch_selftest_artifact.json";
  const std::string batch_single_request_path = "gpu_femm_batch_selftest_single_request.json";
  const std::string batch_single_response_path = "gpu_femm_batch_selftest_single_response.json";
  const std::string batch_request_path = "gpu_femm_batch_selftest_request.json";
  const std::string batch_response_path = "gpu_femm_batch_selftest_response.json";
  const std::string batch_default_response_path = "gpu_femm_batch_selftest_default_response.json";
  const std::string mixed_request_path = "gpu_femm_batch_selftest_mixed_request.json";
  const std::string mixed_response_path = "gpu_femm_batch_selftest_mixed_response.json";
  std::string batch_mesh_artifact = mesh_artifact;
  const std::string selftest_boundary = "\"node_indices\":[0,1,2,3],\"A_Wb_per_m\":[0,0,0,0]";
  const size_t selftest_boundary_at = batch_mesh_artifact.find(selftest_boundary);
  if (selftest_boundary_at != std::string::npos)
    batch_mesh_artifact.replace(selftest_boundary_at, selftest_boundary.size(),
        "\"node_indices\":[3],\"A_Wb_per_m\":[0]");
  const std::string artifact_sha = Sha256Hex(batch_mesh_artifact);
  std::string selftest_motor_request = motor_request_json;
  const size_t artifact_sha_at = selftest_motor_request.find("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
  if (artifact_sha_at != std::string::npos) selftest_motor_request.replace(artifact_sha_at, 64, artifact_sha);
  const size_t artifact_path_at = selftest_motor_request.find("fixture.json");
  if (artifact_path_at != std::string::npos) selftest_motor_request.replace(artifact_path_at, 12, batch_artifact_path);
  {
    std::ofstream artifact_output(batch_artifact_path, std::ios::binary | std::ios::trunc);
    artifact_output << batch_mesh_artifact;
    std::ofstream single_request_output(batch_single_request_path, std::ios::trunc);
    single_request_output << selftest_motor_request;
    std::ofstream batch_request_output(batch_request_path, std::ios::trunc);
    batch_request_output << "{\"protocol\":\"gpu_femm_motor_batch_v1\",\"max_items_per_chunk\":4,\"items\":["
                         << "{\"task_id\":\"first\",\"request\":" << selftest_motor_request << "},"
                         << "{\"task_id\":\"second\",\"request\":" << selftest_motor_request << "},"
                         << "{\"task_id\":\"third\",\"request\":" << selftest_motor_request << "},"
                         << "{\"task_id\":\"fourth\",\"request\":" << selftest_motor_request << "}]}";
    std::string bad_hash_request = selftest_motor_request;
    const size_t hash_at = bad_hash_request.find(artifact_sha);
    if (hash_at != std::string::npos) bad_hash_request.replace(hash_at, artifact_sha.size(), std::string(64, '0'));
    const size_t bad_path_at = bad_hash_request.find(batch_artifact_path);
    if (bad_path_at != std::string::npos)
      bad_hash_request.replace(bad_path_at, batch_artifact_path.size(), "./" + batch_artifact_path);
    std::ofstream mixed_request_output(mixed_request_path, std::ios::trunc);
    mixed_request_output << "{\"protocol\":\"gpu_femm_motor_batch_v1\",\"max_items_per_chunk\":4,\"items\":["
                         << "{\"task_id\":\"good_one\",\"request\":" << selftest_motor_request << "},"
                         << "{\"task_id\":\"bad\",\"request\":" << bad_hash_request << "},"
                         << "{\"task_id\":\"good_two\",\"request\":" << selftest_motor_request << "},"
                         << "{\"task_id\":\"good_three\",\"request\":" << selftest_motor_request << "}]}";
  }
  MotorSampleRequest selftest_batch_request;
  MotorBatchRequest alias_batch_request;
  const bool selftest_batch_request_valid = ReadMotorSampleRequestJson(
      selftest_motor_request, &selftest_batch_request, &parser_error);
  if (selftest_batch_request_valid) {
    MotorSampleRequest dot_alias_request = selftest_batch_request;
    dot_alias_request.mesh_artifact_path = "./" + batch_artifact_path;
    alias_batch_request.items.push_back({ "canonical_first", selftest_batch_request });
    alias_batch_request.items.push_back({ "canonical_alias", dot_alias_request });
  }
  const MotorBatchArtifactLoadPlan alias_load_plan =
      MakeMotorBatchArtifactLoadPlan(alias_batch_request);
  expect(selftest_batch_request_valid && alias_load_plan.canonical_paths.size() == 1
          && alias_load_plan.item_slots.size() == 2
          && alias_load_plan.item_slots[0] == alias_load_plan.item_slots[1]
          && alias_load_plan.slot_use_counts[0] == 2,
      "motor batch deduplicates canonical artifact-path aliases before SHA and parse");
  const int single_adapter_status = MotorSingleSampleAdapter(batch_single_request_path, batch_single_response_path);
  const int batch_adapter_status = MotorBatchAdapter(batch_request_path, batch_response_path, true);
  const int batch_default_adapter_status = MotorBatchAdapter(batch_request_path, batch_default_response_path);
  const int mixed_adapter_status = MotorBatchAdapter(mixed_request_path, mixed_response_path);
  std::ifstream single_response_input(batch_single_response_path); std::stringstream single_response_bytes;
  single_response_bytes << single_response_input.rdbuf();
  std::ifstream batch_response_input(batch_response_path); std::stringstream batch_response_bytes;
  batch_response_bytes << batch_response_input.rdbuf();
  std::ifstream batch_default_response_input(batch_default_response_path);
  std::stringstream batch_default_response_bytes;
  batch_default_response_bytes << batch_default_response_input.rdbuf();
  std::ifstream mixed_response_input(mixed_response_path); std::stringstream mixed_response_bytes;
  mixed_response_bytes << mixed_response_input.rdbuf();
  StrictJson single_response_json, batch_response_json;
  StrictJson mixed_response_json;
  const bool single_response_valid = single_adapter_status == 0
      && StrictJsonParser(single_response_bytes.str()).Parse(&single_response_json, &parser_error);
  const bool batch_response_valid = batch_adapter_status == 0
      && StrictJsonParser(batch_response_bytes.str()).Parse(&batch_response_json, &parser_error);
  const bool batch_default_response_valid = batch_default_adapter_status == 0
      && batch_default_response_bytes.str().find("\"timing\"") == std::string::npos;
  const bool mixed_response_valid = mixed_adapter_status == 1
      && StrictJsonParser(mixed_response_bytes.str()).Parse(&mixed_response_json, &parser_error);
  const StrictJson* batch_status = batch_response_valid ? JsonMember(batch_response_json, "status", StrictJson::Type::kString, &parser_error) : nullptr;
  const StrictJson* batch_solve_status = batch_response_valid ? JsonMember(batch_response_json, "solve_status", StrictJson::Type::kString, &parser_error) : nullptr;
  const StrictJson* batch_items = batch_response_valid ? JsonMember(batch_response_json, "items", StrictJson::Type::kArray, &parser_error) : nullptr;
  expect(single_response_valid && batch_response_valid && batch_default_response_valid
          && batch_status != nullptr && batch_solve_status != nullptr
          && batch_status->string == "PASS" && batch_solve_status->string == "PASS"
          && batch_items != nullptr && batch_items->array.size() == 4
          && batch_items->array[0].object.at("task_id").string == "first"
          && batch_items->array[1].object.at("task_id").string == "second"
          && batch_items->array[2].object.at("task_id").string == "third"
          && batch_items->array[3].object.at("task_id").string == "fourth"
          && batch_response_bytes.str().find("\"mesh_cache_hits\": 3") != std::string::npos
          && batch_response_bytes.str().find("\"actual_parallel_width\": 4") != std::string::npos
          && batch_response_bytes.str().find("\"batched_pcg_launches\": ") != std::string::npos
          && batch_response_bytes.str().find("\"csr_symbolic_cache_hits\": ") != std::string::npos
          && batch_response_bytes.str().find(
              "\"schema_version\": \"gpu_femm_motor_batch_timing_v1\"") != std::string::npos
          && batch_response_bytes.str().find("\"host_assembly_seconds\": ") != std::string::npos
          && batch_response_bytes.str().find("\"postprocess_seconds\": ") != std::string::npos,
      "four-item motor batch adapter preserves PASS, order, and cache evidence");
  const StrictJson* mixed_status = mixed_response_valid ? JsonMember(mixed_response_json, "status", StrictJson::Type::kString, &parser_error) : nullptr;
  const StrictJson* mixed_items = mixed_response_valid ? JsonMember(mixed_response_json, "items", StrictJson::Type::kArray, &parser_error) : nullptr;
  expect(mixed_status != nullptr && mixed_status->string == "FAIL" && mixed_items != nullptr
          && mixed_items->array.size() == 4
          && mixed_response_bytes.str().find("\"timing\"") == std::string::npos
          && mixed_items->array[0].object.at("task_id").string == "good_one"
          && mixed_items->array[0].object.at("response").object.at("status").string == "PASS"
          && mixed_items->array[1].object.at("task_id").string == "bad"
          && mixed_items->array[1].object.at("response").object.at("status").string == "FAIL"
          && mixed_items->array[3].object.at("task_id").string == "good_three"
          && mixed_items->array[3].object.at("response").object.at("status").string == "PASS"
          && mixed_response_bytes.str().find("\"actual_parallel_width\": 3") != std::string::npos,
      "failed item leaves compacted active batch ordered and independently valid");
  if (single_response_valid && batch_response_valid && batch_items != nullptr && batch_items->array.size() == 4) {
    const StrictJson& first_response = batch_items->array[0].object.at("response");
    const auto same_number = [&parser_error](const StrictJson& left, const StrictJson& right, const char* field) {
      const StrictJson* lhs = JsonMember(left, field, StrictJson::Type::kNumber, &parser_error);
      const StrictJson* rhs = JsonMember(right, field, StrictJson::Type::kNumber, &parser_error);
      return lhs != nullptr && rhs != nullptr && lhs->number == rhs->number;
    };
    const auto same_array = [&parser_error](const StrictJson& left, const StrictJson& right, const char* field) {
      const StrictJson* lhs = JsonMember(left, field, StrictJson::Type::kArray, &parser_error);
      const StrictJson* rhs = JsonMember(right, field, StrictJson::Type::kArray, &parser_error);
      if (lhs == nullptr || rhs == nullptr || lhs->array.size() != rhs->array.size()) return false;
      for (size_t index = 0; index < lhs->array.size(); ++index)
        if (lhs->array[index].type != StrictJson::Type::kNumber || rhs->array[index].type != StrictJson::Type::kNumber
            || lhs->array[index].number != rhs->array[index].number) return false;
      return true;
    };
    expect(same_number(single_response_json, first_response, "Fx_N")
            && same_number(single_response_json, first_response, "Fy_N")
            && same_number(single_response_json, first_response, "torque_Nm")
            && same_array(single_response_json, first_response, "actual_circuit_currents_A")
            && same_array(single_response_json, first_response, "circuit_flux_linkage_Wb")
            && same_array(single_response_json, first_response, "airgap_sample_angles_deg")
            && same_array(single_response_json, first_response, "airgap_radial_flux_density_T"),
        "batched B=4 item matches every single-sample output exactly");
  }
  const NonlinearTriangle& selected_triangle = parsed_artifact.model.triangles[0];
  const NonlinearTriangle& air_triangle = parsed_artifact.model.triangles[1];
  FrozenPostprocessOptions group_options;
  group_options.selected_group_number = 20; group_options.air_group_number = 50;
  expect(IsSelectedPostprocessMaterial(parsed_artifact.model.materials[selected_triangle.material], group_options,
             selected_triangle.material)
          && IsAirPostprocessMaterial(parsed_artifact.model.materials[air_triangle.material], group_options,
             air_triangle.material), "group-based selected and air regions are distinct");
  StrictJson duplicate_json;
  StrictJsonParser duplicate_parser("{\"x\":1,\"x\":2}");
  expect(!duplicate_parser.Parse(&duplicate_json, &parser_error),
      "gpu_femm_mesh_v1 parser rejects duplicate keys");

  Model no_boundary = UnitSquareFixture();
  no_boundary.dirichlet_nodes.clear();
  no_boundary.dirichlet_a_wb_per_m.clear();
  expect(ValidateModel(no_boundary) == Status::kBoundaryInvalid,
      "missing boundary returns BOUNDARY_INVALID");
  Model clockwise = UnitSquareFixture();
  std::swap(clockwise.triangles[0].node[1], clockwise.triangles[0].node[2]);
  expect(ValidateModel(clockwise) == Status::kMeshInvalid,
      "clockwise element returns MESH_INVALID");
  Model floating_component = UnitSquareFixture();
  floating_component.nodes.push_back({ 2.0, 2.0 });
  expect(ValidateModel(floating_component) == Status::kBoundaryInvalid,
      "floating component returns BOUNDARY_INVALID");
  expect(std::string(StatusName(Status::kLinearSolveNotConverged)) == "LINEAR_SOLVE_NOT_CONVERGED",
      "stable error-code name");
  expect(std::string(StatusName(Status::kNonlinearSolveNotConverged)) == "NONLINEAR_SOLVE_NOT_CONVERGED",
      "stable nonlinear error-code name");

  // Analytic P1 contract: b_i=(dN_i/dy,-dN_i/dx), f_PM=S Hc.b_i.
  const Node analytic_nodes[3] = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 } };
  NonlinearElementTerms analytic_terms;
  expect(BuildElementTerms(analytic_nodes, &analytic_terms), "analytic element terms");
  const double expected_pm[3] = { -5.0, 0.0, 5.0 };
  for (int i = 0; i < 3; ++i) {
    expect(Near(analytic_terms.area_m2 * 10.0 * analytic_terms.b_x[i], expected_pm[i]),
        "PM source sign");
  }
  // At A=[0,0,1], v=1 and dv/d(B^2)=.5.  This is K+C exactly.
  const double bx = analytic_terms.b_x[2];
  const double by = analytic_terms.b_y[2];
  const double expected_jacobian[3][3] = {
    { 1.5, -0.5, -1.0 }, { -0.5, 0.5, 0.0 }, { -1.0, 0.0, 1.0 }
  };
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double b_dot_i = bx * analytic_terms.b_x[i] + by * analytic_terms.b_y[i];
      const double b_dot_j = bx * analytic_terms.b_x[j] + by * analytic_terms.b_y[j];
      const double actual = analytic_terms.area_m2
              * (analytic_terms.b_x[i] * analytic_terms.b_x[j]
                  + analytic_terms.b_y[i] * analytic_terms.b_y[j])
          + 2.0 * analytic_terms.area_m2 * 0.5 * b_dot_i * b_dot_j;
      expect(Near(actual, expected_jacobian[i][j]), "analytic Newton Jacobian");
    }
  }
  NonlinearModel missing_material = NonlinearThreeRegionFixture();
  missing_material.triangles[0].material = -1;
  expect(ValidateNonlinearModel(missing_material) == Status::kInvalidMaterial,
      "unassigned nonlinear material is rejected");

  // DC weighted-stress primitive: T*h in free space, in SI units.
  const double primitive_bx = 2.0, primitive_by = 3.0;
  const double primitive_hx = 0.25, primitive_hy = -0.5;
  const double primitive_fx = ((primitive_bx * primitive_bx - primitive_by * primitive_by)
      * primitive_hx + 2.0 * primitive_bx * primitive_by * primitive_hy) / (2.0 * kMu0);
  const double primitive_fy = (2.0 * primitive_bx * primitive_by * primitive_hx
      + (primitive_by * primitive_by - primitive_bx * primitive_bx) * primitive_hy) / (2.0 * kMu0);
  expect(Near(primitive_fx, -7.25 / (2.0 * kMu0)), "weighted-stress Fx primitive");
  expect(Near(primitive_fy, 0.5 / (2.0 * kMu0)), "weighted-stress Fy primitive");

  // A=x gives B=(0,-1) in every P1 triangle.  The selected inner PM square
  // is surrounded by air, so FEMM's WeightingScheme=0 mask is valid and its
  // net load must vanish in a uniform field.  Polar air samples exercise the
  // default-smoothed point field and B dot e_r projection.
  const NonlinearModel postprocess_model = PostprocessRingFixture();
  NonlinearSolveResult postprocess_solution;
  postprocess_solution.a_wb_per_m.resize(postprocess_model.nodes.size());
  for (size_t node = 0; node < postprocess_model.nodes.size(); ++node)
    postprocess_solution.a_wb_per_m[node] = postprocess_model.nodes[node].x_m;
  postprocess_solution.bx_t.assign(postprocess_model.triangles.size(), 0.0);
  postprocess_solution.by_t.assign(postprocess_model.triangles.size(), -1.0);
  FrozenPostprocessOptions postprocess_options;
  postprocess_options.airgap_radius_m = 0.75;
  postprocess_options.airgap_angles_rad = { 0.0, 0.5 * 3.141592653589793238462643383279502884 };
  const FrozenPostprocessResult postprocess = ComputeFrozenPostprocess(
      postprocess_model, postprocess_solution, postprocess_options);
  expect(postprocess.status == Status::kOk,
      std::string("weighted-stress mask postprocess: ") + StatusName(postprocess.status));
  if (postprocess.status == Status::kOk) {
    expect(Near(postprocess.force_x_n, 0.0, 1e-8), "uniform-field weighted Fx cancellation");
    expect(Near(postprocess.force_y_n, 0.0, 1e-8), "uniform-field weighted Fy cancellation");
    expect(Near(postprocess.torque_nm, 0.0, 1e-8), "uniform-field weighted torque cancellation");
    expect(postprocess.airgap_samples.size() == 2, "airgap sample count");
    if (postprocess.airgap_samples.size() == 2) {
      expect(Near(postprocess.airgap_samples[0].radial_b_t, 0.0, 1e-12),
          "airgap radial B at zero degrees");
      expect(Near(postprocess.airgap_samples[1].radial_b_t, -1.0, 1e-12),
          "airgap radial B at ninety degrees");
    }
  }
  FrozenPostprocessOptions invalid_selection = postprocess_options;
  invalid_selection.selected_material_label = 2;
  const FrozenPostprocessResult no_selection = ComputeFrozenPostprocess(
      postprocess_model, postprocess_solution, invalid_selection);
  expect(no_selection.status == Status::kInvalidMaterial,
      "missing selected material is rejected");

  // Group selection is the production artifact path: all rotor labels in
  // group 20 are selected together and only group 30 is free-space air.
  NonlinearModel group_postprocess_model = PostprocessRingFixture();
  group_postprocess_model.circuit_count = 2;
  group_postprocess_model.materials[1].group_number = 20;
  group_postprocess_model.materials[3].group_number = 30;
  NonlinearP1FixtureSolver group_solver;
  const Status group_initialize = group_solver.Initialize(group_postprocess_model);
  NonlinearSolveResult group_solution;
  if (group_initialize == Status::kOk)
    group_solution = group_solver.Solve(std::vector<double> { 0.0, 0.0 });
  FrozenPostprocessOptions artifact_group_options;
  artifact_group_options.selected_group_number = 20;
  artifact_group_options.air_group_number = 30;
  artifact_group_options.airgap_radius_m = 0.75;
  artifact_group_options.airgap_angles_rad = { 0.0 };
  const FrozenPostprocessResult group_postprocess = group_initialize == Status::kOk
      && group_solution.info.status == Status::kOk
      ? ComputeFrozenPostprocess(group_postprocess_model, group_solution, artifact_group_options)
      : FrozenPostprocessResult {};
  expect(group_initialize == Status::kOk && group_solution.circuit_flux_linkage_wb.size() == 2
          && group_postprocess.status == Status::kOk && group_postprocess.airgap_samples.size() == 1,
      "two-circuit group-based artifact solve and postprocess");

  // 258^2 nodes imply a 35 GiB dense double matrix.  This only assembles the
  // analytic stencil (no long FEMM/CUDA solve); it is a regression guard for
  // accidentally reintroducing node_count^2 host storage in all three paths.
  constexpr int kScalabilityCellsPerSide = 257;
  constexpr size_t kScalabilityNodesPerSide = kScalabilityCellsPerSide + 1;
  constexpr size_t kScalabilityNodeCount = kScalabilityNodesPerSide * kScalabilityNodesPerSide;
  constexpr size_t kDenseBytesAvoided = kScalabilityNodeCount * kScalabilityNodeCount * sizeof(double);
  expect(kDenseBytesAvoided > static_cast<size_t>(32) * 1024 * 1024 * 1024,
      "scalability fixture exceeds practical dense host allocation");
  {
    const Model scalable_linear = LargeStructuredLinearFixture(kScalabilityCellsPerSide);
    Assembly scalable_assembly;
    const Status scalable_status = Assemble(scalable_linear, &scalable_assembly);
    expect(scalable_status == Status::kOk,
        std::string("large sparse linear assembly: ") + StatusName(scalable_status));
    expect(scalable_assembly.free_nodes.size() + scalable_linear.dirichlet_nodes.size()
            == scalable_linear.nodes.size()
            && scalable_assembly.values.size() < scalable_linear.triangles.size() * 9,
        "large linear assembly retains P1 sparsity");
  }
  {
    const NonlinearModel scalable_mask = LargeStructuredMaskFixture(kScalabilityCellsPerSide);
    const std::vector<double> zero_a(scalable_mask.nodes.size(), 0.0);
    Assembly scalable_newton;
    const Status newton_status = AssembleNonlinearNewton(
        scalable_mask, zero_a, 0.0, false, &scalable_newton);
    expect(newton_status == Status::kOk,
        std::string("large sparse nonlinear assembly: ") + StatusName(newton_status));
    expect(scalable_newton.free_nodes.size() == 3
            && scalable_newton.values.size() <= scalable_mask.triangles.size() * 9,
        "large nonlinear assembly retains P1 sparsity");
    FrozenPostprocessOptions scalable_mask_options;
    scalable_mask_options.selected_material_label = 1;
    scalable_mask_options.air_material_label = 0;
    std::vector<double> scalable_mask_values;
    const Status mask_status = BuildWeightedStressMask(
        scalable_mask, scalable_mask_options, &scalable_mask_values);
    expect(mask_status == Status::kOk,
        std::string("large sparse WST mask assembly: ") + StatusName(mask_status));
    expect(scalable_mask_values.size() == scalable_mask.nodes.size(),
        "large WST mask dimensions");
  }

  Assembly three_by_three;
  three_by_three.row_offsets = { 0, 2, 5, 7 };
  three_by_three.column_indices = { 0, 1, 0, 1, 2, 1, 2 };
  three_by_three.values = { 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0 };
  three_by_three.diagonal = { 4.0, 4.0, 3.0 };
  three_by_three.free_nodes = { 0, 1, 2 };
  GpuCsrSolver csr_solver;
  Status csr_status = csr_solver.Initialize(three_by_three);
  expect(csr_status == Status::kOk, "3x3 CSR initialization");
  std::vector<double> csr_solution;
  const SolveInfo csr_info = csr_solver.Solve({ 2.0, 4.0, 7.0 }, 1e-13, 64, &csr_solution);
  expect(csr_info.status == Status::kOk, "3x3 PCG convergence");
  expect(csr_solution.size() == 3 && Near(csr_solution[0], 1.0) && Near(csr_solution[1], 2.0) && Near(csr_solution[2], 3.0),
      "3x3 PCG reference parity");

  // 257 exercises the strided reduction tail beyond one 256-thread block.
  Assembly reduction_tail;
  constexpr int kReductionTailN = kPcgBlockThreads + 1;
  reduction_tail.row_offsets.resize(kReductionTailN + 1);
  reduction_tail.column_indices.resize(kReductionTailN);
  reduction_tail.values.assign(kReductionTailN, 1.0);
  reduction_tail.diagonal.assign(kReductionTailN, 1.0);
  reduction_tail.free_nodes.resize(kReductionTailN);
  std::vector<double> reduction_rhs(kReductionTailN);
  for (int i = 0; i < kReductionTailN; ++i) {
    reduction_tail.row_offsets[i] = i; reduction_tail.column_indices[i] = i;
    reduction_tail.free_nodes[i] = i; reduction_rhs[i] = static_cast<double>((i % 11) - 5);
  }
  reduction_tail.row_offsets[kReductionTailN] = kReductionTailN;
  GpuCsrSolver reduction_solver;
  std::vector<double> reduction_solution;
  const Status reduction_initialize = reduction_solver.Initialize(reduction_tail);
  const SolveInfo reduction_info = reduction_initialize == Status::kOk
      ? reduction_solver.Solve(reduction_rhs, 1e-13, 16, &reduction_solution) : SolveInfo {};
  bool reduction_matches = reduction_info.status == Status::kOk && reduction_solution.size() == reduction_rhs.size();
  for (size_t index = 0; reduction_matches && index < reduction_rhs.size(); ++index)
    reduction_matches = reduction_solution[index] == reduction_rhs[index];
  expect(reduction_matches,
      "deterministic reduction handles non-multiple-of-block-width RHS");

  Assembly indefinite;
  indefinite.row_offsets = { 0, 1 };
  indefinite.column_indices = { 0 };
  indefinite.values = { -1.0 };
  indefinite.diagonal = { -1.0 };
  indefinite.free_nodes = { 0 };
  GpuCsrSolver breakdown_solver;
  Status breakdown_status = breakdown_solver.Initialize(indefinite);
  expect(breakdown_status == Status::kOk, "breakdown fixture initialization");
  std::vector<double> breakdown_solution;
  const SolveInfo breakdown_info = breakdown_solver.Solve({ 1.0 }, 1e-13, 8, &breakdown_solution);
  expect(breakdown_info.status == Status::kLinearSolveBreakdown,
      "indefinite matrix returns LINEAR_SOLVE_BREAKDOWN");

  LinearP1FixtureSolver solver;
  Status initialize_status = solver.Initialize(UnitSquareFixture());
  expect(initialize_status == Status::kOk,
      std::string("fixture initialization: ") + StatusName(initialize_status));
  if (initialize_status != Status::kOk)
    return failures + 1;

  const SolveResult positive = solver.Solve(12.0);
  expect(positive.info.status == Status::kOk,
      std::string("positive solve: ") + StatusName(positive.info.status));
  const double expected_a[5] = { 0.0, 0.0, 0.0, 0.0, 1.0 };
  const double expected_bx[4] = { 2.0, 0.0, -2.0, 0.0 };
  const double expected_by[4] = { 0.0, 2.0, 0.0, -2.0 };
  expect(positive.a_wb_per_m.size() == 5 && positive.bx_t.size() == 4 && positive.by_t.size() == 4,
      "positive result dimensions");
  if (positive.a_wb_per_m.size() == 5 && positive.bx_t.size() == 4 && positive.by_t.size() == 4) {
    for (int i = 0; i < 5; ++i) {
      expect(Near(positive.a_wb_per_m[i], expected_a[i]), "nodal A parity");
    }
    for (int i = 0; i < 4; ++i) {
      expect(Near(positive.bx_t[i], expected_bx[i]), "Bx sign/parity");
      expect(Near(positive.by_t[i], expected_by[i]), "By sign/parity");
    }
  }
  expect(Near(positive.flux_linkage_wb, 1.0 / 3.0), "flux-linkage parity");
  expect(positive.info.residual_l2 <= 1e-13, "linear residual tolerance");

  const SolveResult negative = solver.Solve(-12.0);
  expect(negative.info.status == Status::kOk, "negative-current solve");
  expect(negative.a_wb_per_m.size() == 5 && negative.bx_t.size() == 4 && negative.by_t.size() == 4,
      "negative result dimensions");
  if (negative.a_wb_per_m.size() == 5 && negative.bx_t.size() == 4 && negative.by_t.size() == 4) {
    for (int i = 0; i < 5; ++i) {
      expect(Near(negative.a_wb_per_m[i], -expected_a[i]), "negative A sign");
    }
    for (int i = 0; i < 4; ++i) {
      expect(Near(negative.bx_t[i], -expected_bx[i]), "negative Bx sign");
      expect(Near(negative.by_t[i], -expected_by[i]), "negative By sign");
    }
  }
  expect(Near(negative.flux_linkage_wb, -1.0 / 3.0), "negative flux sign");

  const SolveResult tiny = solver.Solve(1e-14);
  expect(tiny.info.status == Status::kOk && tiny.info.iterations > 0,
      "small nonzero RHS is solved, not accepted at x=0");
  expect(tiny.a_wb_per_m.size() == 5 && Near(tiny.a_wb_per_m[4], 1e-14 / 12.0, 1e-28),
      "small nonzero RHS parity");
  const SolveResult subnormal_square = solver.Solve(1e-200);
  expect(subnormal_square.info.status == Status::kOk && subnormal_square.info.iterations > 0,
      "subnormal-square RHS is solved, not accepted at x=0");
  expect(subnormal_square.a_wb_per_m.size() == 5 && Near(subnormal_square.a_wb_per_m[4], 1e-200 / 12.0, 1e-214),
      "subnormal-square RHS parity");
  const SolveResult invalid_tolerance = solver.Solve(12.0, std::numeric_limits<double>::infinity(), 8);
  expect(invalid_tolerance.info.status == Status::kInvalidArgument,
      "infinite tolerance returns INVALID_ARGUMENT");

  const SolveResult not_converged = solver.Solve(12.0, 1e-13, 0);
  expect(not_converged.info.status == Status::kLinearSolveNotConverged,
      "iteration limit returns LINEAR_SOLVE_NOT_CONVERGED");

  NonlinearP1FixtureSolver nonlinear_solver;
  const Status nonlinear_initialize = nonlinear_solver.Initialize(NonlinearThreeRegionFixture());
  expect(nonlinear_initialize == Status::kOk,
      std::string("nonlinear fixture initialization: ") + StatusName(nonlinear_initialize));
  if (nonlinear_initialize == Status::kOk) {
    const NonlinearSolveResult nonlinear = nonlinear_solver.Solve(2.0);
    expect(nonlinear.info.status == Status::kOk,
        std::string("nonlinear PM+coil solve: ") + StatusName(nonlinear.info.status));
    expect(nonlinear.a_wb_per_m.size() == 5 && nonlinear.bx_t.size() == 4
            && nonlinear.by_t.size() == 4 && nonlinear.residual_history.size() >= 2,
        "nonlinear result dimensions and history");
    expect(std::isfinite(nonlinear.flux_linkage_wb), "nonlinear circuit flux");
    if (nonlinear.info.status == Status::kOk) {
      const NonlinearSolveResult warm = nonlinear_solver.Solve(2.0, {}, &nonlinear.a_wb_per_m);
      expect(warm.info.status == Status::kOk, "nonlinear warm-start solve");
      expect(nonlinear_solver.csr_symbolic_reuse_count() > 0,
          "nonlinear Newton corrections reuse uploaded CSR structure");
      expect(warm.info.iterations <= nonlinear.info.iterations, "warm start does not add Newton steps");
      expect(warm.a_wb_per_m.size() == nonlinear.a_wb_per_m.size(), "warm result dimensions");
      if (warm.a_wb_per_m.size() == nonlinear.a_wb_per_m.size()) {
        for (size_t i = 0; i < warm.a_wb_per_m.size(); ++i)
          expect(Near(warm.a_wb_per_m[i], nonlinear.a_wb_per_m[i], 1e-9),
              "warm-start solution parity");
      }
    }
    NonlinearOptions no_iterations;
    no_iterations.max_newton_iterations = 0;
    const NonlinearSolveResult nonlinear_limited = nonlinear_solver.Solve(2.0, no_iterations);
    expect(nonlinear_limited.info.status == Status::kNonlinearSolveNotConverged,
        "nonlinear iteration cap returns NONLINEAR_SOLVE_NOT_CONVERGED");
  }

  NonlinearModel invalid_circuit = NonlinearTwoCircuitFixture();
  invalid_circuit.materials[2].circuit_index = 2;
  expect(ValidateNonlinearModel(invalid_circuit) == Status::kInvalidMaterial,
      "out-of-range nonlinear circuit label is rejected");
  NonlinearP1FixtureSolver two_circuit_solver;
  const Status two_circuit_initialize = two_circuit_solver.Initialize(NonlinearTwoCircuitFixture());
  expect(two_circuit_initialize == Status::kOk, "two-circuit fixture initialization");
  if (two_circuit_initialize == Status::kOk) {
    const NonlinearSolveResult circuit_0 = two_circuit_solver.Solve(
        std::vector<double> { 2.0, 0.0 });
    const NonlinearSolveResult circuit_1 = two_circuit_solver.Solve(
        std::vector<double> { 0.0, -3.0 });
    const NonlinearSolveResult combined = two_circuit_solver.Solve(
        std::vector<double> { 2.0, -3.0 });
    expect(circuit_0.info.status == Status::kOk && circuit_1.info.status == Status::kOk
            && combined.info.status == Status::kOk,
        "two-circuit signed source solves");
    expect(combined.circuit_currents_a.size() == 2
            && combined.circuit_flux_linkage_wb.size() == 2,
        "two-circuit ordered current and flux dimensions");
    if (circuit_0.a_wb_per_m.size() == 5 && circuit_1.a_wb_per_m.size() == 5
        && combined.a_wb_per_m.size() == 5) {
      expect(Near(circuit_0.a_wb_per_m[4], 0.5)
              && Near(circuit_1.a_wb_per_m[4], -0.375)
              && Near(combined.a_wb_per_m[4], 0.125)
              && Near(combined.a_wb_per_m[4],
                  circuit_0.a_wb_per_m[4] + circuit_1.a_wb_per_m[4]),
          "two-circuit independent signed source superposition");
    }
    if (combined.circuit_flux_linkage_wb.size() == 2) {
      expect(Near(combined.circuit_flux_linkage_wb[0], 0.125)
              && Near(combined.circuit_flux_linkage_wb[1], 0.0625)
              && std::isnan(combined.flux_linkage_wb),
          "two-circuit independent flux linkage vector");
    }
    const NonlinearSolveResult wrong_current_count = two_circuit_solver.Solve(
        std::vector<double> { 2.0 });
    expect(wrong_current_count.info.status == Status::kInvalidArgument,
        "wrong nonlinear circuit-current vector length is rejected");
  }

  solver.Solve(12.0); // warm up all runtime paths before memory accounting.
  size_t free_before = 0;
  size_t total_before = 0;
  const cudaError_t memory_before_status = cudaMemGetInfo(&free_before, &total_before);
  expect(memory_before_status == cudaSuccess, "pre-repeat cudaMemGetInfo");
  for (int repeat = 0; repeat < 100; ++repeat) {
    const SolveResult repeated = solver.Solve(12.0);
    expect(SameBits(positive, repeated), "100-run bitwise determinism");
  }
  size_t free_after = 0;
  size_t total_after = 0;
  const cudaError_t memory_after_status = cudaMemGetInfo(&free_after, &total_after);
  expect(memory_after_status == cudaSuccess, "post-repeat cudaMemGetInfo");
  if (memory_before_status == cudaSuccess && memory_after_status == cudaSuccess) {
    expect(free_after >= free_before, "no GPU-memory growth across 100 solves");
  }

  if (failures == 0) {
    std::cout << "PASS gpu_linear_p1_selftest\n"
              << "  status=" << StatusName(positive.info.status)
              << " iterations=" << positive.info.iterations
              << " residual=" << positive.info.residual_l2 << '\n'
              << "  A_center=" << positive.a_wb_per_m.at(4)
              << " flux_Wb=" << positive.flux_linkage_wb << '\n'
              << "  deterministic_repeats=100 gpu_memory_delta_bytes="
              << static_cast<long long>(free_before) - static_cast<long long>(free_after)
              << '\n';
  }
  return failures == 0 ? 0 : 1;
}

} // namespace gpu_femm

int main(int argc, char** argv)
{
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    return gpu_femm::SelfTest();
  }
  if (argc == 3 && std::string(argv[1]) == "--femm-reference") {
    return gpu_femm::FemmReferenceTest(argv[2]);
  }
  if (argc == 4 && std::string(argv[1]) == "--nonlinear-reference") {
    return gpu_femm::NonlinearFemmReferenceTest(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--postprocess-reference") {
    return gpu_femm::FrozenPostprocessReferenceTest(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--single-sample") {
    return gpu_femm::SingleSampleAdapter(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--motor-single-sample") {
    return gpu_femm::MotorSingleSampleAdapter(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--motor-batch") {
    return gpu_femm::MotorBatchAdapter(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--motor-batch-profile") {
    return gpu_femm::MotorBatchAdapter(argv[2], argv[3], true);
  }
  if (argc == 3 && std::string(argv[1]) == "--mesh-artifact") {
    return gpu_femm::GpuFemmMeshArtifactTest(argv[2]);
  }
  std::cerr << "Usage: gpu_linear_p1_poc --self-test | --femm-reference <stem>"
            << " | --nonlinear-reference <stem> <curve_dir>"
            << " | --postprocess-reference <stem> <curve_dir>"
            << " | --single-sample <request.json> <response.json>"
            << " | --motor-single-sample <request.json> <response.json>"
            << " | --motor-batch <request.json> <response.json>"
            << " | --motor-batch-profile <request.json> <response.json>"
            << " | --mesh-artifact <gpu_femm_mesh_v1.json>\n";
  return 2;
}

#include "cudss_direct_solver.cuh"
#include "../core/profile.h"
#include "pcg_device.cuh"
#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <mutex>

namespace gpu_femm
{

#ifdef GPU_FEMM_ENABLE_CUDSS
namespace
{
// cuDSS permits streams, not concurrent analysis/allocation API calls from
// arbitrary host threads. One dispatcher submits independent per-stream plans.
std::mutex execution_mutex;
std::mutex stats_mutex;
constexpr size_t kDirectStreams = 2;

struct DirectOrdering {
  std::vector<int32_t> permutation, partition_tree;
};
struct CachedOrdering {
  int device = 0;
  std::vector<int32_t> rows, columns;
  DirectOrdering ordering;
  size_t Bytes() const
  {
    return sizeof(int32_t) * (rows.size() + columns.size() + ordering.permutation.size() +
                              ordering.partition_tree.size());
  }
};
// CPU ordering metadata only, never factors or field solutions. An exact CSR
// comparison (not a hash, path or timestamp) is required before reuse.
std::deque<CachedOrdering> ordering_cache;
size_t ordering_cache_bytes = 0;
constexpr size_t kOrderingCacheBytes = 128 * 1024 * 1024;

const DirectOrdering *FindOrdering(int device, const std::vector<int32_t> &rows,
                                   const std::vector<int32_t> &columns)
{
  for (const auto &entry : ordering_cache)
    if (entry.device == device && entry.rows == rows && entry.columns == columns)
      return &entry.ordering;
  return nullptr;
}

void SaveOrdering(int device, const std::vector<int32_t> &rows, const std::vector<int32_t> &columns,
                  const DirectOrdering &ordering)
{
  const size_t bytes =
      sizeof(int32_t) *
      (rows.size() + columns.size() + ordering.permutation.size() + ordering.partition_tree.size());
  if (bytes > kOrderingCacheBytes)
    return;
  while (!ordering_cache.empty() &&
         (ordering_cache.size() >= 64 || ordering_cache_bytes + bytes > kOrderingCacheBytes)) {
    ordering_cache_bytes -= ordering_cache.front().Bytes();
    ordering_cache.pop_front();
  }
  ordering_cache.push_back({device, rows, columns, ordering});
  ordering_cache_bytes += bytes;
}

Status Call(cudssStatus_t value)
{
  if (value == CUDSS_STATUS_SUCCESS)
    return Status::kOk;
  std::cerr << "cuDSS API error: " << static_cast<int>(value) << '\n';
  return value == CUDSS_STATUS_ALLOC_FAILED    ? Status::kGpuAllocationFailed
         : value == CUDSS_STATUS_NOT_SUPPORTED ? Status::kUnsupportedFeature
                                               : Status::kInternalError;
}

struct DirectStatsRecord {
  DirectSolverStats value;
  ~DirectStatsRecord()
  {
    std::lock_guard<std::mutex> lock(stats_mutex);
    direct_solver_stats.analyses += value.analyses;
    direct_solver_stats.factorizations += value.factorizations;
    direct_solver_stats.solves += value.solves;
    direct_solver_stats.factor_reuses += value.factor_reuses;
    direct_solver_stats.peak_device_estimate_bytes =
        std::max(direct_solver_stats.peak_device_estimate_bytes, value.peak_device_estimate_bytes);
    direct_solver_stats.analysis_seconds += value.analysis_seconds;
    direct_solver_stats.factor_seconds += value.factor_seconds;
    direct_solver_stats.solve_seconds += value.solve_seconds;
    direct_solver_stats.parallel_batches += value.parallel_batches;
    direct_solver_stats.ordering_reuses += value.ordering_reuses;
    direct_solver_stats.peak_parallel_width =
        std::max(direct_solver_stats.peak_parallel_width, value.peak_parallel_width);
  }
};

// Keep exactly the original per-system reduction order and full-CSR residual.
// Blocks are independent; batching the validation does not combine equations.
__global__ void CheckDirectSolutions(int n, const int32_t *rows, const int32_t *columns,
                                     const int32_t *transpose, const double *values,
                                     const double *diagonal, const double *rhs, double *x,
                                     size_t value_stride, bool solved, double tolerance,
                                     SolveInfo *info)
{
  const size_t item = blockIdx.x;
  values += item * value_stride;
  diagonal += value_stride ? item * n : 0;
  rhs += item * n;
  x += item * n;
  info += item;
  __shared__ double scratch[kPcgBlockThreads];
  const int lane = threadIdx.x;
  double invalid = 0, asymmetric = 0, scale = 0;
  for (int row = lane; row < n; row += blockDim.x) {
    if (!isfinite(rhs[row]) || !isfinite(diagonal[row]) || (solved && !isfinite(x[row])))
      invalid = 2;
    else if (!(diagonal[row] > 0))
      invalid = fmax(invalid, 1.0);
    scale = fmax(scale, fabs(rhs[row]));
    for (int i = rows[row]; i < rows[row + 1]; ++i) {
      const double value = values[i], other = values[transpose[i]];
      if (!isfinite(value))
        invalid = 2;
      const double entry_scale =
          fmax(fabs(value), fmax(fabs(diagonal[row]), fabs(diagonal[columns[i]])));
      if (fabs(value - other) > 1e-13 * entry_scale)
        asymmetric = 1;
    }
  }
  invalid = DeterministicBlockMax(invalid, scratch);
  asymmetric = DeterministicBlockMax(asymmetric, scratch);
  scale = DeterministicBlockMax(scale, scratch);
  if (invalid || asymmetric) {
    if (lane == 0) {
      info->status = invalid == 2 ? Status::kNumericalNonfinite
                     : invalid    ? Status::kLinearSolveBreakdown
                                  : Status::kUnsupportedFeature;
      info->iterations = 0;
      info->residual_l2 = INFINITY;
    }
    return;
  }
  if (!solved || scale == 0) {
    if (scale == 0)
      for (int row = lane; row < n; row += blockDim.x)
        x[row] = 0;
    if (lane == 0) {
      info->status = Status::kOk;
      info->iterations = 0;
      info->residual_l2 = scale == 0 ? 0 : INFINITY;
    }
    return;
  }
  double rr = 0, bb = 0;
  for (int row = lane; row < n; row += blockDim.x) {
    double product = 0;
    for (int i = rows[row]; i < rows[row + 1]; ++i)
      product += values[i] * (x[columns[i]] / scale);
    const double b = rhs[row] / scale, r = product - b;
    rr += r * r;
    bb += b * b;
  }
  rr = DeterministicBlockSum(rr, scratch);
  bb = DeterministicBlockSum(bb, scratch);
  if (lane == 0) {
    info->iterations = 0;
    info->residual_l2 = sqrt(rr / bb);
    info->status = !isfinite(info->residual_l2)     ? Status::kNumericalNonfinite
                   : info->residual_l2 <= tolerance ? Status::kOk
                                                    : Status::kLinearSolveNotConverged;
  }
}
} // namespace

// Each context owns its factors, stream and error state. CSR indices are
// immutable borrowed buffers; no numeric factors cross between current states.
struct CudssContext : DirectOrdering {
  cudaStream_t stream = nullptr;
  cudaEvent_t factor_begin = nullptr, factor_end = nullptr;
  cudaEvent_t solve_begin = nullptr, solve_end = nullptr;
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t a = nullptr, b = nullptr, x = nullptr;
  const double *fact_values = nullptr;
  bool factor_valid = false;
  size_t estimate_bytes = 0;

  ~CudssContext()
  {
    if (stream)
      cudaStreamSynchronize(stream);
    if (a)
      cudssMatrixDestroy(a);
    if (b)
      cudssMatrixDestroy(b);
    if (x)
      cudssMatrixDestroy(x);
    if (data)
      cudssDataDestroy(handle, data);
    if (config)
      cudssConfigDestroy(config);
    if (handle)
      cudssDestroy(handle);
    if (factor_begin)
      cudaEventDestroy(factor_begin);
    if (factor_end)
      cudaEventDestroy(factor_end);
    if (solve_begin)
      cudaEventDestroy(solve_begin);
    if (solve_end)
      cudaEventDestroy(solve_end);
    if (stream)
      cudaStreamDestroy(stream);
  }

  Status Initialize(int n, int nnz, int32_t *rows, int32_t *columns, const double *values,
                    const double *rhs, double *solution, DirectSolverStats *stats,
                    const DirectOrdering *ordering = nullptr)
  {
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreate(&factor_begin) != cudaSuccess ||
        cudaEventCreate(&factor_end) != cudaSuccess ||
        cudaEventCreate(&solve_begin) != cudaSuccess || cudaEventCreate(&solve_end) != cudaSuccess)
      return Status::kInternalError;
    Status status;
    if ((status = Call(cudssCreate(&handle))) != Status::kOk ||
        (status = Call(cudssSetStream(handle, stream))) != Status::kOk ||
        (status = Call(cudssConfigCreate(&config))) != Status::kOk ||
        (status = Call(cudssDataCreate(handle, &data))) != Status::kOk)
      return status;
    const int refinement_steps = 0, deterministic = 1;
    const double pivot_epsilon = 0;
    if ((status = Call(cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS, &refinement_steps,
                                      sizeof(refinement_steps)))) != Status::kOk ||
        (status = Call(cudssConfigSet(config, CUDSS_CONFIG_DETERMINISTIC_MODE, &deterministic,
                                      sizeof(deterministic)))) != Status::kOk ||
        (status = Call(cudssConfigSet(config, CUDSS_CONFIG_PIVOT_EPSILON, &pivot_epsilon,
                                      sizeof(pivot_epsilon)))) != Status::kOk ||
        (status = Call(cudssMatrixCreateCsr(&a, n, n, nnz, rows, nullptr, columns, values,
                                            CUDA_R_32I, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SPD,
                                            CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO))) != Status::kOk ||
        (status = Call(cudssMatrixCreateDn(&b, n, 1, n, rhs, CUDA_R_64F,
                                           CUDSS_LAYOUT_COL_MAJOR))) != Status::kOk ||
        (status = Call(cudssMatrixCreateDn(&x, n, 1, n, solution, CUDA_R_64F,
                                           CUDSS_LAYOUT_COL_MAJOR))) != Status::kOk)
      return status;
    if (ordering) {
      if ((status =
               Call(cudssDataSet(handle, data, CUDSS_DATA_USER_PERM, ordering->permutation.data(),
                                 ordering->permutation.size() * sizeof(int32_t)))) != Status::kOk ||
          (status = Call(cudssDataSet(
               handle, data, CUDSS_DATA_USER_ND_PARTITION_TREE, ordering->partition_tree.data(),
               ordering->partition_tree.size() * sizeof(int32_t)))) != Status::kOk)
        return status;
      permutation = ordering->permutation;
      partition_tree = ordering->partition_tree;
      ++stats->ordering_reuses;
    }
    const auto start = ProfileClock::now();
    if ((status = Enqueue(CUDSS_PHASE_ANALYSIS)) != Status::kOk || (status = Wait()) != Status::kOk)
      return status;
    ++stats->analyses;
    stats->analysis_seconds += ProfileSecondsSince(start);
    int64_t estimates[16] = {};
    size_t written = 0;
    if ((status = Call(cudssDataGet(handle, data, CUDSS_DATA_MEMORY_ESTIMATES, estimates,
                                    sizeof(estimates), &written))) != Status::kOk)
      return status;
    if (written < 2 * sizeof(int64_t) || estimates[1] < 0)
      return Status::kInternalError;
    estimate_bytes = static_cast<size_t>(estimates[1]);
    if (!ordering) {
      auto extract = [&](cudssDataParam_t key, std::vector<int32_t> *output) {
        size_t bytes = 0;
        Status result = Call(cudssDataGet(handle, data, key, nullptr, 0, &bytes));
        if (result != Status::kOk || bytes == 0 || bytes % sizeof(int32_t))
          return result != Status::kOk ? result : Status::kInternalError;
        output->resize(bytes / sizeof(int32_t));
        return Call(cudssDataGet(handle, data, key, output->data(), bytes, &bytes));
      };
      if ((status = extract(CUDSS_DATA_PERM_REORDER_ROW, &permutation)) != Status::kOk ||
          (status = extract(CUDSS_DATA_ND_PARTITION_TREE, &partition_tree)) != Status::kOk)
        return status;
    }
    return Status::kOk;
  }

  Status Bind(const double *values, const double *rhs, double *solution)
  {
    Status status;
    if ((status = Call(cudssMatrixSetValues(a, values))) != Status::kOk ||
        (status = Call(cudssMatrixSetValues(b, rhs))) != Status::kOk)
      return status;
    return Call(cudssMatrixSetValues(x, solution));
  }

  Status Enqueue(int phase) { return Call(cudssExecute(handle, phase, config, data, a, x, b)); }
  Status Wait()
  {
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      return Status::kInternalError;
    int error = 0;
    size_t written = 0;
    const Status status =
        Call(cudssDataGet(handle, data, CUDSS_DATA_INFO, &error, sizeof(error), &written));
    return status != Status::kOk ? status : error ? Status::kLinearSolveBreakdown : Status::kOk;
  }
  double Seconds(cudaEvent_t begin, cudaEvent_t end)
  {
    float milliseconds = 0;
    return cudaEventElapsedTime(&milliseconds, begin, end) == cudaSuccess
               ? static_cast<double>(milliseconds) * 1e-3
               : 0.0;
  }
};

CudssDirectPlan::CudssDirectPlan() = default;
CudssDirectPlan::~CudssDirectPlan() { Reset(); }

void CudssDirectPlan::Reset()
{
  std::lock_guard<std::mutex> lock(execution_mutex);
  contexts_.clear();
  transpose_.reset();
  info_.reset();
  if (host_info_)
    cudaFreeHost(host_info_);
  if (host_solutions_)
    cudaFreeHost(host_solutions_);
  host_info_ = nullptr;
  host_solutions_ = nullptr;
  transfer_capacity_ = 0;
  n_ = nnz_ = 0;
  host_rows_ = host_columns_ = nullptr;
}

void CudssDirectPlan::InvalidateValues()
{
  for (auto &context : contexts_)
    context->factor_valid = false;
}

Status CudssDirectPlan::Initialize(int n, const std::vector<int32_t> &rows,
                                   const std::vector<int32_t> &columns, int32_t *device_rows,
                                   int32_t *device_columns)
{
  Reset();
  n_ = n;
  nnz_ = static_cast<int>(columns.size());
  rows_ = device_rows;
  columns_ = device_columns;
  host_rows_ = &rows;
  host_columns_ = &columns;
  std::vector<int32_t> transpose(columns.size());
  for (int row = 0; row < n; ++row) {
    for (int j = rows[row]; j < rows[row + 1]; ++j) {
      if (j > rows[row] && columns[j - 1] >= columns[j])
        return Status::kInvalidArgument;
      const int col = columns[j];
      auto it = std::lower_bound(columns.begin() + rows[col], columns.begin() + rows[col + 1], row);
      if (it == columns.begin() + rows[col + 1] || *it != row)
        return Status::kUnsupportedFeature;
      transpose[j] = static_cast<int32_t>(it - columns.begin());
    }
  }
  const Status status = transpose_.allocate(transpose.size());
  return status != Status::kOk
             ? status
             : CopyToDevice(transpose_.get(), transpose.data(), transpose.size() * sizeof(int32_t));
}

Status CudssDirectPlan::EnsureTransferCapacity(size_t count)
{
  if (transfer_capacity_ >= count)
    return Status::kOk;
  if (host_info_)
    cudaFreeHost(host_info_);
  if (host_solutions_)
    cudaFreeHost(host_solutions_);
  host_info_ = nullptr;
  host_solutions_ = nullptr;
  transfer_capacity_ = 0;
  const Status status = info_.allocate(count);
  if (status != Status::kOk)
    return status;
  if (cudaMallocHost(reinterpret_cast<void **>(&host_info_), count * sizeof(SolveInfo)) !=
          cudaSuccess ||
      cudaMallocHost(reinterpret_cast<void **>(&host_solutions_),
                     std::min(count, kDirectStreams) * n_ * sizeof(double)) != cudaSuccess)
    return Status::kGpuAllocationFailed;
  transfer_capacity_ = count;
  return Status::kOk;
}

Status CudssDirectPlan::SolveBatch(size_t count, size_t value_stride, const double *values,
                                   const double *diagonal, const double *rhs, double *solution,
                                   double tolerance, int max_iterations,
                                   std::vector<SolveInfo> *infos,
                                   std::vector<std::vector<double>> *solutions, int *launches,
                                   GpuBatchSolveTiming *timing)
{
  if (n_ <= 0 || !count || !(tolerance > 0 && tolerance < 1) || !std::isfinite(tolerance) ||
      max_iterations < 0 || !infos || !solutions || !values || !diagonal || !rhs || !solution ||
      (value_stride && value_stride != static_cast<size_t>(nnz_)))
    return Status::kInvalidArgument;
  std::lock_guard<std::mutex> lock(execution_mutex);
  DirectStatsRecord record;
  auto &stats = record.value;
  bool completed = false;
  struct DrainOnFailure {
    std::vector<std::unique_ptr<CudssContext>> &contexts;
    bool &completed;
    ~DrainOnFailure()
    {
      if (!completed)
        for (auto &context : contexts) {
          cudaStreamSynchronize(context->stream);
          context->factor_valid = false;
        }
    }
  } drain{contexts_, completed};
  const auto start = ProfileClock::now();
  const Status capacity = EnsureTransferCapacity(count);
  if (capacity != Status::kOk)
    return capacity;
  infos->assign(count, SolveInfo{});
  solutions->assign(count, std::vector<double>(n_, 0));
  CheckDirectSolutions<<<static_cast<unsigned>(count), kPcgBlockThreads>>>(
      n_, rows_, columns_, transpose_.get(), values, diagonal, rhs, solution, value_stride, false,
      tolerance, info_.get());
  if (cudaGetLastError() != cudaSuccess ||
      CopyToHost(host_info_, info_.get(), count * sizeof(SolveInfo)) != Status::kOk)
    return Status::kInternalError;
  std::vector<size_t> active;
  for (size_t item = 0; item < count; ++item) {
    (*infos)[item] = host_info_[item];
    if (host_info_[item].status != Status::kOk || host_info_[item].residual_l2 == 0)
      continue;
    if (max_iterations == 0)
      (*infos)[item].status = Status::kLinearSolveNotConverged;
    else
      active.push_back(item);
  }
  // A shared numeric matrix keeps one factorization, including across calls.
  // Different matrices use two deterministic independent contexts. More streams
  // can compete for factorization bandwidth rather than reducing wall time.
  size_t width = std::min(value_stride ? kDirectStreams : size_t{1}, active.size());
  while (contexts_.size() < width) {
    const size_t lane = contexts_.size(), item = active[lane];
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
      return Status::kInternalError;
    if (lane && contexts_.front()->estimate_bytes > free_bytes / 2) {
      width = lane;
      break;
    }
    auto context = std::make_unique<CudssContext>();
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
      return Status::kInternalError;
    const DirectOrdering *ordering =
        lane ? contexts_.front().get() : FindOrdering(device, *host_rows_, *host_columns_);
    const Status status =
        context->Initialize(n_, nnz_, rows_, columns_, values + item * value_stride,
                            rhs + item * n_, solution + item * n_, &stats, ordering);
    if (status != Status::kOk)
      return status;
    if (!ordering)
      SaveOrdering(device, *host_rows_, *host_columns_, *context);
    contexts_.push_back(std::move(context));
  }
  size_t total_estimate = 0;
  for (const auto &context : contexts_)
    total_estimate += context->estimate_bytes;
  stats.peak_device_estimate_bytes = total_estimate;
  stats.peak_parallel_width = width;
  for (size_t begin = 0; begin < active.size(); begin += width) {
    const size_t lanes = std::min(width, active.size() - begin);
    stats.parallel_batches += lanes > 1;
    std::vector<bool> refactored(lanes, false), solved(lanes, false);
    for (size_t lane = 0; lane < lanes; ++lane) {
      const size_t item = active[begin + lane];
      auto &context = *contexts_[lane];
      const double *v = values + item * value_stride;
      Status status = context.Bind(v, rhs + item * n_, solution + item * n_);
      if (status != Status::kOk)
        return status;
      if (value_stride || context.fact_values != v)
        context.factor_valid = false;
      if (context.factor_valid) {
        ++stats.factor_reuses;
        continue;
      }
      const int clear = 0;
      if ((status = Call(cudssDataSet(context.handle, context.data, CUDSS_DATA_INFO, &clear,
                                      sizeof(clear)))) != Status::kOk)
        return status;
      if (cudaEventRecord(context.factor_begin, context.stream) != cudaSuccess)
        return Status::kInternalError;
      status = context.Enqueue(CUDSS_PHASE_FACTORIZATION);
      if (cudaEventRecord(context.factor_end, context.stream) != cudaSuccess)
        return Status::kInternalError;
      ++stats.factorizations;
      refactored[lane] = true;
      (*infos)[item].status = status;
    }
    for (size_t lane = 0; lane < lanes; ++lane) {
      const size_t item = active[begin + lane];
      auto &context = *contexts_[lane];
      if (refactored[lane]) {
        const Status status = context.Wait();
        stats.factor_seconds += context.Seconds(context.factor_begin, context.factor_end);
        if ((*infos)[item].status == Status::kOk)
          (*infos)[item].status = status;
        context.factor_valid = (*infos)[item].status == Status::kOk;
        context.fact_values = values + item * value_stride;
      }
      if ((*infos)[item].status != Status::kOk)
        continue;
      if (cudaEventRecord(context.solve_begin, context.stream) != cudaSuccess)
        return Status::kInternalError;
      const Status status = context.Enqueue(CUDSS_PHASE_SOLVE);
      if (cudaEventRecord(context.solve_end, context.stream) != cudaSuccess)
        return Status::kInternalError;
      ++stats.solves;
      if (status != Status::kOk) {
        context.factor_valid = false;
        (*infos)[item].status = status;
        continue;
      }
      CheckDirectSolutions<<<1, kPcgBlockThreads, 0, context.stream>>>(
          n_, rows_, columns_, transpose_.get(), values + item * value_stride,
          diagonal + (value_stride ? item * n_ : 0), rhs + item * n_, solution + item * n_, 0, true,
          tolerance, info_.get() + item);
      if (cudaGetLastError() != cudaSuccess ||
          cudaMemcpyAsync(host_info_ + item, info_.get() + item, sizeof(SolveInfo),
                          cudaMemcpyDeviceToHost, context.stream) != cudaSuccess ||
          cudaMemcpyAsync(host_solutions_ + lane * n_, solution + item * n_, n_ * sizeof(double),
                          cudaMemcpyDeviceToHost, context.stream) != cudaSuccess)
        return Status::kInternalError;
      solved[lane] = true;
    }
    for (size_t lane = 0; lane < lanes; ++lane) {
      if (!solved[lane])
        continue;
      const size_t item = active[begin + lane];
      auto &context = *contexts_[lane];
      const Status status = context.Wait();
      stats.solve_seconds += context.Seconds(context.solve_begin, context.solve_end);
      if (status != Status::kOk) {
        context.factor_valid = false;
        (*infos)[item].status = status;
      } else {
        (*infos)[item] = host_info_[item];
        std::copy(host_solutions_ + lane * n_, host_solutions_ + (lane + 1) * n_,
                  (*solutions)[item].begin());
      }
    }
  }
  if (timing)
    timing->kernel_sync_seconds += ProfileSecondsSince(start);
  completed = true;
  return Status::kOk;
}
#endif

} // namespace gpu_femm

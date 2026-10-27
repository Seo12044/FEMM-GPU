// Included inside gpu_femm after the common CUDA buffers/reductions.
enum class LinearSolverBackend { kPcg, kCudss };
LinearSolverBackend selected_linear_solver = LinearSolverBackend::kPcg;

struct DirectSolverStats {
  size_t analyses = 0, factorizations = 0, solves = 0, factor_reuses = 0;
  size_t peak_device_estimate_bytes = 0;
  double analysis_seconds = 0, factor_seconds = 0, solve_seconds = 0;
};
DirectSolverStats direct_solver_stats;

#ifdef GPU_FEMM_ENABLE_CUDSS
// Geometry-mask plans may solve on prefetch threads. Keep diagnostic updates
// separate from each plan's numeric state and publish once on every exit.
std::mutex direct_stats_mutex;
struct DirectStatsRecord {
  DirectSolverStats value;
  ~DirectStatsRecord()
  {
    std::lock_guard<std::mutex> lock(direct_stats_mutex);
    direct_solver_stats.analyses += value.analyses;
    direct_solver_stats.factorizations += value.factorizations;
    direct_solver_stats.solves += value.solves;
    direct_solver_stats.factor_reuses += value.factor_reuses;
    direct_solver_stats.peak_device_estimate_bytes = std::max(
        direct_solver_stats.peak_device_estimate_bytes, value.peak_device_estimate_bytes);
    direct_solver_stats.analysis_seconds += value.analysis_seconds;
    direct_solver_stats.factor_seconds += value.factor_seconds;
    direct_solver_stats.solve_seconds += value.solve_seconds;
  }
};

// Acceptance always uses the original full CSR, not just the triangle consumed
// by Cholesky. Scaling avoids squaring tiny/large right-hand sides directly.
__global__ void CheckDirectSolution(int n, const int32_t* rows,
    const int32_t* columns, const int32_t* transpose, const double* values,
    const double* diagonal, const double* rhs, double* x, bool solved,
    double tolerance, SolveInfo* info)
{
  __shared__ double scratch[kPcgBlockThreads];
  const int lane = threadIdx.x;
  double invalid = 0, asymmetric = 0, scale = 0;
  for (int row = lane; row < n; row += blockDim.x) {
    if (!isfinite(rhs[row]) || !isfinite(diagonal[row])
        || (solved && !isfinite(x[row]))) invalid = 2;
    else if (!(diagonal[row] > 0)) invalid = fmax(invalid, 1.0);
    scale = fmax(scale, fabs(rhs[row]));
    for (int i = rows[row]; i < rows[row+1]; ++i) {
      const double value = values[i], other = values[transpose[i]];
      if (!isfinite(value)) invalid = 2;
      // Cancellation near a zero off-diagonal must be judged on the local
      // operator scale. Acceptance still checks the original full matrix.
      const double entry_scale = fmax(fabs(value),
          fmax(fabs(diagonal[row]), fabs(diagonal[columns[i]])));
      if (fabs(value-other) > 1e-13*entry_scale) asymmetric = 1;
    }
  }
  invalid = DeterministicBlockMax(invalid, scratch);
  asymmetric = DeterministicBlockMax(asymmetric, scratch);
  scale = DeterministicBlockMax(scale, scratch);
  if (invalid || asymmetric) {
    if (lane == 0) {
      info->status = invalid == 2 ? Status::kNumericalNonfinite
          : invalid ? Status::kLinearSolveBreakdown : Status::kUnsupportedFeature;
      info->iterations = 0;
      info->residual_l2 = INFINITY;
    }
    return;
  }
  if (!solved || scale == 0) {
    if (scale == 0)
      for (int row = lane; row < n; row += blockDim.x) x[row] = 0;
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
    for (int i = rows[row]; i < rows[row+1]; ++i)
      product += values[i]*(x[columns[i]]/scale);
    const double b = rhs[row]/scale, r = product-b;
    rr += r*r;
    bb += b*b;
  }
  rr = DeterministicBlockSum(rr, scratch);
  bb = DeterministicBlockSum(bb, scratch);
  if (lane == 0) {
    info->iterations = 0; // Direct factorizations are not PCG iterations.
    info->residual_l2 = sqrt(rr/bb);
    info->status = !isfinite(info->residual_l2) ? Status::kNumericalNonfinite
        : info->residual_l2 <= tolerance ? Status::kOk : Status::kLinearSolveNotConverged;
  }
}

class CudssDirectPlan {
 public:
  ~CudssDirectPlan() { Reset(); }
  CudssDirectPlan() = default;
  CudssDirectPlan(const CudssDirectPlan&) = delete;
  CudssDirectPlan& operator=(const CudssDirectPlan&) = delete;

  void Reset()
  {
    if (handle_) cudaDeviceSynchronize();
    if (a_) cudssMatrixDestroy(a_);
    if (b_) cudssMatrixDestroy(b_);
    if (x_) cudssMatrixDestroy(x_);
    if (data_) cudssDataDestroy(handle_, data_);
    if (config_) cudssConfigDestroy(config_);
    if (handle_) cudssDestroy(handle_);
    a_ = b_ = x_ = nullptr;
    handle_ = nullptr; data_ = nullptr; config_ = nullptr;
    analyzed_ = factor_valid_ = false;
    fact_values_ = nullptr;
    transpose_.reset(); info_.reset();
  }

  void InvalidateValues() { factor_valid_ = false; }

  Status Initialize(int n, const std::vector<int32_t>& rows,
      const std::vector<int32_t>& columns, int32_t* device_rows,
      int32_t* device_columns, double* values, double* rhs, double* solution)
  {
    Reset();
    n_ = n; nnz_ = static_cast<int>(columns.size());
    rows_ = device_rows; columns_ = device_columns;
    std::vector<int32_t> transpose(columns.size());
    for (int row = 0; row < n; ++row) {
      for (int j = rows[row]; j < rows[row+1]; ++j) {
        if (j > rows[row] && columns[j-1] >= columns[j]) return Status::kInvalidArgument;
        const int col = columns[j];
        auto it = std::lower_bound(columns.begin()+rows[col], columns.begin()+rows[col+1], row);
        if (it == columns.begin()+rows[col+1] || *it != row) return Status::kUnsupportedFeature;
        transpose[j] = static_cast<int32_t>(it-columns.begin());
      }
    }
    Status status = transpose_.allocate(transpose.size());
    if (status != Status::kOk || (status = info_.allocate(1)) != Status::kOk) return status;
    if ((status = CopyToDevice(transpose_.get(), transpose.data(), transpose.size()*sizeof(int32_t))) != Status::kOk) return status;
    if ((status = Call(cudssCreate(&handle_))) != Status::kOk
        || (status = Call(cudssConfigCreate(&config_))) != Status::kOk
        || (status = Call(cudssDataCreate(handle_, &data_))) != Status::kOk) return status;
    // cuDSS 1.0 deterministic kernels do not support library refinement.
    // Every result still has to pass our full-CSR true-residual gate.
    const int refinement_steps = 0;
    const int deterministic = 1;
    const double pivot_epsilon = 0;
    if ((status = Call(cudssConfigSet(config_, CUDSS_CONFIG_IR_N_STEPS, &refinement_steps, sizeof(refinement_steps)))) != Status::kOk
        || (status = Call(cudssConfigSet(config_, CUDSS_CONFIG_DETERMINISTIC_MODE, &deterministic, sizeof(deterministic)))) != Status::kOk
        || (status = Call(cudssConfigSet(config_, CUDSS_CONFIG_PIVOT_EPSILON, &pivot_epsilon, sizeof(pivot_epsilon)))) != Status::kOk) return status;
    if ((status = Call(cudssMatrixCreateCsr(&a_, n, n, nnz_, rows_, nullptr,
              columns_, values, CUDA_R_32I, CUDA_R_32I, CUDA_R_64F,
              CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO))) != Status::kOk
        || (status = Call(cudssMatrixCreateDn(&b_, n, 1, n, rhs, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR))) != Status::kOk
        || (status = Call(cudssMatrixCreateDn(&x_, n, 1, n, solution, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR))) != Status::kOk) return status;
    return Status::kOk;
  }

  Status SolveBatch(size_t count, size_t value_stride, const double* values,
      const double* diagonal, const double* rhs, double* solution,
      double tolerance, int max_iterations, std::vector<SolveInfo>* infos,
      std::vector<std::vector<double>>* solutions, int* launches,
      GpuBatchSolveTiming* timing)
  {
    if (!handle_ || !count || !(tolerance > 0 && tolerance < 1) || !std::isfinite(tolerance)
        || max_iterations < 0) return Status::kInvalidArgument;
    infos->assign(count, SolveInfo {});
    solutions->assign(count, std::vector<double>(n_, 0));
    DirectStatsRecord record;
    auto& stats = record.value;
    const auto start = ProfileClock::now();
    for (size_t item = 0; item < count; ++item) {
      const double* v = values+item*value_stride;
      const double* d = diagonal+(value_stride ? item*n_ : 0);
      const double* r = rhs+item*n_;
      double* s = solution+item*n_;
      SolveInfo& info = (*infos)[item];
      Status status = Check(v, d, r, s, false, tolerance, &info);
      if (status != Status::kOk) return status;
      if (info.status != Status::kOk || info.residual_l2 == 0) continue;
      if (max_iterations == 0) { info.status = Status::kLinearSolveNotConverged; continue; }
      if (value_stride || v != fact_values_) factor_valid_ = false;
      if ((status = Call(cudssMatrixSetValues(a_, v))) != Status::kOk
          || (status = Call(cudssMatrixSetValues(b_, r))) != Status::kOk
          || (status = Call(cudssMatrixSetValues(x_, s))) != Status::kOk) return status;
      if (!analyzed_) {
        const auto phase = ProfileClock::now();
        if ((status = Execute(CUDSS_PHASE_ANALYSIS)) != Status::kOk) return status;
        analyzed_ = true; ++stats.analyses;
        stats.analysis_seconds += ProfileSecondsSince(phase);
        int64_t estimates[16] = {}; size_t written = 0;
        if ((status = Call(cudssDataGet(handle_, data_, CUDSS_DATA_MEMORY_ESTIMATES,
                  estimates, sizeof(estimates), &written))) != Status::kOk) return status;
        if (written < 2*sizeof(int64_t) || estimates[1] < 0) return Status::kInternalError;
        stats.peak_device_estimate_bytes = std::max(
            stats.peak_device_estimate_bytes, static_cast<size_t>(estimates[1]));
      }
      if (!factor_valid_) {
        const auto phase = ProfileClock::now();
        const int clear = 0;
        if ((status = Call(cudssDataSet(handle_, data_, CUDSS_DATA_INFO, &clear, sizeof(clear)))) != Status::kOk) return status;
        status = Execute(CUDSS_PHASE_FACTORIZATION);
        ++stats.factorizations;
        stats.factor_seconds += ProfileSecondsSince(phase);
        if (status != Status::kOk) { factor_valid_ = false; info.status = status; continue; }
        factor_valid_ = true; fact_values_ = v;
      } else ++stats.factor_reuses;
      const auto phase = ProfileClock::now();
      status = Execute(CUDSS_PHASE_SOLVE);
      ++stats.solves;
      stats.solve_seconds += ProfileSecondsSince(phase);
      if (status != Status::kOk) { factor_valid_ = false; info.status = status; continue; }
      if ((status = Check(v, d, r, s, true, tolerance, &info)) != Status::kOk) return status;
      if (CopyToHost((*solutions)[item].data(), s, n_*sizeof(double)) != Status::kOk) return Status::kInternalError;
    }
    // The legacy launches counter denotes PCG launches, not direct solves.
    if (timing) timing->kernel_sync_seconds += ProfileSecondsSince(start);
    return Status::kOk;
  }

 private:
  static Status Call(cudssStatus_t value)
  {
    if (value == CUDSS_STATUS_SUCCESS) return Status::kOk;
    std::cerr << "cuDSS API error: " << static_cast<int>(value) << '\n';
    return value == CUDSS_STATUS_ALLOC_FAILED ? Status::kGpuAllocationFailed
        : value == CUDSS_STATUS_NOT_SUPPORTED ? Status::kUnsupportedFeature : Status::kInternalError;
  }
  Status Execute(int phase)
  {
    Status status = Call(cudssExecute(handle_, phase, config_, data_, a_, x_, b_));
    if (status != Status::kOk) return status;
    if (cudaDeviceSynchronize() != cudaSuccess) return Status::kInternalError;
    int error = 0; size_t written = 0;
    status = Call(cudssDataGet(handle_, data_, CUDSS_DATA_INFO, &error, sizeof(error), &written));
    if (status != Status::kOk) return status;
    return error ? Status::kLinearSolveBreakdown : Status::kOk;
  }
  Status Check(const double* values, const double* diagonal, const double* rhs,
      double* x, bool solved, double tolerance, SolveInfo* info)
  {
    CheckDirectSolution<<<1, kPcgBlockThreads>>>(n_, rows_, columns_,
        transpose_.get(), values, diagonal, rhs, x, solved, tolerance, info_.get());
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return Status::kInternalError;
    return CopyToHost(info, info_.get(), sizeof(*info));
  }
  int n_ = 0, nnz_ = 0;
  int32_t* rows_ = nullptr; int32_t* columns_ = nullptr;
  cudssHandle_t handle_ = nullptr;
  cudssConfig_t config_ = nullptr;
  cudssData_t data_ = nullptr;
  cudssMatrix_t a_ = nullptr, b_ = nullptr, x_ = nullptr;
  bool analyzed_ = false, factor_valid_ = false;
  const double* fact_values_ = nullptr;
  DeviceBuffer<int32_t> transpose_;
  DeviceBuffer<SolveInfo> info_;
};
#endif

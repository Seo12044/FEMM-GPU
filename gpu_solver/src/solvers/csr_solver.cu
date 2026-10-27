#include "csr_solver.cuh"
#include "../core/profile.h"
#include "pcg_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <utility>

namespace gpu_femm
{

GpuCsrSolver::GpuCsrSolver(PcgPreconditioner mode, bool interleaved, LinearSolverBackend backend)
    : preconditioner_mode_(mode), interleaved_(interleaved), backend_(backend)
{
}

const PreconditionerStats &GpuCsrSolver::preconditioner_stats() const
{
  return block_plan_.stats();
}

size_t GpuCsrSolver::preconditioner_bytes() const { return block_plan_.bytes(); }

bool GpuCsrSolver::interleaved_spmv() const { return spmv_layout_.enabled(); }

Status GpuCsrSolver::Initialize(const Assembly &assembly)
{
#ifdef GPU_FEMM_ENABLE_CUDSS
  direct_plan_.Reset(); // Release wrappers before their borrowed buffers.
#else
  if (backend_ == LinearSolverBackend::kCudss)
    return Status::kUnsupportedFeature;
#endif
  initialized_ = false;
  spmv_layout_.Reset();
  current_preconditioner_ready_ = false;
  block_plan_.Invalidate();
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
    return Status::kGpuUnavailable;
  }
  n_ = static_cast<int>(assembly.free_nodes.size());
  if (n_ <= 0 || assembly.row_offsets.size() != static_cast<size_t>(n_ + 1) ||
      assembly.diagonal.size() != static_cast<size_t>(n_) || assembly.row_offsets.front() != 0 ||
      assembly.row_offsets.back() != static_cast<int32_t>(assembly.values.size()) ||
      assembly.column_indices.size() != assembly.values.size()) {
    return Status::kInvalidArgument;
  }
  for (int row = 0; row < n_; ++row) {
    if (assembly.row_offsets[row] > assembly.row_offsets[row + 1] ||
        !std::isfinite(assembly.diagonal[row])) {
      return Status::kInvalidArgument;
    }
  }
  for (size_t entry = 0; entry < assembly.values.size(); ++entry) {
    if (assembly.column_indices[entry] < 0 || assembly.column_indices[entry] >= n_ ||
        !std::isfinite(assembly.values[entry])) {
      return Status::kInvalidArgument;
    }
  }
  std::vector<double> inverse_diagonal;
  inverse_diagonal_usable_ = BuildJacobiInverseDiagonal(assembly.diagonal, &inverse_diagonal);
  Status status = Status::kOk;
  if ((status = row_offsets_.allocate(assembly.row_offsets.size())) != Status::kOk ||
      (status = column_indices_.allocate(assembly.column_indices.size())) != Status::kOk ||
      (status = values_.allocate(assembly.values.size())) != Status::kOk ||
      (status = diagonal_.allocate(assembly.diagonal.size())) != Status::kOk ||
      (status = inverse_diagonal_.allocate(assembly.diagonal.size())) != Status::kOk ||
      (status = rhs_.allocate(n_)) != Status::kOk ||
      (status = solution_.allocate(n_)) != Status::kOk ||
      (status = residual_.allocate(n_)) != Status::kOk ||
      (status = direction_.allocate(n_)) != Status::kOk ||
      (status = preconditioned_.allocate(n_)) != Status::kOk ||
      (status = matrix_direction_.allocate(n_)) != Status::kOk ||
      (status = info_.allocate(1)) != Status::kOk) {
    return status;
  }
  if ((status = CopyToDevice(row_offsets_.get(), assembly.row_offsets.data(),
                             assembly.row_offsets.size() * sizeof(int32_t))) != Status::kOk ||
      (status = CopyToDevice(column_indices_.get(), assembly.column_indices.data(),
                             assembly.column_indices.size() * sizeof(int32_t))) != Status::kOk ||
      (status = CopyToDevice(values_.get(), assembly.values.data(),
                             assembly.values.size() * sizeof(double))) != Status::kOk ||
      (status = CopyToDevice(diagonal_.get(), assembly.diagonal.data(),
                             assembly.diagonal.size() * sizeof(double))) != Status::kOk ||
      (status = CopyToDevice(inverse_diagonal_.get(), inverse_diagonal.data(),
                             inverse_diagonal.size() * sizeof(double))) != Status::kOk) {
    return status;
  }
  host_row_offsets_ = assembly.row_offsets;
  host_column_indices_ = assembly.column_indices;
  if (backend_ == LinearSolverBackend::kCudss) {
#ifdef GPU_FEMM_ENABLE_CUDSS
    status = direct_plan_.Initialize(n_, host_row_offsets_, host_column_indices_,
                                     row_offsets_.get(), column_indices_.get());
#endif
  } else
    status = spmv_layout_.Initialize(host_row_offsets_, host_column_indices_, interleaved_);
  if (status != Status::kOk)
    return status;
  initialized_ = true;
  return Status::kOk;
}

bool GpuCsrSolver::HasMatchingStructure(const Assembly &assembly) const
{
  return initialized_ && n_ == static_cast<int>(assembly.free_nodes.size()) &&
         assembly.row_offsets == host_row_offsets_ &&
         assembly.column_indices == host_column_indices_;
}

Status GpuCsrSolver::UpdateValues(const Assembly &assembly)
{
  current_preconditioner_ready_ = false;
#ifdef GPU_FEMM_ENABLE_CUDSS
  direct_plan_.InvalidateValues();
#endif
  if (!HasMatchingStructure(assembly) || assembly.values.size() != host_column_indices_.size() ||
      assembly.diagonal.size() != static_cast<size_t>(n_))
    return Status::kInvalidArgument;
  for (double value : assembly.values)
    if (!std::isfinite(value))
      return Status::kInvalidArgument;
  for (double value : assembly.diagonal)
    if (!std::isfinite(value))
      return Status::kInvalidArgument;
  std::vector<double> inverse_diagonal;
  inverse_diagonal_usable_ = BuildJacobiInverseDiagonal(assembly.diagonal, &inverse_diagonal);
  Status status =
      CopyToDevice(values_.get(), assembly.values.data(), assembly.values.size() * sizeof(double));
  if (status != Status::kOk)
    return status;
  if ((status = CopyToDevice(diagonal_.get(), assembly.diagonal.data(),
                             assembly.diagonal.size() * sizeof(double))) != Status::kOk)
    return status;
  return CopyToDevice(inverse_diagonal_.get(), inverse_diagonal.data(),
                      inverse_diagonal.size() * sizeof(double));
}

SolveInfo GpuCsrSolver::Solve(const std::vector<double> &rhs, double relative_tolerance,
                              int max_iterations, std::vector<double> *solution)
{
  SolveInfo result;
  result.status = Status::kInvalidArgument;
  if (!initialized_ || rhs.size() != static_cast<size_t>(n_) || solution == nullptr)
    return result;
  // A single large equation system needs row parallelism too. Reuse the
  // shared-operator batch implementation rather than copying K or keeping
  // a second cooperative launch/validation path. Small fixtures retain the
  // established single-block arithmetic; unsupported devices still fall
  // back inside SolveCurrentOperatorBatch.
  if (backend_ == LinearSolverBackend::kCudss || n_ >= kPcgMinCooperativeRows) {
    std::vector<SolveInfo> infos;
    std::vector<std::vector<double>> solutions;
    const Status status = SolveCurrentOperatorBatch({rhs}, relative_tolerance, max_iterations,
                                                    &infos, &solutions, nullptr);
    if (status != Status::kOk) {
      result.status = status;
      return result;
    }
    *solution = std::move(solutions.front());
    return infos.front();
  }
  Status copy_status = CopyToDevice(rhs_.get(), rhs.data(), rhs.size() * sizeof(double));
  if (copy_status != Status::kOk) {
    result.status = copy_status;
    return result;
  }
  result.status = PreparePreconditioner(1, values_.get(), diagonal_.get(), nullptr);
  if (result.status != Status::kOk)
    return result;
  if (inverse_diagonal_usable_) {
    LaunchDeterministicPcg<true>(
        1, BlockInverse(), BlockOwner(), spmv_layout_.slices(), n_, row_offsets_.get(),
        SpmvColumns(), SpmvValues(values_.get()), diagonal_.get(), inverse_diagonal_.get(),
        rhs_.get(), solution_.get(), residual_.get(), direction_.get(), preconditioned_.get(),
        matrix_direction_.get(), relative_tolerance, max_iterations, info_.get(),
        SpmvStride(host_column_indices_.size()));
  } else {
    LaunchDeterministicPcg<false>(
        1, BlockInverse(), BlockOwner(), spmv_layout_.slices(), n_, row_offsets_.get(),
        SpmvColumns(), SpmvValues(values_.get()), diagonal_.get(), inverse_diagonal_.get(),
        rhs_.get(), solution_.get(), residual_.get(), direction_.get(), preconditioned_.get(),
        matrix_direction_.get(), relative_tolerance, max_iterations, info_.get(),
        SpmvStride(host_column_indices_.size()));
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

Status GpuCsrSolver::SolveBatch(const std::vector<Assembly> &assemblies,
                                const std::vector<std::vector<double>> &rhs_values,
                                double relative_tolerance, int max_iterations,
                                std::vector<SolveInfo> *infos,
                                std::vector<std::vector<double>> *solutions, int *launches,
                                GpuBatchSolveTiming *timing)
{
  if (infos == nullptr || solutions == nullptr || assemblies.empty() ||
      assemblies.size() != rhs_values.size())
    return Status::kInvalidArgument;
  const size_t count = assemblies.size();
  for (size_t item = 0; item < count; ++item) {
    if (!HasMatchingStructure(assemblies[item]) ||
        rhs_values[item].size() != static_cast<size_t>(n_))
      return Status::kInvalidArgument;
  }
  const size_t nnz = host_column_indices_.size();
  const ProfileClock::time_point upload_start = ProfileClock::now();
  const Status capacity_status = EnsureBatchCapacity(count, nnz);
  if (capacity_status != Status::kOk)
    return capacity_status;
  std::vector<double> inverse_diagonal;
  bool batch_inverse_diagonal_usable = true;
  for (size_t item = 0; item < count; ++item) {
    batch_inverse_diagonal_usable =
        BuildJacobiInverseDiagonal(assemblies[item].diagonal, &inverse_diagonal) &&
        batch_inverse_diagonal_usable;
    Status status = CopyToDevice(batch_values_.get() + item * nnz, assemblies[item].values.data(),
                                 nnz * sizeof(double));
    if (status != Status::kOk)
      return status;
    if ((status = CopyToDevice(batch_diagonal_.get() + item * n_, assemblies[item].diagonal.data(),
                               n_ * sizeof(double))) != Status::kOk ||
        (status = CopyToDevice(batch_inverse_diagonal_.get() + item * n_, inverse_diagonal.data(),
                               n_ * sizeof(double))) != Status::kOk ||
        (status = CopyToDevice(batch_rhs_.get() + item * n_, rhs_values[item].data(),
                               n_ * sizeof(double))) != Status::kOk)
      return status;
  }
  if (timing != nullptr)
    timing->upload_seconds += ProfileSecondsSince(upload_start);
  if (backend_ == LinearSolverBackend::kCudss)
    return SolveDirectBatch(count, nnz, batch_values_.get(), batch_diagonal_.get(),
                            batch_rhs_.get(), batch_solution_.get(), relative_tolerance,
                            max_iterations, infos, solutions, launches, timing);
  const Status preconditioner_status =
      PreparePreconditioner(count, batch_values_.get(), batch_diagonal_.get(), timing);
  if (preconditioner_status != Status::kOk)
    return preconditioner_status;
  const ProfileClock::time_point kernel_start = ProfileClock::now();
  const CooperativeLaunchResult cooperative_result = TryLaunchCooperativeBatch(
      count, nnz, relative_tolerance, max_iterations, batch_inverse_diagonal_usable, false);
  if (cooperative_result == CooperativeLaunchResult::kFailed)
    return Status::kInternalError;
  const bool launched_cooperatively = cooperative_result == CooperativeLaunchResult::kCompleted;
  if (!launched_cooperatively && batch_inverse_diagonal_usable) {
    LaunchDeterministicPcg<true>(
        static_cast<unsigned>(count), BlockInverse(), BlockOwner(), spmv_layout_.slices(), n_,
        row_offsets_.get(), SpmvColumns(), SpmvValues(batch_values_.get()), batch_diagonal_.get(),
        batch_inverse_diagonal_.get(), batch_rhs_.get(), batch_solution_.get(),
        batch_residual_.get(), batch_direction_.get(), batch_preconditioned_.get(),
        batch_matrix_direction_.get(), relative_tolerance, max_iterations, batch_info_.get(),
        SpmvStride(nnz));
  } else if (!launched_cooperatively) {
    LaunchDeterministicPcg<false>(
        static_cast<unsigned>(count), BlockInverse(), BlockOwner(), spmv_layout_.slices(), n_,
        row_offsets_.get(), SpmvColumns(), SpmvValues(batch_values_.get()), batch_diagonal_.get(),
        batch_inverse_diagonal_.get(), batch_rhs_.get(), batch_solution_.get(),
        batch_residual_.get(), batch_direction_.get(), batch_preconditioned_.get(),
        batch_matrix_direction_.get(), relative_tolerance, max_iterations, batch_info_.get(),
        SpmvStride(nnz));
  }
  if (launches != nullptr)
    ++*launches;
  if (!launched_cooperatively &&
      (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess))
    return Status::kInternalError;
  if (timing != nullptr)
    timing->kernel_sync_seconds += ProfileSecondsSince(kernel_start);
  const ProfileClock::time_point download_start = ProfileClock::now();
  infos->assign(count, SolveInfo{});
  solutions->assign(count, std::vector<double>(static_cast<size_t>(n_)));
  for (size_t item = 0; item < count; ++item) {
    Status status = CopyToHost(&(*infos)[item], batch_info_.get() + item, sizeof(SolveInfo));
    if (status != Status::kOk)
      return status;
    if ((status = CopyToHost((*solutions)[item].data(), batch_solution_.get() + item * n_,
                             n_ * sizeof(double))) != Status::kOk)
      return status;
  }
  if (timing != nullptr)
    timing->download_seconds += ProfileSecondsSince(download_start);
  return Status::kOk;
}

Status GpuCsrSolver::SolveCurrentOperatorBatch(const std::vector<std::vector<double>> &rhs_values,
                                               double relative_tolerance, int max_iterations,
                                               std::vector<SolveInfo> *infos,
                                               std::vector<std::vector<double>> *solutions,
                                               int *launches, SharedOperatorBatchEvidence *evidence,
                                               GpuBatchSolveTiming *timing)
{
  if (!initialized_ || infos == nullptr || solutions == nullptr || rhs_values.empty())
    return Status::kInvalidArgument;
  const size_t count = rhs_values.size();
  for (const std::vector<double> &rhs : rhs_values)
    if (rhs.size() != static_cast<size_t>(n_))
      return Status::kInvalidArgument;
  const ProfileClock::time_point upload_start = ProfileClock::now();
  const Status capacity_status = EnsureSharedBatchCapacity(count);
  if (capacity_status != Status::kOk)
    return capacity_status;
  for (size_t item = 0; item < count; ++item) {
    const Status status = CopyToDevice(batch_rhs_.get() + item * n_, rhs_values[item].data(),
                                       static_cast<size_t>(n_) * sizeof(double));
    if (status != Status::kOk)
      return status;
  }
  if (timing != nullptr)
    timing->upload_seconds += ProfileSecondsSince(upload_start);
  if (backend_ == LinearSolverBackend::kCudss) {
    if (evidence != nullptr) {
      evidence->physical_operator_count = evidence->operator_upload_count = 1;
      evidence->rhs_count = count;
      evidence->operator_bytes = values_.count() * sizeof(double);
      evidence->rhs_workspace_bytes = count * static_cast<size_t>(n_) * 6 * sizeof(double);
    }
    return SolveDirectBatch(count, 0, values_.get(), diagonal_.get(), batch_rhs_.get(),
                            batch_solution_.get(), relative_tolerance, max_iterations, infos,
                            solutions, launches, timing);
  }
  const Status preconditioner_status =
      PreparePreconditioner(1, values_.get(), diagonal_.get(), timing);
  if (preconditioner_status != Status::kOk)
    return preconditioner_status;
  if (evidence != nullptr) {
    evidence->physical_operator_count = 1;
    evidence->operator_upload_count = 1;
    evidence->rhs_count = count;
    evidence->operator_bytes =
        values_.count() * sizeof(double) + diagonal_.count() * sizeof(double) +
        inverse_diagonal_.count() * sizeof(double) + spmv_layout_.numeric_bytes();
    evidence->rhs_workspace_bytes =
        count * static_cast<size_t>(n_) * 6 * sizeof(double) + count * sizeof(SolveInfo);
  }
  const ProfileClock::time_point kernel_start = ProfileClock::now();
  const CooperativeLaunchResult cooperative_result =
      TryLaunchCooperativeBatch(count, host_column_indices_.size(), relative_tolerance,
                                max_iterations, inverse_diagonal_usable_, true);
  if (cooperative_result == CooperativeLaunchResult::kFailed)
    return Status::kInternalError;
  const bool launched_cooperatively = cooperative_result == CooperativeLaunchResult::kCompleted;
  if (!launched_cooperatively && inverse_diagonal_usable_) {
    LaunchDeterministicPcg<true>(
        static_cast<unsigned>(count), BlockInverse(), BlockOwner(), spmv_layout_.slices(), n_,
        row_offsets_.get(), SpmvColumns(), SpmvValues(values_.get()), diagonal_.get(),
        inverse_diagonal_.get(), batch_rhs_.get(), batch_solution_.get(), batch_residual_.get(),
        batch_direction_.get(), batch_preconditioned_.get(), batch_matrix_direction_.get(),
        relative_tolerance, max_iterations, batch_info_.get(), 0);
  } else if (!launched_cooperatively) {
    LaunchDeterministicPcg<false>(
        static_cast<unsigned>(count), BlockInverse(), BlockOwner(), spmv_layout_.slices(), n_,
        row_offsets_.get(), SpmvColumns(), SpmvValues(values_.get()), diagonal_.get(),
        inverse_diagonal_.get(), batch_rhs_.get(), batch_solution_.get(), batch_residual_.get(),
        batch_direction_.get(), batch_preconditioned_.get(), batch_matrix_direction_.get(),
        relative_tolerance, max_iterations, batch_info_.get(), 0);
  }
  if (launches != nullptr)
    ++*launches;
  if (!launched_cooperatively &&
      (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess))
    return Status::kInternalError;
  if (timing != nullptr)
    timing->kernel_sync_seconds += ProfileSecondsSince(kernel_start);
  const ProfileClock::time_point download_start = ProfileClock::now();
  infos->assign(count, SolveInfo{});
  solutions->assign(count, std::vector<double>(static_cast<size_t>(n_)));
  for (size_t item = 0; item < count; ++item) {
    Status status = CopyToHost(&(*infos)[item], batch_info_.get() + item, sizeof(SolveInfo));
    if (status != Status::kOk)
      return status;
    status = CopyToHost((*solutions)[item].data(), batch_solution_.get() + item * n_,
                        static_cast<size_t>(n_) * sizeof(double));
    if (status != Status::kOk)
      return status;
  }
  if (timing != nullptr)
    timing->download_seconds += ProfileSecondsSince(download_start);
  return Status::kOk;
}

Status GpuCsrSolver::PrepareDeviceBatch(size_t count)
{
  if (!initialized_ || count == 0)
    return Status::kInvalidArgument;
  return EnsureBatchCapacity(count, host_column_indices_.size());
}

int GpuCsrSolver::device_dimension() const { return n_; }

size_t GpuCsrSolver::device_nnz() const { return host_column_indices_.size(); }

double *GpuCsrSolver::device_batch_values() { return batch_values_.get(); }

double *GpuCsrSolver::device_batch_diagonal() { return batch_diagonal_.get(); }

double *GpuCsrSolver::device_batch_inverse_diagonal() { return batch_inverse_diagonal_.get(); }

double *GpuCsrSolver::device_batch_rhs() { return batch_rhs_.get(); }

void GpuCsrSolver::set_device_batch_inverse_diagonal_usable(bool value)
{
  batch_inverse_diagonal_usable_ = value;
}

Status GpuCsrSolver::CopyPreparedDeviceAssembly(size_t item, std::vector<double> *values,
                                                std::vector<double> *diagonal,
                                                std::vector<double> *rhs) const
{
  if (!initialized_ || values == nullptr || diagonal == nullptr || rhs == nullptr ||
      item >= batch_capacity_)
    return Status::kInvalidArgument;
  const size_t nnz = host_column_indices_.size();
  values->resize(nnz);
  diagonal->resize(static_cast<size_t>(n_));
  rhs->resize(static_cast<size_t>(n_));
  Status status =
      CopyToHost(values->data(), batch_values_.get() + item * nnz, nnz * sizeof(double));
  if (status != Status::kOk)
    return status;
  status = CopyToHost(diagonal->data(), batch_diagonal_.get() + item * n_,
                      static_cast<size_t>(n_) * sizeof(double));
  if (status != Status::kOk)
    return status;
  return CopyToHost(rhs->data(), batch_rhs_.get() + item * n_,
                    static_cast<size_t>(n_) * sizeof(double));
}

Status GpuCsrSolver::SolvePreparedDeviceBatch(size_t count, double relative_tolerance,
                                              int max_iterations, std::vector<SolveInfo> *infos,
                                              std::vector<std::vector<double>> *solutions,
                                              int *launches, GpuBatchSolveTiming *timing)
{
  if (!initialized_ || infos == nullptr || solutions == nullptr || count == 0 ||
      count > batch_capacity_)
    return Status::kInvalidArgument;
  const size_t nnz = host_column_indices_.size();
  if (backend_ == LinearSolverBackend::kCudss)
    return SolveDirectBatch(count, nnz, batch_values_.get(), batch_diagonal_.get(),
                            batch_rhs_.get(), batch_solution_.get(), relative_tolerance,
                            max_iterations, infos, solutions, launches, timing);
  const Status preconditioner_status =
      PreparePreconditioner(count, batch_values_.get(), batch_diagonal_.get(), timing);
  if (preconditioner_status != Status::kOk)
    return preconditioner_status;
  const ProfileClock::time_point kernel_start = ProfileClock::now();
  const CooperativeLaunchResult cooperative_result = TryLaunchCooperativeBatch(
      count, nnz, relative_tolerance, max_iterations, batch_inverse_diagonal_usable_, false);
  if (cooperative_result == CooperativeLaunchResult::kFailed)
    return Status::kInternalError;
  const bool launched_cooperatively = cooperative_result == CooperativeLaunchResult::kCompleted;
  // Device assembly has already produced an inverse diagonal.  Its kernel
  // marks invalid entries before this point, so the regular fallback only
  // selects whether Jacobi multiplication or validated division is used.
  if (!launched_cooperatively && batch_inverse_diagonal_usable_) {
    LaunchDeterministicPcg<true>(
        static_cast<unsigned>(count), BlockInverse(), BlockOwner(), spmv_layout_.slices(), n_,
        row_offsets_.get(), SpmvColumns(), SpmvValues(batch_values_.get()), batch_diagonal_.get(),
        batch_inverse_diagonal_.get(), batch_rhs_.get(), batch_solution_.get(),
        batch_residual_.get(), batch_direction_.get(), batch_preconditioned_.get(),
        batch_matrix_direction_.get(), relative_tolerance, max_iterations, batch_info_.get(),
        SpmvStride(nnz));
  } else if (!launched_cooperatively) {
    LaunchDeterministicPcg<false>(
        static_cast<unsigned>(count), BlockInverse(), BlockOwner(), spmv_layout_.slices(), n_,
        row_offsets_.get(), SpmvColumns(), SpmvValues(batch_values_.get()), batch_diagonal_.get(),
        batch_inverse_diagonal_.get(), batch_rhs_.get(), batch_solution_.get(),
        batch_residual_.get(), batch_direction_.get(), batch_preconditioned_.get(),
        batch_matrix_direction_.get(), relative_tolerance, max_iterations, batch_info_.get(),
        SpmvStride(nnz));
  }
  if (launches != nullptr)
    ++*launches;
  if (!launched_cooperatively &&
      (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess))
    return Status::kInternalError;
  if (timing != nullptr)
    timing->kernel_sync_seconds += ProfileSecondsSince(kernel_start);
  const ProfileClock::time_point download_start = ProfileClock::now();
  infos->assign(count, SolveInfo{});
  solutions->assign(count, std::vector<double>(static_cast<size_t>(n_)));
  for (size_t item = 0; item < count; ++item) {
    Status status = CopyToHost(&(*infos)[item], batch_info_.get() + item, sizeof(SolveInfo));
    if (status != Status::kOk)
      return status;
    if ((status = CopyToHost((*solutions)[item].data(), batch_solution_.get() + item * n_,
                             n_ * sizeof(double))) != Status::kOk)
      return status;
  }
  if (timing != nullptr)
    timing->download_seconds += ProfileSecondsSince(download_start);
  return Status::kOk;
}

Status GpuCsrSolver::SolveDirectBatch(size_t count, size_t stride, const double *values,
                                      const double *diagonal, const double *rhs, double *solution,
                                      double tolerance, int max_iterations,
                                      std::vector<SolveInfo> *infos,
                                      std::vector<std::vector<double>> *solutions, int *launches,
                                      GpuBatchSolveTiming *timing)
{
#ifdef GPU_FEMM_ENABLE_CUDSS
  return direct_plan_.SolveBatch(count, stride, values, diagonal, rhs, solution, tolerance,
                                 max_iterations, infos, solutions, launches, timing);
#else
  return Status::kUnsupportedFeature;
#endif
}

const int32_t *GpuCsrSolver::SpmvColumns() const
{
  return spmv_layout_.enabled() ? spmv_layout_.columns() : column_indices_.get();
}

const double *GpuCsrSolver::SpmvValues(const double *csr) const
{
  return spmv_layout_.enabled() ? spmv_layout_.values() : csr;
}

int GpuCsrSolver::SpmvStride(size_t nnz) const
{
  return spmv_layout_.enabled() ? spmv_layout_.slots() : static_cast<int>(nnz);
}

const double *GpuCsrSolver::BlockInverse() const
{
  return preconditioner_mode_ == PcgPreconditioner::kBlockJacobi ? block_plan_.inverse() : nullptr;
}

const int *GpuCsrSolver::BlockOwner() const
{
  return preconditioner_mode_ == PcgPreconditioner::kBlockJacobi ? block_plan_.owner() : nullptr;
}

Status GpuCsrSolver::PreparePreconditioner(size_t count, const double *values,
                                           const double *diagonal, GpuBatchSolveTiming *timing)
{
  const auto pack_start = ProfileClock::now();
  const Status pack_status = spmv_layout_.Pack(count, values);
  if (timing != nullptr) {
    timing->kernel_sync_seconds += ProfileSecondsSince(pack_start);
    if (spmv_layout_.enabled())
      timing->interleaved_spmv_solves += count;
  }
  if (pack_status != Status::kOk)
    return pack_status;
  if (preconditioner_mode_ == PcgPreconditioner::kJacobi)
    return Status::kOk;
  const bool current_operator =
      count == 1 && values == values_.get() && diagonal == diagonal_.get();
  // These private buffers cannot change except through Initialize/UpdateValues.
  // Device-assembled batches never use this shortcut.
  if (current_operator && current_preconditioner_ready_) {
    block_plan_.ReuseUnmodifiedOperator();
    if (timing != nullptr)
      ++timing->preconditioner.reuses;
    return Status::kOk;
  }
  current_preconditioner_ready_ = false;
  const PreconditionerStats before = block_plan_.stats();
  const Status status =
      block_plan_.Prepare(n_, host_column_indices_.size(), count, row_offsets_.get(),
                          column_indices_.get(), values, diagonal);
  current_preconditioner_ready_ = current_operator && status == Status::kOk;
  if (timing != nullptr) {
    const auto &after = block_plan_.stats();
    timing->preconditioner.builds += after.builds - before.builds;
    timing->preconditioner.reuses += after.reuses - before.reuses;
    timing->preconditioner.fallback_blocks += after.fallback_blocks - before.fallback_blocks;
    timing->preconditioner.setup_seconds += after.setup_seconds - before.setup_seconds;
  }
  return status;
}

Status GpuCsrSolver::EnsureSharedBatchCapacity(size_t count)
{
  // A stale legacy batch may own count copies of K. Drop those copies so
  // the physical operator count is exactly one on this path. A later
  // SolveBatch sees batch_nnz_=0 and allocates its legacy buffers again.
  batch_values_.reset();
  batch_diagonal_.reset();
  batch_inverse_diagonal_.reset();
  Status status = Status::kOk;
  if (batch_capacity_ < count || batch_dimension_ != n_) {
    if ((status = batch_rhs_.allocate(count * n_)) != Status::kOk ||
        (status = batch_solution_.allocate(count * n_)) != Status::kOk ||
        (status = batch_residual_.allocate(count * n_)) != Status::kOk ||
        (status = batch_direction_.allocate(count * n_)) != Status::kOk ||
        (status = batch_preconditioned_.allocate(count * n_)) != Status::kOk ||
        (status = batch_matrix_direction_.allocate(count * n_)) != Status::kOk ||
        (status = batch_info_.allocate(count)) != Status::kOk ||
        (status = batch_cooperative_partials_.allocate(
             count * 2 * kPcgMaxCooperativeShardsPerItem)) != Status::kOk ||
        (status = batch_cooperative_scalars_.allocate(count * 3)) != Status::kOk ||
        (status = batch_cooperative_controls_.allocate(count * 4 + 1)) != Status::kOk)
      return status;
    batch_capacity_ = count;
    batch_dimension_ = n_;
  }
  batch_nnz_ = 0;
  return Status::kOk;
}

Status GpuCsrSolver::EnsureBatchCapacity(size_t count, size_t nnz)
{
  if (batch_capacity_ >= count && batch_dimension_ == n_ && batch_nnz_ == nnz &&
      batch_values_.count() >= count * nnz &&
      batch_diagonal_.count() >= count * static_cast<size_t>(n_) &&
      batch_inverse_diagonal_.count() >= count * static_cast<size_t>(n_))
    return Status::kOk;
  Status status = Status::kOk;
  if ((status = batch_values_.allocate(count * nnz)) != Status::kOk ||
      (status = batch_diagonal_.allocate(count * n_)) != Status::kOk ||
      (status = batch_inverse_diagonal_.allocate(count * n_)) != Status::kOk ||
      (status = batch_rhs_.allocate(count * n_)) != Status::kOk ||
      (status = batch_solution_.allocate(count * n_)) != Status::kOk ||
      (status = batch_residual_.allocate(count * n_)) != Status::kOk ||
      (status = batch_direction_.allocate(count * n_)) != Status::kOk ||
      (status = batch_preconditioned_.allocate(count * n_)) != Status::kOk ||
      (status = batch_matrix_direction_.allocate(count * n_)) != Status::kOk ||
      (status = batch_info_.allocate(count)) != Status::kOk ||
      (status = batch_cooperative_partials_.allocate(
           count * 2 * kPcgMaxCooperativeShardsPerItem)) != Status::kOk ||
      (status = batch_cooperative_scalars_.allocate(count * 3)) != Status::kOk ||
      (status = batch_cooperative_controls_.allocate(count * 4 + 1)) != Status::kOk)
    return status;
  batch_capacity_ = count;
  batch_dimension_ = n_;
  batch_nnz_ = nnz;
  return Status::kOk;
}

GpuCsrSolver::CooperativeLaunchResult
GpuCsrSolver::TryLaunchCooperativeBatch(size_t count, size_t nnz, double relative_tolerance,
                                        int max_iterations, bool use_inverse_diagonal,
                                        bool shared_operator)
{
  // Keep large batches and tiny systems on the established one-block path.
  // Small batches use a device/row-count bounded partition, independent of
  // model type, rotor geometry, currents, or the calling application.
  if (count == 0 || count > 4 || n_ < kPcgMinCooperativeRows)
    return CooperativeLaunchResult::kNotSupported;
  int device = 0;
  int cooperative_supported = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&cooperative_supported, cudaDevAttrCooperativeLaunch, device) !=
          cudaSuccess ||
      cooperative_supported == 0 || cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    cudaGetLastError();
    return CooperativeLaunchResult::kNotSupported;
  }
  int active_blocks_per_sm = 0;
  const bool block_preconditioner = preconditioner_mode_ == PcgPreconditioner::kBlockJacobi;
  const void *kernel = CooperativePcgKernel(use_inverse_diagonal, block_preconditioner);
  const cudaError_t occupancy_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, kernel, kPcgBlockThreads, 0);
  int shard_count = PcgCooperativeShardCount(n_, count, properties.multiProcessorCount);
  const size_t required_blocks = count * shard_count;
  if (shard_count == 0 || occupancy_status != cudaSuccess || active_blocks_per_sm <= 0 ||
      required_blocks > static_cast<size_t>(active_blocks_per_sm) *
                            static_cast<size_t>(properties.multiProcessorCount)) {
    cudaGetLastError();
    return CooperativeLaunchResult::kNotSupported;
  }
  int item_count = static_cast<int>(count);
  int values_per_item = shared_operator ? 0 : SpmvStride(nnz);
  const int32_t *row_offsets = row_offsets_.get();
  const int32_t *column_indices = SpmvColumns();
  const double *values = SpmvValues(shared_operator ? values_.get() : batch_values_.get());
  const int32_t *slices = spmv_layout_.slices();
  const double *diagonal = shared_operator ? diagonal_.get() : batch_diagonal_.get();
  const double *inverse_diagonal =
      shared_operator ? inverse_diagonal_.get() : batch_inverse_diagonal_.get();
  const double *rhs = batch_rhs_.get();
  double *solution = batch_solution_.get();
  double *residual = batch_residual_.get();
  double *direction = batch_direction_.get();
  double *preconditioned = batch_preconditioned_.get();
  double *matrix_direction = batch_matrix_direction_.get();
  SolveInfo *info = batch_info_.get();
  double *partials = batch_cooperative_partials_.get();
  double *scalars = batch_cooperative_scalars_.get();
  int *controls = batch_cooperative_controls_.get();
  const double *block_inverse = BlockInverse();
  const int *block_owner = BlockOwner();
  void *arguments[] = {&n_,
                       &row_offsets,
                       &column_indices,
                       &values,
                       &diagonal,
                       &inverse_diagonal,
                       &rhs,
                       &solution,
                       &residual,
                       &direction,
                       &preconditioned,
                       &matrix_direction,
                       &relative_tolerance,
                       &max_iterations,
                       &info,
                       &values_per_item,
                       &item_count,
                       &shard_count,
                       &partials,
                       &scalars,
                       &controls,
                       &block_inverse,
                       &block_owner,
                       &slices};
  const dim3 grid(static_cast<unsigned int>(required_blocks));
  const dim3 block(kPcgBlockThreads);
  const cudaError_t launch_status = cudaLaunchCooperativeKernel(kernel, grid, block, arguments);
  if (launch_status != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
    cudaGetLastError();
    return CooperativeLaunchResult::kFailed;
  }
  return CooperativeLaunchResult::kCompleted;
}

} // namespace gpu_femm

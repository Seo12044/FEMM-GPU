#pragma once

#include "../physics/linear_model.h"
#include "../core/cuda_buffer.cuh"
#include "options.h"
#include "block_jacobi_preconditioner.cuh"
#include "pcg_spmv_layout.cuh"
#include "cudss_direct_solver.cuh"

namespace gpu_femm
{

class GpuCsrSolver
{
public:
  explicit GpuCsrSolver(PcgPreconditioner mode = selected_preconditioner, bool interleaved = true,
                        LinearSolverBackend backend = selected_linear_solver);
  const PreconditionerStats &preconditioner_stats() const;
  size_t preconditioner_bytes() const;
  bool interleaved_spmv() const;

  Status Initialize(const Assembly &assembly);

  bool HasMatchingStructure(const Assembly &assembly) const;

  Status UpdateValues(const Assembly &assembly);

  SolveInfo Solve(const std::vector<double> &rhs, double relative_tolerance, int max_iterations,
                  std::vector<double> *solution);

  Status SolveBatch(const std::vector<Assembly> &assemblies,
                    const std::vector<std::vector<double>> &rhs_values, double relative_tolerance,
                    int max_iterations, std::vector<SolveInfo> *infos,
                    std::vector<std::vector<double>> *solutions, int *launches,
                    GpuBatchSolveTiming *timing = nullptr);

  // Solve several right-hand sides against the numeric operator currently in
  // values_/diagonal_/inverse_diagonal_. Unlike SolveBatch, no CSR numeric
  // array is allocated or uploaded per RHS.
  Status SolveCurrentOperatorBatch(const std::vector<std::vector<double>> &rhs_values,
                                   double relative_tolerance, int max_iterations,
                                   std::vector<SolveInfo> *infos,
                                   std::vector<std::vector<double>> *solutions, int *launches,
                                   SharedOperatorBatchEvidence *evidence = nullptr,
                                   GpuBatchSolveTiming *timing = nullptr);

  // Device nonlinear assembly writes these numeric buffers directly.  The
  // immutable CSR graph remains owned here, while this narrow API avoids a
  // host round trip for values, diagonal and RHS between Newton assembly and
  // PCG. Single-RHS callers may share the same cooperative linear solver,
  // but still assemble and upload their operator through UpdateValues.
  Status PrepareDeviceBatch(size_t count);

  int device_dimension() const;
  size_t device_nnz() const;
  double *device_batch_values();
  double *device_batch_diagonal();
  double *device_batch_inverse_diagonal();
  double *device_batch_rhs();
  void set_device_batch_inverse_diagonal_usable(bool value);

  Status CopyPreparedDeviceAssembly(size_t item, std::vector<double> *values,
                                    std::vector<double> *diagonal, std::vector<double> *rhs) const;

  Status SolvePreparedDeviceBatch(size_t count, double relative_tolerance, int max_iterations,
                                  std::vector<SolveInfo> *infos,
                                  std::vector<std::vector<double>> *solutions, int *launches,
                                  GpuBatchSolveTiming *timing = nullptr);

private:
  Status SolveDirectBatch(size_t count, size_t stride, const double *values, const double *diagonal,
                          const double *rhs, double *solution, double tolerance, int max_iterations,
                          std::vector<SolveInfo> *infos,
                          std::vector<std::vector<double>> *solutions, int *launches,
                          GpuBatchSolveTiming *timing);
  const int32_t *SpmvColumns() const;
  const double *SpmvValues(const double *csr) const;
  int SpmvStride(size_t nnz) const;
  const double *BlockInverse() const;
  const int *BlockOwner() const;

  Status PreparePreconditioner(size_t count, const double *values, const double *diagonal,
                               GpuBatchSolveTiming *timing);

  Status EnsureSharedBatchCapacity(size_t count);

  Status EnsureBatchCapacity(size_t count, size_t nnz);
  enum class CooperativeLaunchResult { kNotSupported, kCompleted, kFailed };

  CooperativeLaunchResult TryLaunchCooperativeBatch(size_t count, size_t nnz,
                                                    double relative_tolerance, int max_iterations,
                                                    bool use_inverse_diagonal,
                                                    bool shared_operator);

  int n_ = 0;
  PcgPreconditioner preconditioner_mode_;
  bool interleaved_;
  LinearSolverBackend backend_;
  PcgSpmvLayout spmv_layout_;
  BlockJacobiPlan block_plan_;
  bool current_preconditioner_ready_ = false;
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
  DeviceBuffer<double> batch_values_, batch_diagonal_, batch_inverse_diagonal_, batch_rhs_,
      batch_solution_, batch_residual_, batch_direction_, batch_preconditioned_,
      batch_matrix_direction_;
  DeviceBuffer<SolveInfo> batch_info_;
  DeviceBuffer<double> batch_cooperative_partials_, batch_cooperative_scalars_;
  DeviceBuffer<int> batch_cooperative_controls_;
  bool batch_inverse_diagonal_usable_ = true;
  int batch_dimension_ = 0;
  size_t batch_nnz_ = 0;
#ifdef GPU_FEMM_ENABLE_CUDSS
  CudssDirectPlan direct_plan_; // Destroy before the borrowed device buffers.
#endif
};

} // namespace gpu_femm

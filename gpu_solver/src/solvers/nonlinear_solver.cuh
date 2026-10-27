#pragma once

#include "physics/nonlinear_assembly.cuh"
#include "physics/nonlinear_model.h"
#include "solvers/csr_solver.cuh"
#include "solvers/options.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace gpu_femm
{

struct NonlinearBatchTiming {
  size_t interleaved_spmv_solves = 0;
  PreconditionerStats preconditioner;
  double host_assembly_seconds = 0.0;
  double device_assembly_seconds = 0.0;
  double gpu_upload_seconds = 0.0;
  double gpu_kernel_sync_seconds = 0.0;
  double gpu_download_seconds = 0.0;
  double state_update_seconds = 0.0;
  double finalize_seconds = 0.0;
  double recovery_seconds = 0.0;
};

enum class NonlinearAssemblyMode { kAuto, kHost, kDevice };

class NonlinearP1Solver
{
public:
  Status Initialize(NonlinearModel model);

  size_t csr_symbolic_reuse_count() const;

  NonlinearSolveResult Solve(double current_a, const NonlinearOptions &options = {},
                             const std::vector<double> *warm_start = nullptr);

  NonlinearSolveResult Solve(const std::vector<double> &circuit_currents_a,
                             NonlinearOptions options = {},
                             const std::vector<double> *warm_start = nullptr);

  NonlinearSolveResult SolveSafeguarded(const std::vector<double> &currents,
                                        const NonlinearOptions &options);

  // Independent nonlinear states share one immutable symbolic CSR and device
  // assembly plan. Every active state is dispatched in one batched assembly
  // and PCG launch; the established host assembler remains the auto fallback.
  std::vector<NonlinearSolveResult>
  SolveBatch(const std::vector<std::vector<double>> &currents, NonlinearOptions options,
             int *batched_pcg_launches = nullptr, NonlinearBatchTiming *timing = nullptr,
             NonlinearAssemblyMode requested_mode = NonlinearAssemblyMode::kAuto);

  // Single-sample CLI entry; Solve() supplies the host reference and warm start.
  NonlinearSolveResult SolveAccelerated(const std::vector<double> &currents,
                                        const NonlinearOptions &options,
                                        NonlinearAssemblyMode mode);

  // Reassemble the consistent Newton tangent at the converged operating
  // point, then solve all command-current directions against that one
  // operator.  No permanent-magnet or fixed-boundary load belongs in a field
  // derivative, so every RHS is formed solely from the circuit source vectors.
  std::vector<TangentSolveResult>
  SolveTangentMultiRhs(const NonlinearSolveResult &base,
                       const std::vector<std::pair<std::string, std::vector<double>>> &directions,
                       const NonlinearOptions &options, int *batched_pcg_launches = nullptr);

private:
  GpuCsrSolver csr_solver_;
  bool csr_initialized_ = false;
  size_t csr_symbolic_reuse_count_ = 0;
  NonlinearSolveResult FinalizeResult(NonlinearSolveResult *result) const;

  bool initialized_ = false;
  NonlinearModel model_;
  bool has_nonlinear_materials_ = false;
  std::unique_ptr<DeviceNonlinearAssemblyPlan> device_assembly_;
};

} // namespace gpu_femm

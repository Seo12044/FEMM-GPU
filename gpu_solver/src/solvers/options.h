#pragma once

#include "../core/types.h"

namespace gpu_femm
{

enum class PcgPreconditioner { kJacobi, kBlockJacobi };
extern PcgPreconditioner selected_preconditioner;
constexpr int kJacobiBlockSize = 8;
constexpr int kJacobiBlockStorage = kJacobiBlockSize * kJacobiBlockSize;

struct PreconditionerStats {
  size_t builds = 0;
  size_t reuses = 0;
  size_t fallback_blocks = 0;
  double setup_seconds = 0.0;
};

void AccumulatePreconditionerStats(PreconditionerStats &target, const PreconditionerStats &source);

enum class LinearSolverBackend { kPcg, kCudss };
extern LinearSolverBackend selected_linear_solver;

struct DirectSolverStats {
  size_t analyses = 0, factorizations = 0, solves = 0, factor_reuses = 0;
  size_t peak_device_estimate_bytes = 0;
  size_t parallel_batches = 0, peak_parallel_width = 0;
  size_t ordering_reuses = 0;
  double analysis_seconds = 0, factor_seconds = 0, solve_seconds = 0;
};
extern DirectSolverStats direct_solver_stats;

struct GpuBatchSolveTiming {
  size_t interleaved_spmv_solves = 0;
  double upload_seconds = 0.0;
  double kernel_sync_seconds = 0.0;
  double download_seconds = 0.0;
  PreconditionerStats preconditioner;
};

struct SharedOperatorBatchEvidence {
  size_t physical_operator_count = 0;
  size_t operator_upload_count = 0;
  size_t rhs_count = 0;
  size_t operator_bytes = 0;
  size_t rhs_workspace_bytes = 0;
};

constexpr int kPcgBlockThreads = 512;
constexpr int kPcgMaxCooperativeShardsPerItem = 32;
constexpr int kPcgMinCooperativeRows = 4096;
constexpr int kPcgSliceRows = 32;

int PcgCooperativeShardCount(int rows, size_t items, int multiprocessors);
bool BuildJacobiInverseDiagonal(const std::vector<double> &diagonal,
                                std::vector<double> *inverse_diagonal);

} // namespace gpu_femm

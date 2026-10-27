#include "options.h"
#include <algorithm>
#include <cmath>

namespace gpu_femm
{

PcgPreconditioner selected_preconditioner = PcgPreconditioner::kJacobi;
LinearSolverBackend selected_linear_solver = LinearSolverBackend::kPcg;
DirectSolverStats direct_solver_stats;

void AccumulatePreconditionerStats(PreconditionerStats &target, const PreconditionerStats &source)
{
  target.builds += source.builds;
  target.reuses += source.reuses;
  target.fallback_blocks += source.fallback_blocks;
  target.setup_seconds += source.setup_seconds;
}
bool BuildJacobiInverseDiagonal(const std::vector<double> &diagonal,
                                std::vector<double> *inverse_diagonal)
{
  if (inverse_diagonal == nullptr)
    return false;
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
int PcgCooperativeShardCount(int rows, size_t items, int multiprocessors)
{
  if (rows < kPcgMinCooperativeRows || items == 0 || items > 4 || multiprocessors <= 0)
    return 0;
  const int limit = std::min({kPcgMaxCooperativeShardsPerItem, rows / kPcgBlockThreads,
                              multiprocessors / static_cast<int>(items)});
  int shards = 1;
  while (shards <= limit / 2)
    shards *= 2;
  return shards >= 2 ? shards : 0;
}

} // namespace gpu_femm

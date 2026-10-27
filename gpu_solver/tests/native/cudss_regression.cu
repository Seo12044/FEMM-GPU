#include "commands.h"
#include "solvers/csr_solver.cuh"

#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <vector>

namespace gpu_femm
{

int CudssRegressionTests()
{
#ifndef GPU_FEMM_ENABLE_CUDSS
  std::cerr << "cuDSS was not enabled in this build\n";
  return 1;
#else
  int failures = 0;
  auto expect = [&](bool ok, const char *message) {
    if (!ok) {
      std::cerr << "FAIL cuDSS: " << message << '\n';
      ++failures;
    }
  };
  auto matrix = [](int n, double off = -1) {
    Assembly a;
    a.row_offsets = {0};
    for (int row = 0; row < n; ++row) {
      a.free_nodes.push_back(row);
      a.diagonal.push_back(2.01);
      if (row) {
        a.column_indices.push_back(row - 1);
        a.values.push_back(off);
      }
      a.column_indices.push_back(row);
      a.values.push_back(2.01);
      if (row + 1 < n) {
        a.column_indices.push_back(row + 1);
        a.values.push_back(off);
      }
      a.row_offsets.push_back(static_cast<int32_t>(a.values.size()));
    }
    return a;
  };
  auto multiply = [](const Assembly &a, const std::vector<double> &x) {
    std::vector<double> b(x.size(), 0);
    for (size_t row = 0; row < x.size(); ++row)
      for (int j = a.row_offsets[row]; j < a.row_offsets[row + 1]; ++j)
        b[row] += a.values[j] * x[a.column_indices[j]];
    return b;
  };
  auto verified = [&](const Assembly &a, const std::vector<double> &b, const std::vector<double> &x,
                      SolveInfo info) {
    if (info.status != Status::kOk || x.size() != b.size())
      return false;
    const auto ax = multiply(a, x);
    double error = 0, norm = 0;
    for (size_t row = 0; row < b.size(); ++row) {
      if (!std::isfinite(x[row]))
        return false;
      error = std::hypot(error, ax[row] - b[row]);
      norm = std::hypot(norm, b[row]);
    }
    return norm == 0 ? error == 0 : error / norm < 1e-10;
  };
  for (int n : {1, 7, 33, 4097}) {
    auto a = matrix(n);
    std::vector<double> exact(n);
    for (int i = 0; i < n; ++i)
      exact[i] = .2 + std::sin(.3 * i);
    auto rhs = multiply(a, exact);
    GpuCsrSolver solver(PcgPreconditioner::kJacobi, true, LinearSolverBackend::kCudss);
    expect(solver.Initialize(a) == Status::kOk, "initialize sorted symmetric CSR");
    std::vector<double> solution;
    auto info = solver.Solve(rhs, 1e-10, 2000, &solution);
    expect(verified(a, rhs, solution, info), "manufactured solution and independent residual");
    if (solution.size() != exact.size())
      continue;
    double error = 0;
    for (int i = 0; i < n; ++i)
      error = std::max(error, std::abs(solution[i] - exact[i]));
    expect(error < 1e-8, "known solution");
    const size_t factors = direct_solver_stats.factorizations,
                 analyses = direct_solver_stats.analyses;
    std::vector<SolveInfo> infos;
    std::vector<std::vector<double>> xs;
    expect(solver.SolveCurrentOperatorBatch({rhs, rhs, std::vector<double>(n, 0)}, 1e-10, 2000,
                                            &infos, &xs, nullptr) == Status::kOk,
           "shared operator solve");
    expect(infos.size() == 3 && verified(a, rhs, xs[0], infos[0]) &&
               verified(a, rhs, xs[1], infos[1]) && infos[2].status == Status::kOk &&
               xs[2] == std::vector<double>(n, 0),
           "separate RHS buffers and zero RHS");
    expect(direct_solver_stats.factorizations == factors,
           "numeric factor reuse for unchanged operator");
    auto changed = matrix(n, -.9);
    auto changed_rhs = multiply(changed, exact);
    expect(solver.UpdateValues(changed) == Status::kOk, "coefficient update");
    info = solver.Solve(changed_rhs, 1e-10, 2000, &solution);
    expect(verified(changed, changed_rhs, solution, info), "off-diagonal-only invalidation");
    expect(direct_solver_stats.factorizations == factors + 1 &&
               direct_solver_stats.analyses == analyses,
           "numeric refactor without symbolic reanalysis");
    expect(solver.SolveBatch({a, changed}, {rhs, changed_rhs}, 1e-10, 2000, &infos, &xs, nullptr) ==
               Status::kOk,
           "distinct numeric operators");
    expect(infos.size() == 2 && verified(a, rhs, xs[0], infos[0]) &&
               verified(changed, changed_rhs, xs[1], infos[1]),
           "batch slot isolation");
    expect(solver.PrepareDeviceBatch(1) == Status::kOk, "device assembly buffers");
    for (const auto &input : {a, changed}) {
      auto b = multiply(input, exact);
      CopyToDevice(solver.device_batch_values(), input.values.data(),
                   input.values.size() * sizeof(double));
      CopyToDevice(solver.device_batch_diagonal(), input.diagonal.data(), n * sizeof(double));
      CopyToDevice(solver.device_batch_rhs(), b.data(), n * sizeof(double));
      expect(solver.SolvePreparedDeviceBatch(1, 1e-10, 2000, &infos, &xs, nullptr) == Status::kOk &&
                 verified(input, b, xs[0], infos[0]),
             "same-pointer device coefficient writes invalidate factors");
    }
    expect(solver.UpdateValues(a) == Status::kOk, "restore current operator after device batch");
    std::vector<double> bad_rhs = rhs;
    bad_rhs[0] = std::numeric_limits<double>::quiet_NaN();
    expect(solver.Solve(bad_rhs, 1e-10, 2000, &solution).status == Status::kNumericalNonfinite,
           "nonfinite RHS rejected");
    expect(solver.SolveCurrentOperatorBatch({bad_rhs, rhs}, 1e-10, 2000, &infos, &xs, nullptr) ==
                   Status::kOk &&
               infos[0].status == Status::kNumericalNonfinite && verified(a, rhs, xs[1], infos[1]),
           "failed RHS does not poison the next slot");
    for (double tolerance : {0.0, -1.0, 1.0, std::numeric_limits<double>::infinity()})
      expect(solver.Solve(rhs, tolerance, 100, &solution).status == Status::kInvalidArgument,
             "invalid tolerance rejected");
    expect(solver.Solve(rhs, 1e-10, 0, &solution).status == Status::kLinearSolveNotConverged,
           "zero iteration budget is not silently ignored");
    expect(solver.Solve(std::vector<double>(n, 0), 1e-10, 0, &solution).status == Status::kOk,
           "zero RHS requires no factorization");
    expect(solver.Initialize(a) == Status::kOk,
           "reinitialize releases old plan before CSR buffers");
    info = solver.Solve(rhs, 1e-10, 2000, &solution);
    expect(verified(a, rhs, solution, info), "fresh symbolic plan");
    if (n == 33) {
      const auto reference = solution;
      size_t free_before = 0, free_after = 0, total = 0;
      cudaMemGetInfo(&free_before, &total);
      for (int repeat = 0; repeat < 100; ++repeat) {
        solver.UpdateValues(a);
        info = solver.Solve(rhs, 1e-10, 100, &solution);
        expect(verified(a, rhs, solution, info) && solution == reference,
               "deterministic repeated numeric factorization");
      }
      cudaDeviceSynchronize();
      cudaMemGetInfo(&free_after, &total);
      expect(free_after >= free_before, "no GPU memory growth over 100 warmed solves");
      for (double scale : {1e-140, 1e140}) {
        auto scaled = a;
        for (double &v : scaled.values)
          v *= scale;
        for (double &v : scaled.diagonal)
          v *= scale;
        const auto b = multiply(scaled, exact);
        solver.UpdateValues(scaled);
        info = solver.Solve(b, 1e-10, 100, &solution);
        expect(verified(scaled, b, solution, info), "scaled matrix and original residual");
      }
    }
  }
  GpuCsrSolver solver(PcgPreconditioner::kJacobi, true, LinearSolverBackend::kCudss);
  auto singular = matrix(2, 1);
  singular.diagonal = {1, 1};
  singular.values = {1, 1, 1, 1};
  expect(solver.Initialize(singular) == Status::kOk, "singular input structure");
  std::vector<double> x;
  expect(solver.Solve({1, 1}, 1e-10, 100, &x).status != Status::kOk,
         "non-positive minor cannot be accepted");
  auto good = matrix(2);
  expect(solver.UpdateValues(good) == Status::kOk, "update after numeric failure");
  expect(verified(good, {1, 2}, x, solver.Solve({1, 2}, 1e-10, 100, &x)),
         "failed factorization is not reused");
  auto asymmetric = good;
  asymmetric.values[1] = -.5;
  expect(solver.UpdateValues(asymmetric) == Status::kOk, "asymmetric numeric input staged");
  expect(solver.Solve({1, 2}, 1e-10, 100, &x).status == Status::kUnsupportedFeature,
         "full matrix symmetry checked, not silently triangularized");
  auto roundoff = good;
  roundoff.values[1] = 1e-16;
  roundoff.values[2] = 0;
  expect(solver.UpdateValues(roundoff) == Status::kOk &&
             verified(roundoff, {1, 2}, x, solver.Solve({1, 2}, 1e-10, 100, &x)),
         "cancellation-scale skew is checked against the original equation");
  good.diagonal[0] = -1;
  expect(solver.UpdateValues(good) == Status::kOk, "negative diagonal staged");
  expect(solver.Solve({1, 2}, 1e-10, 100, &x).status == Status::kLinearSolveBreakdown,
         "negative diagonal rejected");
  auto missing_transpose = matrix(3);
  missing_transpose.column_indices[1] = 2;
  expect(solver.Initialize(missing_transpose) == Status::kUnsupportedFeature,
         "missing reciprocal CSR entry rejected before factorization");
  const auto before = direct_solver_stats;
  std::vector<std::future<bool>> workers;
  for (int worker = 0; worker < 4; ++worker) {
    workers.push_back(std::async(std::launch::async, [&, worker]() {
      const auto a = matrix(33, -.8 - .01 * worker);
      const std::vector<double> exact(33, 1.0 + worker);
      const auto rhs = multiply(a, exact);
      GpuCsrSolver local(PcgPreconditioner::kJacobi, true, LinearSolverBackend::kCudss);
      if (local.Initialize(a) != Status::kOk)
        return false;
      std::vector<double> solution;
      for (int repeat = 0; repeat < 8; ++repeat) {
        const auto info = local.Solve(rhs, 1e-10, 100, &solution);
        if (!verified(a, rhs, solution, info))
          return false;
      }
      return true;
    }));
  }
  for (auto &worker : workers)
    expect(worker.get(), "concurrent independent plans");
  expect(direct_solver_stats.analyses == before.analyses + 4 &&
             direct_solver_stats.factorizations == before.factorizations + 4 &&
             direct_solver_stats.solves == before.solves + 32 &&
             direct_solver_stats.factor_reuses == before.factor_reuses + 28,
         "concurrent phase accounting is not lost");
  if (!failures)
    std::cout << "PASS cuDSS linear backend regression\n";
  return failures;
#endif
}

} // namespace gpu_femm

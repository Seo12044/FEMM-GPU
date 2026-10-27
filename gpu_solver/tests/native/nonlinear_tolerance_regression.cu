#include "commands.h"
#include "comparison.h"
#include "fixtures.h"
#include "solvers/nonlinear_solver.cuh"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace gpu_femm
{

int NonlinearToleranceTests()
{
  int failures = 0;
  auto expect = [&failures](bool ok, const char *message) {
    if (!ok) {
      std::cerr << "FAIL: " << message << '\n';
      ++failures;
    }
  };
  NonlinearOptions adaptive = ProductionSampleOptions();
  const double exact = adaptive.linear_relative_tolerance;
  expect(NewtonLinearTolerance(adaptive, 1.0) == 1e-8 &&
             Near(NewtonLinearTolerance(adaptive, 0.1), 1e-9, 1e-22),
         "inner tolerance follows nonlinear progress and is capped");
  for (double step : {0.0, -1.0, 1e-7, std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()})
    expect(NewtonLinearTolerance(adaptive, step) == exact,
           "first, converged, or invalid step uses exact inner tolerance");
  expect(!NewtonStepConverged(adaptive, 2, 1e-8, 1e-7) &&
             NewtonStepConverged(adaptive, 2, 1e-8, exact) &&
             !NewtonStepConverged(adaptive, 0, 0.0, exact) &&
             !NewtonStepConverged(adaptive, 2, 1e-3, exact),
         "inexact and premature Newton steps cannot certify convergence");
  expect(BatchNewtonLinearTolerance(adaptive, {0.1, 0.0, 0.01}, {0, 2}) == exact &&
             BatchNewtonLinearTolerance(adaptive, {0.1, 0.0, 0.01}, {0, 1, 2}) == exact,
         "batch uses the tightest active tolerance, ignoring finished slots");
  NonlinearOptions fixed = adaptive;
  fixed.adaptive_linear_tolerance = false;
  expect(NewtonLinearTolerance(fixed, 1.0) == exact, "fixed tolerance remains available");

  constexpr int side = 12;
  NonlinearModel model = LargeStructuredMaskFixture(side);
  model.dirichlet_nodes.clear();
  model.dirichlet_a_wb_per_m.clear();
  for (int y = 0; y <= side; ++y)
    for (int x = 0; x <= side; ++x) {
      const int node = y * (side + 1) + x;
      model.nodes[node] = {x / double(side), y / double(side)};
      if (x == 0 || y == 0 || x == side || y == side) {
        model.dirichlet_nodes.push_back(node);
        model.dirichlet_a_wb_per_m.push_back(0.0);
      }
    }
  model.materials = {
      {1.0, 0.3, 8.0, 0.0, 0.0, -1, 0},
      {0.5, 1.0, 0.0, 0.0, 0.0, 0},
  };
  model.bh_curves = NonlinearThreeRegionFixture().bh_curves;
  for (size_t i = 0; i < model.triangles.size(); ++i)
    model.triangles[i].material = i % 3 == 0 ? 1 : 0;
  NonlinearP1Solver solver;
  if (solver.Initialize(model) != Status::kOk) {
    expect(false, "structured nonlinear tolerance fixture initialization");
    return failures;
  }
  adaptive.relative_tolerance = 1e-10;
  adaptive.linear_relative_tolerance = 1e-12;
  adaptive.recover_nonconvergence = false;
  fixed = adaptive;
  fixed.adaptive_linear_tolerance = false;
  auto same_field = [](const NonlinearSolveResult &a, const NonlinearSolveResult &b) {
    if (a.info.status != Status::kOk || b.info.status != Status::kOk ||
        a.a_wb_per_m.size() != b.a_wb_per_m.size())
      return false;
    for (size_t i = 0; i < a.a_wb_per_m.size(); ++i)
      if (std::abs(a.a_wb_per_m[i] - b.a_wb_per_m[i]) > 1e-8)
        return false;
    return std::abs(a.flux_linkage_wb - b.flux_linkage_wb) < 1e-8;
  };
  const auto reference = solver.Solve(2.0, fixed);
  const auto accelerated = solver.Solve(2.0, adaptive);
  expect(same_field(reference, accelerated), "adaptive single solve matches fixed-precision field");
  expect(accelerated.last_linear_tolerance == adaptive.linear_relative_tolerance,
         "adaptive single solve finishes at exact precision");
  expect(accelerated.linear_iterations < reference.linear_iterations,
         "adaptive nonlinear fixture reduces PCG iterations");
  if (accelerated.info.status == Status::kOk) {
    const auto warm = solver.Solve(2.0, adaptive, &accelerated.a_wb_per_m);
    expect(same_field(reference, warm), "adaptive warm start preserves field");
  }
  const std::vector<std::vector<double>> currents = {
      {2.0}, {0.0}, {-1.5}, {}, {std::numeric_limits<double>::quiet_NaN()}};
  for (auto mode : {NonlinearAssemblyMode::kHost, NonlinearAssemblyMode::kDevice}) {
    const auto control = solver.SolveBatch(currents, fixed, nullptr, nullptr, mode);
    const auto batch = solver.SolveBatch(currents, adaptive, nullptr, nullptr, mode);
    for (size_t item = 0; item < 3; ++item) {
      expect(same_field(control[item], batch[item]), "adaptive host/device batch field parity");
      expect(batch[item].last_linear_tolerance == adaptive.linear_relative_tolerance,
             "each valid batch item finishes at exact precision");
    }
    expect(batch[3].info.status == Status::kInvalidArgument &&
               batch[4].info.status == Status::kInvalidArgument,
           "invalid batch currents remain failures");
  }
  const auto guarded = solver.SolveSafeguarded({2.0}, adaptive);
  expect(same_field(reference, guarded) &&
             guarded.verified_equation_residual_relative <= adaptive.relative_tolerance &&
             guarded.last_linear_tolerance == adaptive.linear_relative_tolerance,
         "recovery retains equation-residual and exact-precision gates");
  adaptive.max_newton_iterations = 1;
  expect(solver.Solve(2.0, adaptive).info.status == Status::kNonlinearSolveNotConverged,
         "adaptive solve cannot bypass the Newton iteration budget");
  adaptive.max_newton_iterations = 32;
  adaptive.max_linear_iterations = 0;
  expect(solver.Solve(2.0, adaptive).info.status == Status::kLinearSolveNotConverged,
         "adaptive solve cannot bypass the PCG iteration budget");
  std::cout << "nonlinear tolerance fixture PCG iterations: " << reference.linear_iterations
            << " -> " << accelerated.linear_iterations << '\n';
  return failures;
}

} // namespace gpu_femm

#include "commands.h"
#include "comparison.h"
#include "reference.h"
#include "solvers/linear_solver.cuh"
#include "solvers/nonlinear_solver.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace gpu_femm
{

namespace
{
// FEMM's centimetre-based dump stores A = (100 * mu0) V.
constexpr double kFemmInternalToPhysicalA = 100.0 * kMu0;
} // namespace

int FemmReferenceTest(const std::string &stem)
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
  auto expect = [&failures](bool condition, const std::string &message) {
    if (!condition) {
      std::cerr << "FAIL: " << message << '\n';
      ++failures;
    }
  };

  LinearP1Solver physical_solver;
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
        max_a_error =
            std::max(max_a_error, std::abs(result.a_wb_per_m[i] - reference.expected_a[i]));
        expect(NearMixed(result.a_wb_per_m[i], reference.expected_a[i], 1e-12, 1e-4),
               "FEMM nodal A parity");
      }
    }
    if (result.bx_t.size() == reference.expected_bx.size() &&
        result.by_t.size() == reference.expected_by.size()) {
      for (size_t i = 0; i < result.bx_t.size(); ++i) {
        max_b_error =
            std::max(max_b_error, std::max(std::abs(result.bx_t[i] - reference.expected_bx[i]),
                                           std::abs(result.by_t[i] - reference.expected_by[i])));
        expect(NearMixed(result.bx_t[i], reference.expected_bx[i], 1e-9, 1e-4), "FEMM Bx parity");
        expect(NearMixed(result.by_t[i], reference.expected_by[i], 1e-9, 1e-4), "FEMM By parity");
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
    const SolveInfo info = dump_solver.Solve(reference.dump_rhs, 1e-13, 256, &internal_solution);
    expect(info.status == Status::kOk, std::string("FEMM dump solve: ") + StatusName(info.status));
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
              << " triangles=" << reference.model.triangles.size() << " max_A_error=" << max_a_error
              << " max_B_error=" << max_b_error << '\n'
              << "  flux_reference_Wb=" << reference.flux_linkage_wb << '\n';
  }
  return failures == 0 ? 0 : 1;
}

int NonlinearFemmReferenceTest(const std::string &stem, const std::string &curve_directory)
{
  NonlinearFemmReference reference;
  std::string error;
  if (!ReadNonlinearFemmReference(stem, curve_directory, &reference, &error)) {
    std::cerr << "FAIL nonlinear FEMM reference input: " << error << '\n';
    return 1;
  }
  NonlinearP1Solver solver;
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
  const bool parity = NearMixed(actual.flux_linkage_wb, reference.flux_linkage_wb, 1e-10, 1e-6) &&
                      max_a_error <= 1e-9 && max_b_error <= 1e-6;
  if (!parity) {
    std::cerr << "FAIL nonlinear FEMM parity A=" << max_a_error << " B=" << max_b_error
              << " flux=" << actual.flux_linkage_wb
              << " expected_flux=" << reference.flux_linkage_wb << '\n';
    return 1;
  }
  std::cout << "PASS gpu_nonlinear_p1_femm_reference\n"
            << "  nodes=" << reference.model.nodes.size()
            << " triangles=" << reference.model.triangles.size()
            << " iterations=" << actual.info.iterations << " max_A_error=" << max_a_error
            << " max_B_error=" << max_b_error << " flux_Wb=" << actual.flux_linkage_wb << '\n';
  return 0;
}

int FrozenPostprocessReferenceTest(const std::string &stem, const std::string &curve_directory)
{
  NonlinearFemmReference nonlinear_reference;
  FrozenPostprocessReference postprocess_reference;
  std::string error;
  if (!ReadNonlinearFemmReference(stem, curve_directory, &nonlinear_reference, &error) ||
      !ReadFrozenPostprocessReference(stem, &postprocess_reference, &error)) {
    std::cerr << "FAIL postprocess FEMM reference input: " << error << '\n';
    return 1;
  }
  NonlinearP1Solver solver;
  const Status initialize = solver.Initialize(nonlinear_reference.model);
  if (initialize != Status::kOk) {
    std::cerr << "FAIL postprocess FEMM initialization: " << StatusName(initialize) << '\n';
    return 1;
  }
  NonlinearOptions nonlinear_options;
  nonlinear_options.relative_tolerance = 1e-8;
  nonlinear_options.max_newton_iterations = 128;
  const NonlinearSolveResult solution =
      solver.Solve(nonlinear_reference.current_a, nonlinear_options);
  if (solution.info.status != Status::kOk) {
    std::cerr << "FAIL postprocess FEMM solve: " << StatusName(solution.info.status) << '\n';
    return 1;
  }
  const FrozenPostprocessResult actual =
      ComputeFrozenPostprocess(nonlinear_reference.model, solution, postprocess_reference.options);
  if (actual.status != Status::kOk) {
    std::cerr << "FAIL postprocess calculation: " << StatusName(actual.status) << '\n';
    return 1;
  }
  bool pass = NearMixed(actual.force_x_n, postprocess_reference.expected_force_x_n, 1e-8, 1e-5) &&
              NearMixed(actual.force_y_n, postprocess_reference.expected_force_y_n, 1e-8, 1e-5) &&
              NearMixed(actual.torque_nm, postprocess_reference.expected_torque_nm, 1e-10, 1e-5) &&
              nonlinear_reference.model.nodes.size() ==
                  static_cast<size_t>(postprocess_reference.expected_node_count) &&
              nonlinear_reference.model.triangles.size() ==
                  static_cast<size_t>(postprocess_reference.expected_element_count) &&
              NearMixed(nonlinear_reference.current_a, postprocess_reference.expected_current_a,
                        1e-12, 1e-12) &&
              NearMixed(solution.flux_linkage_wb, postprocess_reference.expected_flux_linkage_wb,
                        1e-10, 1e-6) &&
              actual.airgap_samples.size() == postprocess_reference.expected_radial_b_t.size();
  double max_airgap_error = 0.0;
  if (actual.airgap_samples.size() == postprocess_reference.expected_radial_b_t.size()) {
    for (size_t index = 0; index < actual.airgap_samples.size(); ++index) {
      max_airgap_error =
          std::max(max_airgap_error, std::abs(actual.airgap_samples[index].radial_b_t -
                                              postprocess_reference.expected_radial_b_t[index]));
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
            << " torque_Nm=" << actual.torque_nm << " max_airgap_B_error=" << max_airgap_error
            << '\n';
  return 0;
}

} // namespace gpu_femm

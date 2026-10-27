#include "comparison.h"
#include "fixtures.h"
#include "suites.h"
#include "physics/nonlinear_model.h"
#include "postprocess/postprocess.h"
#include "solvers/nonlinear_solver.cuh"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace gpu_femm
{

int PhysicsRegressionTests()
{
  int failures = 0;
  auto expect = [&failures](bool condition, const std::string &message) {
    if (!condition) {
      std::cerr << "FAIL: " << message << '\n';
      ++failures;
    }
  };

  Model no_boundary = UnitSquareFixture();
  no_boundary.dirichlet_nodes.clear();
  no_boundary.dirichlet_a_wb_per_m.clear();
  expect(ValidateModel(no_boundary) == Status::kBoundaryInvalid,
         "missing boundary returns BOUNDARY_INVALID");
  Model clockwise = UnitSquareFixture();
  std::swap(clockwise.triangles[0].node[1], clockwise.triangles[0].node[2]);
  expect(ValidateModel(clockwise) == Status::kMeshInvalid,
         "clockwise element returns MESH_INVALID");
  Model floating_component = UnitSquareFixture();
  floating_component.nodes.push_back({2.0, 2.0});
  expect(ValidateModel(floating_component) == Status::kBoundaryInvalid,
         "floating component returns BOUNDARY_INVALID");
  expect(std::string(StatusName(Status::kLinearSolveNotConverged)) == "LINEAR_SOLVE_NOT_CONVERGED",
         "stable error-code name");
  expect(std::string(StatusName(Status::kNonlinearSolveNotConverged)) ==
             "NONLINEAR_SOLVE_NOT_CONVERGED",
         "stable nonlinear error-code name");

  // Analytic P1 contract: b_i=(dN_i/dy,-dN_i/dx), f_PM=S Hc.b_i.
  const Node analytic_nodes[3] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  NonlinearElementTerms analytic_terms;
  expect(BuildElementTerms(analytic_nodes, &analytic_terms), "analytic element terms");
  const double expected_pm[3] = {-5.0, 0.0, 5.0};
  for (int i = 0; i < 3; ++i) {
    expect(Near(analytic_terms.area_m2 * 10.0 * analytic_terms.b_x[i], expected_pm[i]),
           "PM source sign");
  }
  // At A=[0,0,1], v=1 and dv/d(B^2)=.5.  This is K+C exactly.
  const double bx = analytic_terms.b_x[2];
  const double by = analytic_terms.b_y[2];
  const double expected_jacobian[3][3] = {{1.5, -0.5, -1.0}, {-0.5, 0.5, 0.0}, {-1.0, 0.0, 1.0}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double b_dot_i = bx * analytic_terms.b_x[i] + by * analytic_terms.b_y[i];
      const double b_dot_j = bx * analytic_terms.b_x[j] + by * analytic_terms.b_y[j];
      const double actual =
          analytic_terms.area_m2 * (analytic_terms.b_x[i] * analytic_terms.b_x[j] +
                                    analytic_terms.b_y[i] * analytic_terms.b_y[j]) +
          2.0 * analytic_terms.area_m2 * 0.5 * b_dot_i * b_dot_j;
      expect(Near(actual, expected_jacobian[i][j]), "analytic Newton Jacobian");
    }
  }
  NonlinearModel missing_material = NonlinearThreeRegionFixture();
  missing_material.triangles[0].material = -1;
  expect(ValidateNonlinearModel(missing_material) == Status::kInvalidMaterial,
         "unassigned nonlinear material is rejected");

  // DC weighted-stress primitive: T*h in free space, in SI units.
  const double primitive_bx = 2.0, primitive_by = 3.0;
  const double primitive_hx = 0.25, primitive_hy = -0.5;
  const double primitive_fx =
      ((primitive_bx * primitive_bx - primitive_by * primitive_by) * primitive_hx +
       2.0 * primitive_bx * primitive_by * primitive_hy) /
      (2.0 * kMu0);
  const double primitive_fy =
      (2.0 * primitive_bx * primitive_by * primitive_hx +
       (primitive_by * primitive_by - primitive_bx * primitive_bx) * primitive_hy) /
      (2.0 * kMu0);
  expect(Near(primitive_fx, -7.25 / (2.0 * kMu0)), "weighted-stress Fx primitive");
  expect(Near(primitive_fy, 0.5 / (2.0 * kMu0)), "weighted-stress Fy primitive");

  // A=x gives B=(0,-1) in every P1 triangle.  The selected inner PM square
  // is surrounded by air, so FEMM's WeightingScheme=0 mask is valid and its
  // net load must vanish in a uniform field.  Polar air samples exercise the
  // default-smoothed point field and B dot e_r projection.
  const NonlinearModel postprocess_model = PostprocessRingFixture();
  NonlinearSolveResult postprocess_solution;
  postprocess_solution.a_wb_per_m.resize(postprocess_model.nodes.size());
  for (size_t node = 0; node < postprocess_model.nodes.size(); ++node)
    postprocess_solution.a_wb_per_m[node] = postprocess_model.nodes[node].x_m;
  postprocess_solution.bx_t.assign(postprocess_model.triangles.size(), 0.0);
  postprocess_solution.by_t.assign(postprocess_model.triangles.size(), -1.0);
  FrozenPostprocessOptions postprocess_options;
  postprocess_options.airgap_radius_m = 0.75;
  postprocess_options.airgap_angles_rad = {0.0, 0.5 * 3.141592653589793238462643383279502884};
  const FrozenPostprocessResult postprocess =
      ComputeFrozenPostprocess(postprocess_model, postprocess_solution, postprocess_options);
  FrozenPostprocessPlan reusable_postprocess_plan;
  const Status reusable_postprocess_plan_status = BuildFrozenPostprocessPlan(
      postprocess_model, postprocess_options, &reusable_postprocess_plan);
  const FrozenPostprocessResult reused_postprocess =
      reusable_postprocess_plan_status == Status::kOk
          ? ComputeFrozenPostprocess(postprocess_model, postprocess_solution, postprocess_options,
                                     &reusable_postprocess_plan)
          : FrozenPostprocessResult{};
  expect(postprocess.status == Status::kOk,
         std::string("weighted-stress mask postprocess: ") + StatusName(postprocess.status));
  expect(reusable_postprocess_plan_status == Status::kOk &&
             reused_postprocess.status == postprocess.status &&
             Near(reused_postprocess.force_x_n, postprocess.force_x_n, 1e-12) &&
             Near(reused_postprocess.force_y_n, postprocess.force_y_n, 1e-12) &&
             Near(reused_postprocess.torque_nm, postprocess.torque_nm, 1e-12) &&
             reused_postprocess.airgap_samples.size() == postprocess.airgap_samples.size(),
         "reused frozen postprocess plan preserves load and field results");
  if (postprocess.status == Status::kOk) {
    expect(Near(postprocess.force_x_n, 0.0, 1e-8), "uniform-field weighted Fx cancellation");
    expect(Near(postprocess.force_y_n, 0.0, 1e-8), "uniform-field weighted Fy cancellation");
    expect(Near(postprocess.torque_nm, 0.0, 1e-8), "uniform-field weighted torque cancellation");
    expect(postprocess.airgap_samples.size() == 2, "airgap sample count");
    if (postprocess.airgap_samples.size() == 2) {
      expect(Near(postprocess.airgap_samples[0].radial_b_t, 0.0, 1e-12),
             "airgap radial B at zero degrees");
      expect(Near(postprocess.airgap_samples[1].radial_b_t, -1.0, 1e-12),
             "airgap radial B at ninety degrees");
    }
  }
  FrozenPostprocessOptions invalid_selection = postprocess_options;
  invalid_selection.selected_material_label = 2;
  const FrozenPostprocessResult no_selection =
      ComputeFrozenPostprocess(postprocess_model, postprocess_solution, invalid_selection);
  expect(no_selection.status == Status::kInvalidMaterial, "missing selected material is rejected");

  // Group selection is the production artifact path: all rotor labels in
  // group 20 are selected together and only group 30 is free-space air.
  NonlinearModel group_postprocess_model = PostprocessRingFixture();
  group_postprocess_model.circuit_count = 2;
  group_postprocess_model.materials[1].group_number = 20;
  group_postprocess_model.materials[3].group_number = 30;
  NonlinearP1Solver group_solver;
  const Status group_initialize = group_solver.Initialize(group_postprocess_model);
  NonlinearSolveResult group_solution;
  if (group_initialize == Status::kOk)
    group_solution = group_solver.Solve(std::vector<double>{0.0, 0.0});
  FrozenPostprocessOptions artifact_group_options;
  artifact_group_options.selected_group_number = 20;
  artifact_group_options.air_group_number = 30;
  artifact_group_options.airgap_radius_m = 0.75;
  artifact_group_options.airgap_angles_rad = {0.0};
  const FrozenPostprocessResult group_postprocess =
      group_initialize == Status::kOk && group_solution.info.status == Status::kOk
          ? ComputeFrozenPostprocess(group_postprocess_model, group_solution,
                                     artifact_group_options)
          : FrozenPostprocessResult{};
  expect(group_initialize == Status::kOk && group_solution.circuit_flux_linkage_wb.size() == 2 &&
             group_postprocess.status == Status::kOk &&
             group_postprocess.airgap_samples.size() == 1,
         "two-circuit group-based artifact solve and postprocess");

  // 258^2 nodes imply a 35 GiB dense double matrix.  This only assembles the
  // analytic stencil (no long FEMM/CUDA solve); it is a regression guard for
  // accidentally reintroducing node_count^2 host storage in all three paths.
  constexpr int kScalabilityCellsPerSide = 257;
  constexpr size_t kScalabilityNodesPerSide = kScalabilityCellsPerSide + 1;
  constexpr size_t kScalabilityNodeCount = kScalabilityNodesPerSide * kScalabilityNodesPerSide;
  constexpr size_t kDenseBytesAvoided =
      kScalabilityNodeCount * kScalabilityNodeCount * sizeof(double);
  expect(kDenseBytesAvoided > static_cast<size_t>(32) * 1024 * 1024 * 1024,
         "scalability fixture exceeds practical dense host allocation");
  {
    const Model scalable_linear = LargeStructuredLinearFixture(kScalabilityCellsPerSide);
    Assembly scalable_assembly;
    const Status scalable_status = Assemble(scalable_linear, &scalable_assembly);
    expect(scalable_status == Status::kOk,
           std::string("large sparse linear assembly: ") + StatusName(scalable_status));
    expect(scalable_assembly.free_nodes.size() + scalable_linear.dirichlet_nodes.size() ==
                   scalable_linear.nodes.size() &&
               scalable_assembly.values.size() < scalable_linear.triangles.size() * 9,
           "large linear assembly retains P1 sparsity");
  }
  {
    const NonlinearModel scalable_mask = LargeStructuredMaskFixture(kScalabilityCellsPerSide);
    const std::vector<double> zero_a(scalable_mask.nodes.size(), 0.0);
    Assembly scalable_newton;
    const Status newton_status =
        AssembleNonlinearNewton(scalable_mask, zero_a, 0.0, false, &scalable_newton);
    expect(newton_status == Status::kOk,
           std::string("large sparse nonlinear assembly: ") + StatusName(newton_status));
    expect(scalable_newton.free_nodes.size() == 3 &&
               scalable_newton.values.size() <= scalable_mask.triangles.size() * 9,
           "large nonlinear assembly retains P1 sparsity");
    FrozenPostprocessOptions scalable_mask_options;
    scalable_mask_options.selected_material_label = 1;
    scalable_mask_options.air_material_label = 0;
    std::vector<double> scalable_mask_values;
    const Status mask_status =
        BuildWeightedStressMask(scalable_mask, scalable_mask_options, &scalable_mask_values);
    expect(mask_status == Status::kOk,
           std::string("large sparse WST mask assembly: ") + StatusName(mask_status));
    expect(scalable_mask_values.size() == scalable_mask.nodes.size(), "large WST mask dimensions");
  }
  {
    NonlinearModel paired_mask = LargeStructuredMaskFixture(4);
    paired_mask.dirichlet_nodes.clear();
    paired_mask.dirichlet_a_wb_per_m.clear();
    constexpr int paired_side = 5;
    for (int y = 0; y < paired_side; ++y)
      for (int x = 0; x < paired_side; ++x)
        if (x == 0 || y == 0 || x + 1 == paired_side || y + 1 == paired_side) {
          paired_mask.dirichlet_nodes.push_back(y * paired_side + x);
          paired_mask.dirichlet_a_wb_per_m.push_back(0.0);
        }
    paired_mask.node_constraints = {{6, 8, -1}};
    paired_mask.symmetry_apex_nodes = {12};
    paired_mask.dirichlet_nodes.push_back(12);
    paired_mask.dirichlet_a_wb_per_m.push_back(0.0);
    FrozenPostprocessOptions paired_mask_options;
    paired_mask_options.selected_material_label = 1;
    paired_mask_options.air_material_label = 0;
    std::vector<double> paired_mask_values;
    const Status paired_mask_status =
        BuildWeightedStressMask(paired_mask, paired_mask_options, &paired_mask_values);
    expect(paired_mask_status == Status::kOk &&
               paired_mask_values.size() == paired_mask.nodes.size() &&
               paired_mask_values[6] == paired_mask_values[8] && paired_mask_values[12] == 1.0,
           "WST mask uses periodic cut pairs and ignores magnetic AP apex zero as a physical mask "
           "boundary");
  }
  return failures;
}

} // namespace gpu_femm

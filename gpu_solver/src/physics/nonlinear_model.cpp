#include "physics/nonlinear_model.h"
#include "axisymmetric_p1.h"
#include "native_airgap_matrix.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <utility>

namespace gpu_femm
{

bool BuildNonlinearBhSpline(NonlinearBhCurve *curve)
{
  const int n = static_cast<int>(curve->b_t.size());
  if (n < 2 || curve->h_a_per_m.size() != curve->b_t.size() || curve->b_t.front() != 0.0 ||
      curve->h_a_per_m.front() != 0.0)
    return false;
  if (!(curve->cold_secant_reluctivity_m_per_h > 0.0))
    curve->cold_secant_reluctivity_m_per_h = curve->h_a_per_m[1] / curve->b_t[1];
  for (int i = 1; i < n; ++i) {
    if (!(curve->b_t[i] > curve->b_t[i - 1]) || curve->h_a_per_m[i] < 0.0 ||
        !std::isfinite(curve->b_t[i]) || !std::isfinite(curve->h_a_per_m[i]))
      return false;
  }
  // Exact FEMM DC natural-spline linear system (including its end rows).
  std::vector<double> matrix(static_cast<size_t>(n) * n, 0.0), rhs(n, 0.0);
  const double first = curve->b_t[1] - curve->b_t[0];
  matrix[0] = 4.0 / first;
  matrix[1] = 2.0 / first;
  rhs[0] = 6.0 * (curve->h_a_per_m[1] - curve->h_a_per_m[0]) / (first * first);
  const double last = curve->b_t[n - 1] - curve->b_t[n - 2];
  matrix[(n - 1) * n + n - 1] = 4.0 / last;
  matrix[(n - 1) * n + n - 2] = 2.0 / last;
  rhs[n - 1] = 6.0 * (curve->h_a_per_m[n - 1] - curve->h_a_per_m[n - 2]) / (last * last);
  for (int i = 1; i < n - 1; ++i) {
    const double left = curve->b_t[i] - curve->b_t[i - 1];
    const double right = curve->b_t[i + 1] - curve->b_t[i];
    matrix[i * n + i - 1] = 2.0 / left;
    matrix[i * n + i] = 4.0 * (left + right) / (left * right);
    matrix[i * n + i + 1] = 2.0 / right;
    rhs[i] = 6.0 * (curve->h_a_per_m[i] - curve->h_a_per_m[i - 1]) / (left * left) +
             6.0 * (curve->h_a_per_m[i + 1] - curve->h_a_per_m[i]) / (right * right);
  }
  for (int pivot_col = 0; pivot_col < n; ++pivot_col) {
    int pivot = pivot_col;
    for (int row = pivot_col + 1; row < n; ++row) {
      if (std::abs(matrix[row * n + pivot_col]) > std::abs(matrix[pivot * n + pivot_col]))
        pivot = row;
    }
    if (!(std::abs(matrix[pivot * n + pivot_col]) > 0.0))
      return false;
    for (int col = pivot_col; col < n; ++col)
      std::swap(matrix[pivot_col * n + col], matrix[pivot * n + col]);
    std::swap(rhs[pivot_col], rhs[pivot]);
    for (int row = pivot_col + 1; row < n; ++row) {
      const double factor = matrix[row * n + pivot_col] / matrix[pivot_col * n + pivot_col];
      for (int col = pivot_col; col < n; ++col)
        matrix[row * n + col] -= factor * matrix[pivot_col * n + col];
      rhs[row] -= factor * rhs[pivot_col];
    }
  }
  curve->slope.assign(n, 0.0);
  for (int row = n - 1; row >= 0; --row) {
    double value = rhs[row];
    for (int col = row + 1; col < n; ++col)
      value -= matrix[row * n + col] * curve->slope[col];
    curve->slope[row] = value / matrix[row * n + row];
    if (!std::isfinite(curve->slope[row]))
      return false;
  }
  // FEMM repeats the natural-spline solve after 3-point smoothing until no
  // segment has an interior derivative root. Keep the original first secant
  // above for the cold (iteration zero) assembly rule.
  bool monotone = true;
  for (int i = 1; i < n; ++i) {
    const double length = curve->b_t[i] - curve->b_t[i - 1];
    const double d0 = curve->slope[i - 1], d1 = curve->slope[i];
    const double h0 = curve->h_a_per_m[i - 1], h1 = curve->h_a_per_m[i];
    const double c0 = d0;
    const double c1 =
        -2.0 * (2.0 * d0 * length + d1 * length + 3.0 * h0 - 3.0 * h1) / (length * length);
    const double c2 =
        3.0 * (d0 * length + d1 * length + 2.0 * h0 - 2.0 * h1) / (length * length * length);
    double root0 = -1.0, root1 = -1.0;
    const double discriminant = c1 * c1 - 4.0 * c0 * c2;
    if (c2 == 0.0) {
      if (c1 != 0.0)
        root0 = -c0 / c1;
    } else if (discriminant > 0.0) {
      const double root = std::sqrt(discriminant);
      root0 = -(c1 + root) / (2.0 * c2);
      root1 = (-c1 + root) / (2.0 * c2);
    }
    if ((root0 >= 0.0 && root0 <= length) || (root1 >= 0.0 && root1 <= length)) {
      monotone = false;
      break;
    }
  }
  if (monotone)
    return true;
  if (++curve->smoothing_passes > 1024)
    return false;
  std::vector<double> next_b = curve->b_t, next_h = curve->h_a_per_m;
  for (int i = 1; i < n - 1; ++i) {
    next_b[i] = (curve->b_t[i - 1] + curve->b_t[i] + curve->b_t[i + 1]) / 3.0;
    next_h[i] = (curve->h_a_per_m[i - 1] + curve->h_a_per_m[i] + curve->h_a_per_m[i + 1]) / 3.0;
  }
  curve->b_t = std::move(next_b);
  curve->h_a_per_m = std::move(next_h);
  curve->slope.clear();
  return BuildNonlinearBhSpline(curve);
}

NonlinearBhEvaluation EvaluateNonlinearBh(const NonlinearBhCurve &curve, double field_b)
{
  NonlinearBhEvaluation result;
  const double b = std::abs(field_b);
  if (curve.slope.size() != curve.b_t.size() || b == 0.0) {
    result.dh_db = curve.slope.empty() ? std::numeric_limits<double>::quiet_NaN() : curve.slope[0];
    result.reluctivity_m_per_h = result.dh_db;
    return result;
  }
  const int n = static_cast<int>(curve.b_t.size());
  if (b > curve.b_t.back()) {
    result.h_a_per_m = curve.h_a_per_m.back() + curve.slope.back() * (b - curve.b_t.back());
    result.dh_db = curve.slope.back();
  } else {
    for (int i = 0; i < n - 1; ++i) {
      if (b >= curve.b_t[i] && b <= curve.b_t[i + 1]) {
        const double length = curve.b_t[i + 1] - curve.b_t[i];
        const double z = (b - curve.b_t[i]) / length, z2 = z * z;
        result.h_a_per_m = (1.0 - 3.0 * z2 + 2.0 * z2 * z) * curve.h_a_per_m[i] +
                           z * (1.0 - 2.0 * z + z2) * length * curve.slope[i] +
                           z2 * (3.0 - 2.0 * z) * curve.h_a_per_m[i + 1] +
                           z2 * (z - 1.0) * length * curve.slope[i + 1];
        result.dh_db = 6.0 * z * (z - 1.0) * curve.h_a_per_m[i] / length +
                       (1.0 - 4.0 * z + 3.0 * z2) * curve.slope[i] +
                       6.0 * z * (1.0 - z) * curve.h_a_per_m[i + 1] / length +
                       z * (3.0 * z - 2.0) * curve.slope[i + 1];
        break;
      }
    }
  }
  result.reluctivity_m_per_h = result.h_a_per_m / b;
  result.dv_db2 = 0.5 * (result.dh_db / (b * b) - result.h_a_per_m / (b * b * b));
  return result;
}

bool ReadNonlinearBhTable(const std::string &path, NonlinearBhCurve *curve)
{
  std::ifstream input(path);
  std::string header;
  if (!input || !std::getline(input, header) ||
      header.find("H (A_per_meter)") == std::string::npos ||
      header.find("B (tesla)") == std::string::npos)
    return false;
  double h = 0.0, b = 0.0;
  while (input >> h >> b) {
    curve->h_a_per_m.push_back(h);
    curve->b_t.push_back(b);
  }
  return input.eof() && BuildNonlinearBhSpline(curve);
}

double NewtonLinearTolerance(const NonlinearOptions &options, double previous_step)
{
  const double exact = options.linear_relative_tolerance;
  if (!options.adaptive_linear_tolerance || !std::isfinite(previous_step) ||
      previous_step <= 100.0 * options.relative_tolerance || !(exact > 0.0) ||
      !std::isfinite(exact))
    return exact;
  return std::max(exact, 1e-8 * std::min(1.0, previous_step));
}

double BatchNewtonLinearTolerance(const NonlinearOptions &options,
                                  const std::vector<double> &previous_steps,
                                  const std::vector<size_t> &slots)
{
  double tolerance = std::numeric_limits<double>::infinity();
  for (size_t slot : slots)
    tolerance = std::min(tolerance, NewtonLinearTolerance(options, previous_steps[slot]));
  return tolerance;
}

bool NewtonStepConverged(const NonlinearOptions &options, int iteration, double relative_step,
                         double linear_tolerance)
{
  // An inexact correction cannot certify convergence. Re-solve at the original
  // linear tolerance before accepting the nonlinear step criterion.
  return iteration > 0 && relative_step < 100.0 * options.relative_tolerance &&
         linear_tolerance <= options.linear_relative_tolerance;
}

NonlinearOptions ProductionSampleOptions()
{
  NonlinearOptions options;
  options.relative_tolerance = 1e-8;
  options.max_newton_iterations = 128;
  options.linear_relative_tolerance = kMotorLinearRelativeTolerance;
  options.max_linear_iterations = kMotorMaxLinearIterations;
  options.recover_nonconvergence = true;
  options.adaptive_linear_tolerance = true;
  return options;
}

bool BuildElementTerms(const Node p[3], NonlinearElementTerms *terms)
{
  const double determinant =
      (p[1].x_m - p[0].x_m) * (p[2].y_m - p[0].y_m) - (p[2].x_m - p[0].x_m) * (p[1].y_m - p[0].y_m);
  if (!(determinant > 0.0) || !std::isfinite(determinant))
    return false;
  terms->area_m2 = 0.5 * determinant;
  const double dx[3] = {p[1].y_m - p[2].y_m, p[2].y_m - p[0].y_m, p[0].y_m - p[1].y_m};
  const double dy[3] = {p[2].x_m - p[1].x_m, p[0].x_m - p[2].x_m, p[1].x_m - p[0].x_m};
  for (int i = 0; i < 3; ++i) {
    terms->b_x[i] = dy[i] / determinant;
    terms->b_y[i] = -dx[i] / determinant;
  }
  return true;
}

Status ValidateNonlinearModel(const NonlinearModel &model)
{
  if (model.nodes.empty() || model.triangles.empty() || model.materials.empty() ||
      (model.problem_type == MagneticProblemType::kPlanar &&
       (!(model.depth_m > 0.0) || !std::isfinite(model.depth_m))) ||
      model.circuit_count < 0 || model.dirichlet_nodes.empty() ||
      model.dirichlet_nodes.size() != model.dirichlet_a_wb_per_m.size()) {
    return Status::kInvalidArgument;
  }
  std::vector<bool> boundary(model.nodes.size(), false);
  for (size_t i = 0; i < model.dirichlet_nodes.size(); ++i) {
    const int32_t node = model.dirichlet_nodes[i];
    if (node < 0 || static_cast<size_t>(node) >= model.nodes.size() || boundary[node] ||
        !std::isfinite(model.dirichlet_a_wb_per_m[i]))
      return Status::kBoundaryInvalid;
    boundary[node] = true;
  }
  if (model.dirichlet_nodes.size() == model.nodes.size())
    return Status::kBoundaryInvalid;
  for (const Node &node : model.nodes) {
    if (!std::isfinite(node.x_m) || !std::isfinite(node.y_m) ||
        (model.problem_type == MagneticProblemType::kAxisymmetric && node.x_m < 0.0))
      return Status::kMeshInvalid;
  }
  for (const NonlinearMaterial &material : model.materials) {
    if (!(material.reluctivity_zero_m_per_h > 0.0) || material.alpha_per_t2 < 0.0 ||
        !std::isfinite(material.reluctivity_zero_m_per_h) ||
        !std::isfinite(material.alpha_per_t2) || !std::isfinite(material.source_j_per_a) ||
        !std::isfinite(material.h_c_a_per_m) || !std::isfinite(material.magnetization_deg))
      return Status::kInvalidMaterial;
    if (material.circuit_index < -1 || material.circuit_index >= model.circuit_count ||
        (material.source_j_per_a != 0.0 && material.circuit_index < 0))
      return Status::kInvalidMaterial;
    if (material.bh_curve_index < -1 ||
        (material.bh_curve_index >= 0 &&
         static_cast<size_t>(material.bh_curve_index) >= model.bh_curves.size()))
      return Status::kInvalidMaterial;
    if (material.bh_curve_index >= 0) {
      const NonlinearBhCurve &curve = model.bh_curves[material.bh_curve_index];
      if (curve.b_t.size() < 2 || curve.h_a_per_m.size() != curve.b_t.size() ||
          curve.slope.size() != curve.b_t.size() || !(curve.cold_secant_reluctivity_m_per_h > 0.0))
        return Status::kInvalidMaterial;
    }
  }
  for (const NonlinearTriangle &triangle : model.triangles) {
    if (triangle.material < 0 || static_cast<size_t>(triangle.material) >= model.materials.size())
      return Status::kInvalidMaterial;
    Node p[3];
    for (int local = 0; local < 3; ++local) {
      if (triangle.node[local] < 0 ||
          static_cast<size_t>(triangle.node[local]) >= model.nodes.size())
        return Status::kMeshInvalid;
      p[local] = model.nodes[triangle.node[local]];
    }
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms))
      return Status::kMeshInvalid;
  }
  for (const AirGapElement &age : model.air_gap_elements) {
    if (model.problem_type != MagneticProblemType::kPlanar)
      return Status::kUnsupportedFeature;
    const size_t elements = age.quad_points.size() - 1;
    if (age.quad_points.size() < 3 || !(age.inner_radius_m > 0.0) ||
        !(age.outer_radius_m > age.inner_radius_m) || !(age.arc_length_deg > 0.0) ||
        !std::isfinite(age.center_x_m) || !std::isfinite(age.center_y_m) ||
        !std::isfinite(age.inner_shift) || !std::isfinite(age.outer_shift))
      return Status::kMeshInvalid;
    for (const AirGapQuadPoint &point : age.quad_points)
      for (int local = 0; local < 4; ++local) {
        if (point.node[local] < 0 || static_cast<size_t>(point.node[local]) >= model.nodes.size() ||
            !std::isfinite(point.weight[local]))
          return Status::kMeshInvalid;
      }
    const double dt = age.arc_length_deg / static_cast<double>(elements);
    if (!(dt > 0.0) || !std::isfinite(dt))
      return Status::kMeshInvalid;
  }
  SignedDofMap dofs;
  const SignedDofStatus dof_status =
      BuildSignedDofMap(model.nodes.size(), model.node_constraints, model.dirichlet_nodes,
                        model.dirichlet_a_wb_per_m, &dofs);
  if (dof_status != SignedDofStatus::kOk || dofs.free_roots.empty())
    return Status::kBoundaryInvalid;
  if (model.problem_type == MagneticProblemType::kAxisymmetric) {
    double radius_scale_m = 1e-3;
    for (const Node &node : model.nodes)
      radius_scale_m = std::max(radius_scale_m, node.x_m);
    const double axis_tolerance_m = 1e-12 * radius_scale_m;
    for (size_t node = 0; node < model.nodes.size(); ++node) {
      const int32_t root = dofs.node_root[node];
      if (model.nodes[node].x_m <= axis_tolerance_m &&
          (!dofs.root_fixed[root] || dofs.root_value[root] != 0.0))
        return Status::kBoundaryInvalid;
    }
  }
  return Status::kOk;
}

Status ReduceSignedSystem(const NonlinearModel &model, const SparseRows &full_matrix,
                          const std::vector<double> &full_rhs, Assembly *assembly)
{
  if (assembly == nullptr || full_matrix.size() != model.nodes.size() ||
      full_rhs.size() != model.nodes.size())
    return Status::kInvalidArgument;
  SignedDofMap dofs;
  if (BuildSignedDofMap(model.nodes.size(), model.node_constraints, model.dirichlet_nodes,
                        model.dirichlet_a_wb_per_m, &dofs) != SignedDofStatus::kOk ||
      dofs.free_roots.empty())
    return Status::kBoundaryInvalid;

  const size_t free_count = dofs.free_roots.size();
  SparseRows reduced(free_count);
  std::vector<double> reduced_rhs(free_count, 0.0);
  std::vector<double> rhs_offset(free_count, 0.0);
  assembly->boundary_values.assign(model.nodes.size(), 0.0);
  assembly->node_to_free.assign(model.nodes.size(), -1);
  assembly->node_sign = dofs.node_sign;
  for (size_t node = 0; node < model.nodes.size(); ++node) {
    const int32_t root = dofs.node_root[node];
    assembly->node_to_free[node] = dofs.root_to_free[root];
    if (dofs.root_fixed[root])
      assembly->boundary_values[node] =
          static_cast<double>(dofs.node_sign[node]) * dofs.root_value[root];
  }
  for (size_t node = 0; node < model.nodes.size(); ++node) {
    const int32_t row = assembly->node_to_free[node];
    if (row < 0)
      continue;
    const double row_sign = static_cast<double>(dofs.node_sign[node]);
    reduced_rhs[row] += row_sign * full_rhs[node];
    for (const auto &entry : full_matrix[node]) {
      const int32_t column_node = entry.first;
      const int32_t column = assembly->node_to_free[column_node];
      if (column < 0) {
        rhs_offset[row] -= row_sign * entry.second * assembly->boundary_values[column_node];
      } else {
        const double value =
            row_sign * static_cast<double>(dofs.node_sign[column_node]) * entry.second;
        const Status added = AddSparseEntry(&reduced, row, column, value);
        if (added != Status::kOk)
          return added;
      }
    }
  }
  if (ValidateSparseRows(reduced) != Status::kOk)
    return Status::kNumericalNonfinite;

  assembly->free_nodes = dofs.free_roots;
  assembly->row_offsets.assign(free_count + 1, 0);
  assembly->column_indices.clear();
  assembly->values.clear();
  assembly->diagonal.assign(free_count, 0.0);
  assembly->rhs_per_amp = std::move(reduced_rhs);
  assembly->rhs_offset = std::move(rhs_offset);
  for (size_t row = 0; row < free_count; ++row) {
    assembly->row_offsets[row] = static_cast<int32_t>(assembly->values.size());
    for (const auto &entry : reduced[row]) {
      // Keep exact cancellation slots.  Device assembly retains the symbolic
      // CSR topology, and sliding-band tangent assembly must match that same
      // structure even when signed reduction makes a coefficient exactly zero.
      assembly->column_indices.push_back(entry.first);
      assembly->values.push_back(entry.second);
      if (entry.first == static_cast<int32_t>(row))
        assembly->diagonal[row] = entry.second;
    }
    if (!(assembly->diagonal[row] > 0.0) || !std::isfinite(assembly->diagonal[row]) ||
        !std::isfinite(assembly->rhs_per_amp[row]) || !std::isfinite(assembly->rhs_offset[row]))
      return Status::kAssemblyFailed;
  }
  assembly->row_offsets[free_count] = static_cast<int32_t>(assembly->values.size());
  return Status::kOk;
}

std::vector<double> ExpandAssemblySolution(const Assembly &assembly,
                                           const std::vector<double> &free_solution)
{
  std::vector<double> expanded = assembly.boundary_values;
  if (!assembly.node_to_free.empty()) {
    for (size_t node = 0; node < expanded.size(); ++node) {
      const int32_t row = assembly.node_to_free[node];
      if (row >= 0 && static_cast<size_t>(row) < free_solution.size())
        expanded[node] = static_cast<double>(assembly.node_sign[node]) * free_solution[row];
    }
  } else {
    for (size_t row = 0; row < free_solution.size(); ++row)
      expanded[assembly.free_nodes[row]] = free_solution[row];
  }
  return expanded;
}

std::vector<double> ExpandAssemblyDerivative(const Assembly &assembly,
                                             const std::vector<double> &free_solution)
{
  std::vector<double> expanded(assembly.boundary_values.size(), 0.0);
  for (size_t node = 0; node < expanded.size(); ++node) {
    const int32_t row = assembly.node_to_free.empty() ? -1 : assembly.node_to_free[node];
    if (row >= 0 && static_cast<size_t>(row) < free_solution.size())
      expanded[node] = static_cast<double>(assembly.node_sign[node]) * free_solution[row];
  }
  if (assembly.node_to_free.empty())
    for (size_t row = 0; row < free_solution.size(); ++row)
      expanded[assembly.free_nodes[row]] = free_solution[row];
  return expanded;
}

bool EvaluateNonlinearMaterial(const NonlinearModel &model, const NonlinearMaterial &material,
                               double b_magnitude, bool use_newton, double *reluctivity,
                               double *reluctivity_derivative)
{
  if (reluctivity == nullptr || reluctivity_derivative == nullptr)
    return false;
  const NonlinearBhEvaluation bh =
      material.bh_curve_index >= 0
          ? EvaluateNonlinearBh(model.bh_curves[material.bh_curve_index], b_magnitude)
          : NonlinearBhEvaluation{};
  *reluctivity = material.bh_curve_index >= 0
                     ? bh.reluctivity_m_per_h
                     : material.reluctivity_zero_m_per_h *
                           (1.0 + material.alpha_per_t2 * b_magnitude * b_magnitude);
  *reluctivity_derivative = use_newton
                                ? (material.bh_curve_index >= 0
                                       ? bh.dv_db2
                                       : material.reluctivity_zero_m_per_h * material.alpha_per_t2)
                                : 0.0;
  if (!use_newton && material.bh_curve_index >= 0)
    *reluctivity = model.bh_curves[material.bh_curve_index].cold_secant_reluctivity_m_per_h;
  return *reluctivity > 0.0 && std::isfinite(*reluctivity) &&
         std::isfinite(*reluctivity_derivative);
}

Status AssembleNonlinearNewton(const NonlinearModel &model, const std::vector<double> &a_old,
                               const std::vector<double> &circuit_currents_a, bool use_newton,
                               Assembly *assembly)
{
  if (assembly == nullptr || a_old.size() != model.nodes.size() ||
      circuit_currents_a.size() != static_cast<size_t>(model.circuit_count))
    return Status::kInvalidArgument;
  for (const double current_a : circuit_currents_a) {
    if (!std::isfinite(current_a))
      return Status::kInvalidArgument;
  }
  const Status validation = ValidateNonlinearModel(model);
  if (validation != Status::kOk)
    return validation;
  const size_t node_count = model.nodes.size();
  SparseRows jacobian(node_count);
  std::vector<double> rhs(node_count, 0.0);
  assembly->source_load_per_amp.assign(node_count, 0.0);
  assembly->source_load_per_circuit.assign(static_cast<size_t>(model.circuit_count),
                                           std::vector<double>(node_count, 0.0));

  for (const NonlinearTriangle &triangle : model.triangles) {
    const NonlinearMaterial &material = model.materials[triangle.material];
    Node p[3] = {model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
                 model.nodes[triangle.node[2]]};
    const double theta_rad =
        material.magnetization_deg * 3.141592653589793238462643383279502884 / 180.0;
    const double hcx = material.h_c_a_per_m * std::cos(theta_rad);
    const double hcy = material.h_c_a_per_m * std::sin(theta_rad);
    if (model.problem_type == MagneticProblemType::kPlanar) {
      NonlinearElementTerms terms;
      if (!BuildElementTerms(p, &terms))
        return Status::kMeshInvalid;
      double bx = 0.0, by = 0.0;
      for (int i = 0; i < 3; ++i) {
        bx += a_old[triangle.node[i]] * terms.b_x[i];
        by += a_old[triangle.node[i]] * terms.b_y[i];
      }
      double reluctivity = 0.0, reluctivity_derivative = 0.0;
      if (!EvaluateNonlinearMaterial(model, material, std::hypot(bx, by), use_newton, &reluctivity,
                                     &reluctivity_derivative))
        return Status::kNumericalNonfinite;
      for (int i = 0; i < 3; ++i) {
        const int global_i = triangle.node[i];
        const double coil = terms.area_m2 * material.source_j_per_a / 3.0;
        const double pm = terms.area_m2 * (hcx * terms.b_x[i] + hcy * terms.b_y[i]);
        if (material.circuit_index >= 0) {
          const size_t circuit = static_cast<size_t>(material.circuit_index);
          assembly->source_load_per_circuit[circuit][global_i] += coil;
          rhs[global_i] += circuit_currents_a[circuit] * coil;
        }
        rhs[global_i] += pm;
        const double b_dot_i = bx * terms.b_x[i] + by * terms.b_y[i];
        for (int j = 0; j < 3; ++j) {
          const int global_j = triangle.node[j];
          const double b_dot_j = bx * terms.b_x[j] + by * terms.b_y[j];
          const double k = terms.area_m2 * reluctivity *
                           (terms.b_x[i] * terms.b_x[j] + terms.b_y[i] * terms.b_y[j]);
          const double c =
              use_newton ? 2.0 * terms.area_m2 * reluctivity_derivative * b_dot_i * b_dot_j : 0.0;
          const Status added = AddSparseEntry(&jacobian, global_i, global_j, k + c);
          if (added != Status::kOk)
            return added;
          if (use_newton)
            rhs[global_i] += c * a_old[global_j];
        }
      }
    } else {
      const double radius[3] = {p[0].x_m, p[1].x_m, p[2].x_m};
      const double axial[3] = {p[0].y_m, p[1].y_m, p[2].y_m};
      AxisymmetricP1Element element;
      if (!BuildAxisymmetricP1Element(radius, axial, &element))
        return Status::kMeshInvalid;
      constexpr double kTwoPi = 6.283185307179586476925286766559005768;
      for (const AxisymmetricP1Point &point : element.points) {
        double br = 0.0, bz = 0.0;
        for (int local = 0; local < 3; ++local) {
          const double value = a_old[triangle.node[local]];
          br += value * point.br_basis_per_m[local];
          bz += value * point.bz_basis_per_m[local];
        }
        double reluctivity = 0.0, reluctivity_derivative = 0.0;
        if (!EvaluateNonlinearMaterial(model, material, std::hypot(br, bz), use_newton,
                                       &reluctivity, &reluctivity_derivative))
          return Status::kNumericalNonfinite;
        const double volume_weight = kTwoPi * point.radius_m * point.weight_area_m2;
        for (int i = 0; i < 3; ++i) {
          const int global_i = triangle.node[i];
          const double coil = volume_weight * material.source_j_per_a * point.shape[i];
          const double pm =
              volume_weight * (hcx * point.br_basis_per_m[i] + hcy * point.bz_basis_per_m[i]);
          if (material.circuit_index >= 0) {
            const size_t circuit = static_cast<size_t>(material.circuit_index);
            assembly->source_load_per_circuit[circuit][global_i] += coil;
            rhs[global_i] += circuit_currents_a[circuit] * coil;
          }
          rhs[global_i] += pm;
          const double b_dot_i = br * point.br_basis_per_m[i] + bz * point.bz_basis_per_m[i];
          for (int j = 0; j < 3; ++j) {
            const int global_j = triangle.node[j];
            const double b_dot_j = br * point.br_basis_per_m[j] + bz * point.bz_basis_per_m[j];
            const double k = volume_weight * reluctivity *
                             (point.br_basis_per_m[i] * point.br_basis_per_m[j] +
                              point.bz_basis_per_m[i] * point.bz_basis_per_m[j]);
            const double c =
                use_newton ? 2.0 * volume_weight * reluctivity_derivative * b_dot_i * b_dot_j : 0.0;
            const Status added = AddSparseEntry(&jacobian, global_i, global_j, k + c);
            if (added != Status::kOk)
              return added;
            if (use_newton)
              rhs[global_i] += c * a_old[global_j];
          }
        }
      }
    }
  }
  for (const AirGapElement &age : model.air_gap_elements) {
    const size_t elements = age.quad_points.size() - 1;
    if (elements == 0)
      return Status::kMeshInvalid;
    const double dt = (3.141592653589793238462643383279502884 / 180.0) * age.arc_length_deg /
                      static_cast<double>(elements);
    const double K = 2.0 * (age.outer_radius_m - age.inner_radius_m) /
                     (dt * (age.outer_radius_m + age.inner_radius_m));
    const double Ki = 1.0 / K;
    double ci = age.inner_shift, co = age.outer_shift;
    if (ci > co) {
      ci -= co;
      co = 0.0;
    } else {
      ci = 1.0 - co + ci;
      co = 1.0;
    }
    double matrix[10][10];
    if (!BuildNativeFemmAirGapMatrix(K, Ki, ci, co, matrix))
      return Status::kMeshInvalid;
    for (size_t k = 0; k < elements; ++k) {
      const size_t previous = k == 0 ? elements - 1 : k - 1;
      const size_t next = k + 1;
      const size_t next2 = k + 2 > elements ? 1 : k + 2;
      int32_t node[10] = {age.quad_points[previous].node[0], age.quad_points[k].node[0],
                          age.quad_points[k].node[1],        age.quad_points[next].node[1],
                          age.quad_points[next2].node[1],    age.quad_points[previous].node[2],
                          age.quad_points[k].node[2],        age.quad_points[k].node[3],
                          age.quad_points[next].node[3],     age.quad_points[next2].node[3]};
      double weight[10] = {age.quad_points[previous].weight[0], age.quad_points[k].weight[0],
                           age.quad_points[k].weight[1],        age.quad_points[next].weight[1],
                           age.quad_points[next2].weight[1],    age.quad_points[previous].weight[2],
                           age.quad_points[k].weight[2],        age.quad_points[k].weight[3],
                           age.quad_points[next].weight[3],     age.quad_points[next2].weight[3]};
      if (age.antiperiodic && k == 0) {
        weight[0] = -weight[0];
        weight[5] = -weight[5];
      }
      if (age.antiperiodic && k + 1 == elements) {
        weight[4] = -weight[4];
        weight[9] = -weight[9];
      }
      for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j) {
          // fkn works in a relative-permeability, centimetre-scaled system;
          // this solver's P1 assembly is SI and carries 1/mu0 explicitly.
          const Status added =
              AddSparseEntry(&jacobian, node[i], node[j],
                             matrix[i][j] * weight[i] * weight[j] * kNativeFemmAgeToSiReluctivity);
          if (added != Status::kOk)
            return added;
        }
    }
  }
  const Status sparse_validation = ValidateSparseRows(jacobian);
  if (sparse_validation != Status::kOk)
    return sparse_validation;
  return ReduceSignedSystem(model, jacobian, rhs, assembly);
}

Status AssembleNonlinearNewton(const NonlinearModel &model, const std::vector<double> &a_old,
                               double current_a, bool use_newton, Assembly *assembly)
{
  return AssembleNonlinearNewton(model, a_old, std::vector<double>{current_a}, use_newton,
                                 assembly);
}

} // namespace gpu_femm

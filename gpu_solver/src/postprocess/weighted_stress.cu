#include "postprocess.h"
#include "solvers/csr_solver.cuh"

#include <algorithm>
#include <cmath>

namespace gpu_femm
{

bool IsSelectedPostprocessMaterial(const NonlinearMaterial &material,
                                   const FrozenPostprocessOptions &options, int32_t material_index)
{
  return options.selected_group_number >= 0 ? material.group_number == options.selected_group_number
                                            : material_index == options.selected_material_label;
}

bool IsAirPostprocessMaterial(const NonlinearMaterial &material,
                              const FrozenPostprocessOptions &options, int32_t material_index)
{
  return options.air_group_number >= 0 ? material.group_number == options.air_group_number
                                       : material_index == options.air_material_label;
}

Status PrepareWeightedStressMask(const NonlinearModel &model,
                                 const FrozenPostprocessOptions &options,
                                 PreparedWeightedStressMask *prepared)
{
  if (prepared == nullptr ||
      ((options.selected_group_number < 0) &&
       (options.selected_material_label < 0 ||
        static_cast<size_t>(options.selected_material_label) >= model.materials.size())) ||
      ((options.air_group_number < 0) &&
       (options.air_material_label < 0 ||
        static_cast<size_t>(options.air_material_label) >= model.materials.size())) ||
      (options.selected_group_number >= 0 &&
       options.selected_group_number == options.air_group_number) ||
      (options.selected_group_number < 0 && options.air_group_number < 0 &&
       options.selected_material_label == options.air_material_label) ||
      !(options.mask_relative_tolerance > 0.0) || !std::isfinite(options.mask_relative_tolerance) ||
      options.max_mask_iterations <= 0)
    return Status::kInvalidArgument;

  *prepared = PreparedWeightedStressMask{};

  const Status validation = ValidateNonlinearModel(model);
  if (validation != Status::kOk)
    return validation;
  const size_t node_count = model.nodes.size();
  std::map<EdgeKey, std::vector<int32_t>> edge_elements;
  bool selected_seen = false;
  for (size_t element = 0; element < model.triangles.size(); ++element) {
    const NonlinearTriangle &triangle = model.triangles[element];
    selected_seen =
        selected_seen || IsSelectedPostprocessMaterial(model.materials[triangle.material], options,
                                                       triangle.material);
    for (int local = 0; local < 3; ++local) {
      edge_elements[CanonicalEdge(triangle.node[local], triangle.node[(local + 1) % 3])].push_back(
          static_cast<int32_t>(element));
    }
  }
  if (!selected_seen)
    return Status::kInvalidMaterial;
  // FEMM's MakeMask rejects selections that touch a non-free-space region.
  for (const auto &edge : edge_elements) {
    const std::vector<int32_t> &elements = edge.second;
    bool selected = false;
    bool other_non_air = false;
    for (const int32_t element : elements) {
      const int32_t material = model.triangles[element].material;
      const NonlinearMaterial &properties = model.materials[material];
      selected = selected || IsSelectedPostprocessMaterial(properties, options, material);
      other_non_air =
          other_non_air || (!IsSelectedPostprocessMaterial(properties, options, material) &&
                            !IsAirPostprocessMaterial(properties, options, material));
    }
    if (selected && other_non_air)
      return Status::kInvalidMaterial;
  }

  std::vector<double> fixed(node_count, std::numeric_limits<double>::quiet_NaN());
  for (const int32_t boundary : model.dirichlet_nodes) {
    if (std::find(model.symmetry_apex_nodes.begin(), model.symmetry_apex_nodes.end(), boundary) ==
        model.symmetry_apex_nodes.end())
      fixed[boundary] = 0.0;
  }
  for (const NonlinearTriangle &triangle : model.triangles) {
    const NonlinearMaterial &material = model.materials[triangle.material];
    if (IsSelectedPostprocessMaterial(material, options, triangle.material)) {
      for (const int32_t node : triangle.node) {
        if (std::isfinite(fixed[node]) && fixed[node] != 1.0)
          return Status::kBoundaryInvalid;
        fixed[node] = 1.0;
      }
    } else if (!IsAirPostprocessMaterial(material, options, triangle.material)) {
      for (const int32_t node : triangle.node) {
        if (!std::isfinite(fixed[node]))
          fixed[node] = 0.0;
      }
    }
  }

  SparseRows stiffness(node_count);
  for (const NonlinearTriangle &triangle : model.triangles) {
    Node p[3] = {model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
                 model.nodes[triangle.node[2]]};
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms))
      return Status::kMeshInvalid;
    // FEMM default WeightingScheme=0 uses sqrt(label.MaxArea) when supplied,
    // otherwise sqrt(element area).  The frozen fixture has MaxArea=-1 for
    // every label; changing mm to SI only introduces a common factor and
    // therefore leaves the Dirichlet mask solution unchanged.
    const double mask_weight = std::sqrt(terms.area_m2);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        const Status added = AddSparseEntry(
            &stiffness, triangle.node[row], triangle.node[column],
            mask_weight * terms.area_m2 *
                (terms.b_x[row] * terms.b_x[column] + terms.b_y[row] * terms.b_y[column]));
        if (added != Status::kOk)
          return added;
      }
    }
  }
  const Status sparse_validation = ValidateSparseRows(stiffness);
  if (sparse_validation != Status::kOk)
    return sparse_validation;
  if (std::all_of(fixed.begin(), fixed.end(), [](double value) { return std::isfinite(value); })) {
    prepared->fixed_mask.assign(node_count, 0.0);
    for (size_t node = 0; node < node_count; ++node)
      prepared->fixed_mask[node] = fixed[node] > 0.5 ? 1.0 : 0.0;
    return Status::kOk;
  }
  // The weighting field is a scalar geometric mask.  It is periodic across a
  // 180-degree cut regardless of the magnetic A relation.  Reuse the exact
  // cut-node pairs, force their signs to +1, and reduce the mask Laplacian in
  // that quotient space.  BuildSignedDofMap also rejects a pair whose selected
  // or air classification would impose incompatible fixed mask values.
  NonlinearModel mask_model = model;
  for (SignedNodeConstraint &constraint : mask_model.node_constraints)
    constraint.sign = 1;
  mask_model.dirichlet_nodes.clear();
  mask_model.dirichlet_a_wb_per_m.clear();
  for (size_t node = 0; node < node_count; ++node) {
    if (std::isfinite(fixed[node])) {
      mask_model.dirichlet_nodes.push_back(static_cast<int32_t>(node));
      mask_model.dirichlet_a_wb_per_m.push_back(fixed[node]);
    }
  }
  const std::vector<double> full_rhs(node_count, 0.0);
  return ReduceSignedSystem(mask_model, stiffness, full_rhs, &prepared->assembly);
}

Status SolveWeightedStressMask(const PreparedWeightedStressMask &prepared,
                               const FrozenPostprocessOptions &options, std::vector<double> *mask)
{
  if (!mask || !(options.mask_relative_tolerance > 0) ||
      !std::isfinite(options.mask_relative_tolerance) || options.max_mask_iterations <= 0)
    return Status::kInvalidArgument;
  if (!prepared.fixed_mask.empty()) {
    *mask = prepared.fixed_mask;
    return Status::kOk;
  }
  const Assembly &assembly = prepared.assembly;
  GpuCsrSolver solver;
  const Status initialize = solver.Initialize(assembly);
  if (initialize != Status::kOk)
    return initialize;
  std::vector<double> rhs(assembly.rhs_per_amp.size(), 0.0);
  for (size_t row = 0; row < rhs.size(); ++row)
    rhs[row] = assembly.rhs_per_amp[row] + assembly.rhs_offset[row];
  std::vector<double> free_mask;
  const SolveInfo info =
      solver.Solve(rhs, options.mask_relative_tolerance, options.max_mask_iterations, &free_mask);
  if (info.status != Status::kOk)
    return info.status;
  *mask = ExpandAssemblySolution(assembly, free_mask);
  // FEMM's default WeightingScheme=0 does not use the continuous harmonic
  // mask directly: it thresholds every solved node at V>0.5.
  for (double &value : *mask)
    value = value > 0.5 ? 1.0 : 0.0;
  return Status::kOk;
}

Status BuildWeightedStressMask(const NonlinearModel &model, const FrozenPostprocessOptions &options,
                               std::vector<double> *mask)
{
  if (!mask)
    return Status::kInvalidArgument;
  PreparedWeightedStressMask prepared;
  const Status status = PrepareWeightedStressMask(model, options, &prepared);
  return status == Status::kOk ? SolveWeightedStressMask(prepared, options, mask) : status;
}

} // namespace gpu_femm

#include "physics/nonlinear_topology.h"
#include "axisymmetric_p1.h"
#include "native_airgap_matrix.h"

#include <cmath>
#include <map>
#include <utility>

namespace gpu_femm
{

Status PrepareNonlinearAssembly(const NonlinearModel &model, PreparedNonlinearAssembly *prepared)
{
  if (prepared == nullptr)
    return Status::kInvalidArgument;
  *prepared = {};
  if (ValidateNonlinearModel(model) != Status::kOk)
    return Status::kInvalidArgument;
  const size_t node_count_ = model.nodes.size();
  const int circuit_count_ = model.circuit_count;
  SignedDofMap dofs;
  if (BuildSignedDofMap(node_count_, model.node_constraints, model.dirichlet_nodes,
                        model.dirichlet_a_wb_per_m, &dofs) != SignedDofStatus::kOk)
    return Status::kBoundaryInvalid;
  std::vector<int32_t> node_to_free(node_count_, -1);
  std::vector<int8_t> node_sign = dofs.node_sign;
  std::vector<bool> boundary(node_count_, false);
  std::vector<double> boundary_values(node_count_, 0.0);
  for (size_t node = 0; node < node_count_; ++node) {
    const int32_t root = dofs.node_root[node];
    node_to_free[node] = dofs.root_to_free[root];
    boundary[node] = node_to_free[node] < 0;
    if (boundary[node])
      boundary_values[node] = static_cast<double>(node_sign[node]) * dofs.root_value[root];
  }
  const size_t free_count_ = dofs.free_roots.size();
  std::vector<std::map<int32_t, int32_t>> structural(free_count_);
  const auto add_structure = [&structural, &node_to_free](int32_t row, int32_t column) {
    if (node_to_free[row] >= 0 && node_to_free[column] >= 0)
      structural[node_to_free[row]][node_to_free[column]] = -1;
  };
  for (const NonlinearTriangle &triangle : model.triangles)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        add_structure(triangle.node[i], triangle.node[j]);
  for (const AirGapElement &age : model.air_gap_elements) {
    const size_t elements = age.quad_points.size() - 1;
    for (size_t k = 0; k < elements; ++k) {
      const size_t previous = k == 0 ? elements - 1 : k - 1, next = k + 1,
                   next2 = k + 2 > elements ? 1 : k + 2;
      const int32_t node[10] = {
          age.quad_points[previous].node[0], age.quad_points[k].node[0],
          age.quad_points[k].node[1],        age.quad_points[next].node[1],
          age.quad_points[next2].node[1],    age.quad_points[previous].node[2],
          age.quad_points[k].node[2],        age.quad_points[k].node[3],
          age.quad_points[next].node[3],     age.quad_points[next2].node[3]};
      for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
          add_structure(node[i], node[j]);
    }
  }
  Assembly symbolic;
  symbolic.free_nodes = dofs.free_roots;
  symbolic.boundary_values = boundary_values;
  symbolic.node_to_free = node_to_free;
  symbolic.node_sign = node_sign;
  symbolic.row_offsets.assign(free_count_ + 1, 0);
  std::vector<int32_t> diagonal_slots(free_count_, -1);
  for (size_t row = 0; row < free_count_; ++row) {
    symbolic.row_offsets[row] = static_cast<int32_t>(symbolic.column_indices.size());
    for (const auto &entry : structural[row]) {
      const int32_t slot = static_cast<int32_t>(symbolic.column_indices.size());
      structural[row][entry.first] = slot;
      symbolic.column_indices.push_back(entry.first);
      symbolic.values.push_back(0.0);
      if (entry.first == static_cast<int32_t>(row))
        diagonal_slots[row] = slot;
    }
    if (diagonal_slots[row] < 0)
      return Status::kAssemblyFailed;
  }
  symbolic.row_offsets[free_count_] = static_cast<int32_t>(symbolic.column_indices.size());
  symbolic.diagonal.assign(free_count_, 1.0);
  symbolic.rhs_per_amp.assign(free_count_, 0.0);
  symbolic.rhs_offset.assign(free_count_, 0.0);
  const size_t nnz_ = symbolic.values.size();
  std::vector<std::vector<DeviceMatrixContributor>> contributors(nnz_);
  std::vector<std::vector<DeviceRhsContributor>> rhs_contributors(free_count_);
  std::vector<DeviceNonlinearTriangle> triangles;
  triangles.reserve(model.triangles.size());
  std::vector<double> pm_rhs(free_count_, 0.0),
      circuit_rhs(static_cast<size_t>(circuit_count_) * free_count_, 0.0);
  std::vector<double> curve_b, curve_h, curve_slope;
  std::vector<int32_t> curve_offsets(model.bh_curves.size(), -1),
      curve_counts(model.bh_curves.size(), 0);
  for (size_t curve = 0; curve < model.bh_curves.size(); ++curve) {
    curve_offsets[curve] = static_cast<int32_t>(curve_b.size());
    curve_counts[curve] = static_cast<int32_t>(model.bh_curves[curve].b_t.size());
    curve_b.insert(curve_b.end(), model.bh_curves[curve].b_t.begin(),
                   model.bh_curves[curve].b_t.end());
    curve_h.insert(curve_h.end(), model.bh_curves[curve].h_a_per_m.begin(),
                   model.bh_curves[curve].h_a_per_m.end());
    curve_slope.insert(curve_slope.end(), model.bh_curves[curve].slope.begin(),
                       model.bh_curves[curve].slope.end());
  }
  for (size_t element = 0; element < model.triangles.size(); ++element) {
    const NonlinearTriangle &triangle = model.triangles[element];
    const NonlinearMaterial &material = model.materials[triangle.material];
    Node p[3] = {model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
                 model.nodes[triangle.node[2]]};
    DeviceNonlinearTriangle device{};
    device.reluctivity_zero = material.reluctivity_zero_m_per_h;
    device.alpha = material.alpha_per_t2;
    device.curve_offset =
        material.bh_curve_index >= 0 ? curve_offsets[material.bh_curve_index] : -1;
    device.curve_count = material.bh_curve_index >= 0 ? curve_counts[material.bh_curve_index] : -1;
    device.cold_secant =
        material.bh_curve_index >= 0
            ? model.bh_curves[material.bh_curve_index].cold_secant_reluctivity_m_per_h
            : 0.0;
    device.circuit_index = material.circuit_index;
    const double theta =
                     material.magnetization_deg * 3.141592653589793238462643383279502884 / 180.0,
                 hcx = material.h_c_a_per_m * std::cos(theta),
                 hcy = material.h_c_a_per_m * std::sin(theta);
    if (model.problem_type == MagneticProblemType::kPlanar) {
      NonlinearElementTerms terms;
      if (!BuildElementTerms(p, &terms))
        return Status::kMeshInvalid;
      device.quadrature_count = 1;
      device.weight[0] = terms.area_m2;
      for (int local = 0; local < 3; ++local) {
        device.b_x[0][local] = terms.b_x[local];
        device.b_y[0][local] = terms.b_y[local];
        device.shape[0][local] = 1.0 / 3.0;
      }
    } else {
      const double radius[3] = {p[0].x_m, p[1].x_m, p[2].x_m};
      const double axial[3] = {p[0].y_m, p[1].y_m, p[2].y_m};
      AxisymmetricP1Element axisymmetric;
      if (!BuildAxisymmetricP1Element(radius, axial, &axisymmetric))
        return Status::kMeshInvalid;
      device.quadrature_count = static_cast<int32_t>(axisymmetric.points.size());
      constexpr double kTwoPi = 6.283185307179586476925286766559005768;
      for (int point = 0; point < device.quadrature_count; ++point) {
        const AxisymmetricP1Point &source = axisymmetric.points[point];
        device.weight[point] = kTwoPi * source.radius_m * source.weight_area_m2;
        for (int local = 0; local < 3; ++local) {
          device.b_x[point][local] = source.br_basis_per_m[local];
          device.b_y[point][local] = source.bz_basis_per_m[local];
          device.shape[point][local] = source.shape[local];
        }
      }
    }
    for (int i = 0; i < 3; ++i) {
      device.node[i] = triangle.node[i];
      double pm = 0.0, coil = 0.0;
      for (int point = 0; point < device.quadrature_count; ++point) {
        pm += device.weight[point] * (hcx * device.b_x[point][i] + hcy * device.b_y[point][i]);
        coil += device.weight[point] * material.source_j_per_a * device.shape[point][i];
      }
      const int32_t row = node_to_free[triangle.node[i]];
      if (row >= 0) {
        rhs_contributors[row].push_back(
            {static_cast<int32_t>(element), static_cast<int8_t>(i), node_sign[triangle.node[i]]});
        pm_rhs[row] += static_cast<double>(node_sign[triangle.node[i]]) * pm;
        if (device.circuit_index >= 0)
          circuit_rhs[static_cast<size_t>(device.circuit_index) * free_count_ + row] +=
              static_cast<double>(node_sign[triangle.node[i]]) * coil;
      }
      for (int j = 0; j < 3; ++j)
        if (node_to_free[triangle.node[i]] >= 0 && node_to_free[triangle.node[j]] >= 0)
          contributors[structural[node_to_free[triangle.node[i]]][node_to_free[triangle.node[j]]]]
              .push_back(
                  {static_cast<int32_t>(element), static_cast<int8_t>(i), static_cast<int8_t>(j),
                   static_cast<int8_t>(node_sign[triangle.node[i]] * node_sign[triangle.node[j]])});
    }
    triangles.push_back(device);
  }
  std::vector<double> age_values(nnz_, 0.0), age_rhs(free_count_, 0.0);
  for (const AirGapElement &age : model.air_gap_elements) {
    const size_t elements = age.quad_points.size() - 1;
    const double dt = (3.141592653589793238462643383279502884 / 180.0) * age.arc_length_deg /
                      static_cast<double>(elements);
    const double K = 2.0 * (age.outer_radius_m - age.inner_radius_m) /
                     (dt * (age.outer_radius_m + age.inner_radius_m)),
                 Ki = 1.0 / K;
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
      const size_t previous = k == 0 ? elements - 1 : k - 1, next = k + 1,
                   next2 = k + 2 > elements ? 1 : k + 2;
      const int32_t node[10] = {
          age.quad_points[previous].node[0], age.quad_points[k].node[0],
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
          const double value = matrix[i][j] * weight[i] * weight[j] * kNativeFemmAgeToSiReluctivity;
          const int32_t row = node_to_free[node[i]], column = node_to_free[node[j]];
          if (row >= 0 && column >= 0)
            age_values[structural[row][column]] +=
                static_cast<double>(node_sign[node[i]] * node_sign[node[j]]) * value;
          else if (row >= 0 && boundary[node[j]])
            age_rhs[row] -=
                static_cast<double>(node_sign[node[i]]) * value * boundary_values[node[j]];
        }
    }
  }
  std::vector<int32_t> matrix_offsets(nnz_ + 1, 0), rhs_offsets(free_count_ + 1, 0);
  std::vector<DeviceMatrixContributor> matrix_entries;
  std::vector<DeviceRhsContributor> rhs_entries;
  for (size_t slot = 0; slot < nnz_; ++slot) {
    matrix_offsets[slot] = static_cast<int32_t>(matrix_entries.size());
    matrix_entries.insert(matrix_entries.end(), contributors[slot].begin(),
                          contributors[slot].end());
  }
  matrix_offsets[nnz_] = static_cast<int32_t>(matrix_entries.size());
  for (size_t row = 0; row < free_count_; ++row) {
    rhs_offsets[row] = static_cast<int32_t>(rhs_entries.size());
    rhs_entries.insert(rhs_entries.end(), rhs_contributors[row].begin(),
                       rhs_contributors[row].end());
  }
  rhs_offsets[free_count_] = static_cast<int32_t>(rhs_entries.size());
  prepared->node_count = node_count_;
  prepared->circuit_count = circuit_count_;
  prepared->quadrature_stride = model.problem_type == MagneticProblemType::kPlanar ? 1 : 7;
  prepared->symbolic = std::move(symbolic);
  prepared->triangles = std::move(triangles);
  prepared->curve_b = std::move(curve_b);
  prepared->curve_h = std::move(curve_h);
  prepared->curve_slope = std::move(curve_slope);
  prepared->matrix_offsets = std::move(matrix_offsets);
  prepared->matrix_entries = std::move(matrix_entries);
  prepared->rhs_offsets = std::move(rhs_offsets);
  prepared->rhs_entries = std::move(rhs_entries);
  prepared->age_values = std::move(age_values);
  prepared->age_rhs = std::move(age_rhs);
  prepared->pm_rhs = std::move(pm_rhs);
  prepared->circuit_rhs = std::move(circuit_rhs);
  prepared->boundary_values = std::move(boundary_values);
  prepared->diagonal_slots = std::move(diagonal_slots);
  return Status::kOk;
}

} // namespace gpu_femm

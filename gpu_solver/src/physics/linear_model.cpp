#include "linear_model.h"
#include <cmath>

namespace gpu_femm
{

Status AddSparseEntry(SparseRows *rows, int32_t row, int32_t column, double value)
{
  if (rows == nullptr || row < 0 || column < 0 || static_cast<size_t>(row) >= rows->size() ||
      static_cast<size_t>(column) >= rows->size() || !std::isfinite(value))
    return Status::kNumericalNonfinite;
  double &accumulated = (*rows)[row][column];
  accumulated += value;
  return std::isfinite(accumulated) ? Status::kOk : Status::kNumericalNonfinite;
}

Status ValidateSparseRows(const SparseRows &rows)
{
  for (const auto &row : rows) {
    for (const auto &entry : row) {
      if (!std::isfinite(entry.second))
        return Status::kNumericalNonfinite;
    }
  }
  return Status::kOk;
}

Status ValidateModel(const Model &model)
{
  if (model.nodes.empty() || model.triangles.empty() || !(model.depth_m > 0.0) ||
      !std::isfinite(model.depth_m)) {
    return Status::kInvalidArgument;
  }
  if (model.dirichlet_nodes.empty() ||
      model.dirichlet_nodes.size() != model.dirichlet_a_wb_per_m.size()) {
    return Status::kBoundaryInvalid;
  }
  std::vector<bool> is_boundary(model.nodes.size(), false);
  for (size_t i = 0; i < model.dirichlet_nodes.size(); ++i) {
    const int32_t node = model.dirichlet_nodes[i];
    if (node < 0 || static_cast<size_t>(node) >= model.nodes.size() || is_boundary[node] ||
        !std::isfinite(model.dirichlet_a_wb_per_m[i])) {
      return Status::kBoundaryInvalid;
    }
    is_boundary[node] = true;
  }
  if (model.dirichlet_nodes.size() == model.nodes.size())
    return Status::kBoundaryInvalid;

  for (const Node &node : model.nodes) {
    if (!std::isfinite(node.x_m) || !std::isfinite(node.y_m)) {
      return Status::kMeshInvalid;
    }
  }
  for (const Triangle &triangle : model.triangles) {
    for (int local = 0; local < 3; ++local) {
      if (triangle.node[local] < 0 ||
          static_cast<size_t>(triangle.node[local]) >= model.nodes.size()) {
        return Status::kMeshInvalid;
      }
    }
    const Node p0 = model.nodes[triangle.node[0]];
    const Node p1 = model.nodes[triangle.node[1]];
    const Node p2 = model.nodes[triangle.node[2]];
    const double determinant =
        (p1.x_m - p0.x_m) * (p2.y_m - p0.y_m) - (p2.x_m - p0.x_m) * (p1.y_m - p0.y_m);
    if (!(determinant > 0.0) || !std::isfinite(determinant) ||
        !(triangle.reluctivity_m_per_h > 0.0) || !std::isfinite(triangle.reluctivity_m_per_h) ||
        !std::isfinite(triangle.source_j_per_a)) {
      return Status::kMeshInvalid;
    }
  }

  std::vector<std::vector<int32_t>> adjacency(model.nodes.size());
  for (const Triangle &triangle : model.triangles) {
    for (int edge = 0; edge < 3; ++edge) {
      const int32_t first = triangle.node[edge];
      const int32_t second = triangle.node[(edge + 1) % 3];
      adjacency[first].push_back(second);
      adjacency[second].push_back(first);
    }
  }
  std::vector<bool> visited(model.nodes.size(), false);
  for (size_t start = 0; start < model.nodes.size(); ++start) {
    if (visited[start])
      continue;
    bool component_has_boundary = false;
    std::vector<int32_t> pending = {static_cast<int32_t>(start)};
    visited[start] = true;
    while (!pending.empty()) {
      const int32_t node = pending.back();
      pending.pop_back();
      component_has_boundary = component_has_boundary || is_boundary[node];
      for (const int32_t neighbor : adjacency[node]) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          pending.push_back(neighbor);
        }
      }
    }
    if (!component_has_boundary)
      return Status::kBoundaryInvalid;
  }
  return Status::kOk;
}

Status Assemble(const Model &model, Assembly *assembly)
{
  const Status validation = ValidateModel(model);
  if (validation != Status::kOk)
    return validation;
  const size_t node_count = model.nodes.size();
  SparseRows stiffness(node_count);
  assembly->source_load_per_amp.assign(node_count, 0.0);

  for (const Triangle &triangle : model.triangles) {
    const Node p[3] = {model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
                       model.nodes[triangle.node[2]]};
    const double determinant = (p[1].x_m - p[0].x_m) * (p[2].y_m - p[0].y_m) -
                               (p[2].x_m - p[0].x_m) * (p[1].y_m - p[0].y_m);
    const double area = 0.5 * determinant;
    const double b[3] = {p[1].y_m - p[2].y_m, p[2].y_m - p[0].y_m, p[0].y_m - p[1].y_m};
    const double c[3] = {p[2].x_m - p[1].x_m, p[0].x_m - p[2].x_m, p[1].x_m - p[0].x_m};
    for (int i = 0; i < 3; ++i) {
      const int global_i = triangle.node[i];
      assembly->source_load_per_amp[global_i] += triangle.source_j_per_a * area / 3.0;
      for (int j = 0; j < 3; ++j) {
        const int global_j = triangle.node[j];
        const Status added = AddSparseEntry(&stiffness, global_i, global_j,
                                            triangle.reluctivity_m_per_h *
                                                (b[i] * b[j] + c[i] * c[j]) / (4.0 * area));
        if (added != Status::kOk)
          return added;
      }
    }
  }
  const Status sparse_validation = ValidateSparseRows(stiffness);
  if (sparse_validation != Status::kOk)
    return sparse_validation;
  for (const double value : assembly->source_load_per_amp) {
    if (!std::isfinite(value))
      return Status::kNumericalNonfinite;
  }

  std::vector<bool> is_boundary(node_count, false);
  assembly->boundary_values.assign(node_count, 0.0);
  for (size_t i = 0; i < model.dirichlet_nodes.size(); ++i) {
    is_boundary[model.dirichlet_nodes[i]] = true;
    assembly->boundary_values[model.dirichlet_nodes[i]] = model.dirichlet_a_wb_per_m[i];
  }
  assembly->free_nodes.clear();
  std::vector<int32_t> free_index(node_count, -1);
  for (size_t node = 0; node < node_count; ++node) {
    if (!is_boundary[node]) {
      free_index[node] = static_cast<int32_t>(assembly->free_nodes.size());
      assembly->free_nodes.push_back(static_cast<int32_t>(node));
    }
  }

  const size_t free_count = assembly->free_nodes.size();
  assembly->row_offsets.assign(free_count + 1, 0);
  assembly->column_indices.clear();
  assembly->values.clear();
  assembly->diagonal.assign(free_count, 0.0);
  assembly->rhs_per_amp.assign(free_count, 0.0);
  assembly->rhs_offset.assign(free_count, 0.0);

  for (size_t row = 0; row < free_count; ++row) {
    const int global_row = assembly->free_nodes[row];
    assembly->row_offsets[row] = static_cast<int32_t>(assembly->values.size());
    assembly->rhs_per_amp[row] = assembly->source_load_per_amp[global_row];
    for (const auto &entry : stiffness[global_row]) {
      const int32_t global_column = entry.first;
      const double value = entry.second;
      if (is_boundary[global_column]) {
        assembly->rhs_offset[row] -= value * assembly->boundary_values[global_column];
      } else if (value != 0.0) {
        const int32_t column = free_index[global_column];
        if (column < 0)
          return Status::kAssemblyFailed;
        assembly->column_indices.push_back(column);
        assembly->values.push_back(value);
        if (static_cast<size_t>(column) == row)
          assembly->diagonal[row] = value;
      }
    }
    if (!std::isfinite(assembly->rhs_per_amp[row]) || !std::isfinite(assembly->rhs_offset[row]))
      return Status::kNumericalNonfinite;
    if (!(assembly->diagonal[row] > 0.0) || !std::isfinite(assembly->diagonal[row])) {
      return Status::kAssemblyFailed;
    }
  }
  assembly->row_offsets[free_count] = static_cast<int32_t>(assembly->values.size());
  return Status::kOk;
}

} // namespace gpu_femm

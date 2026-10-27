#pragma once

#include "../core/types.h"
#include <map>

namespace gpu_femm
{

struct Assembly {
  std::vector<int32_t> row_offsets;
  std::vector<int32_t> column_indices;
  std::vector<double> values;
  std::vector<double> diagonal;
  std::vector<double> rhs_per_amp;
  std::vector<double> rhs_offset;
  std::vector<double> source_load_per_amp;
  // Nonlinear path only: source vectors are indexed by NonlinearMaterial::circuit_index.
  std::vector<std::vector<double>> source_load_per_circuit;
  std::vector<int32_t> free_nodes;
  std::vector<double> boundary_values;
  std::vector<int32_t> node_to_free;
  std::vector<int8_t> node_sign;
};

// Element traversal is deliberately the accumulation order: map insertion
// keeps each coefficient's floating-point sum deterministic while avoiding an
// O(node_count^2) host matrix for the intrinsically sparse P1 stencil.
using SparseRows = std::vector<std::map<int32_t, double>>;

Status AddSparseEntry(SparseRows *rows, int32_t row, int32_t column, double value);
Status ValidateSparseRows(const SparseRows &rows);
Status ValidateModel(const Model &model);
Status Assemble(const Model &model, Assembly *assembly);

} // namespace gpu_femm

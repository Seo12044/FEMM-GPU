#pragma once

#include "physics/nonlinear_model.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpu_femm
{

struct DeviceNonlinearTriangle {
  int32_t node[3];
  int32_t quadrature_count;
  double weight[7];
  double b_x[7][3];
  double b_y[7][3];
  double shape[7][3];
  double reluctivity_zero;
  double alpha;
  int32_t curve_offset;
  int32_t curve_count;
  double cold_secant;
  int32_t circuit_index;
};

struct DeviceMatrixContributor {
  int32_t triangle;
  int8_t local_row;
  int8_t local_column;
  int8_t sign;
};

struct DeviceRhsContributor {
  int32_t triangle;
  int8_t local_row;
  int8_t sign;
};

// Host-only topology. No CUDA allocation or numeric solution is retained here.
struct PreparedNonlinearAssembly {
  Assembly symbolic;
  size_t node_count = 0, quadrature_stride = 1;
  int circuit_count = 0;
  std::vector<DeviceNonlinearTriangle> triangles;
  std::vector<double> curve_b, curve_h, curve_slope, age_values, age_rhs, pm_rhs, circuit_rhs,
      boundary_values;
  std::vector<int32_t> matrix_offsets, rhs_offsets, diagonal_slots;
  std::vector<DeviceMatrixContributor> matrix_entries;
  std::vector<DeviceRhsContributor> rhs_entries;
};

Status PrepareNonlinearAssembly(const NonlinearModel &model, PreparedNonlinearAssembly *prepared);

} // namespace gpu_femm

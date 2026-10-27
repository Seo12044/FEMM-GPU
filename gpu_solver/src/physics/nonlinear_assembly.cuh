#pragma once

#include "core/cuda_buffer.cuh"
#include "physics/nonlinear_model.h"
#include "physics/nonlinear_topology.h"
#include "solvers/csr_solver.cuh"

#include <cstddef>
#include <vector>

namespace gpu_femm
{

class DeviceNonlinearAssemblyPlan
{
public:
  Status Initialize(const NonlinearModel &model);
  Status Initialize(PreparedNonlinearAssembly prepared);

  const Assembly &symbolic_assembly() const;
  bool initialized() const;

  Status Assemble(const std::vector<std::vector<double>> &states,
                  const std::vector<std::vector<double>> &currents, bool use_newton,
                  GpuCsrSolver *solver, double *seconds);

private:
  Status EnsureStateCapacity(size_t count);
  bool initialized_ = false;
  size_t node_count_ = 0, free_count_ = 0, nnz_ = 0, state_capacity_ = 0;
  size_t quadrature_stride_ = 1;
  int circuit_count_ = 0;
  Assembly symbolic_;
  DeviceBuffer<DeviceNonlinearTriangle> d_triangles_;
  DeviceBuffer<double> d_curve_b_, d_curve_h_, d_curve_slope_, d_base_values_, d_age_rhs_,
      d_pm_rhs_, d_circuit_rhs_, d_boundary_values_;
  DeviceBuffer<int32_t> d_matrix_offsets_, d_rhs_offsets_, d_diagonal_slots_;
  DeviceBuffer<DeviceMatrixContributor> d_matrix_entries_;
  DeviceBuffer<DeviceRhsContributor> d_rhs_entries_;
  DeviceBuffer<int> d_valid_;
  DeviceBuffer<double> d_old_, d_currents_, d_bx_, d_by_, d_nu_, d_derivative_;
  PinnedBuffer<double> h_old_, h_currents_;
  PinnedBuffer<int> h_valid_;
  std::vector<double> uploaded_currents_;
};

} // namespace gpu_femm

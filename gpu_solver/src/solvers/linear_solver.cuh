#pragma once

#include "csr_solver.cuh"

namespace gpu_femm
{

class LinearP1Solver
{
public:
  Status Initialize(Model model);

  SolveResult Solve(double current_a, double relative_tolerance = 1e-13, int max_iterations = 128);

private:
  bool initialized_ = false;
  Model model_;
  Assembly assembly_;
  GpuCsrSolver solver_;
  DeviceBuffer<Node> nodes_;
  DeviceBuffer<Triangle> triangles_;
  DeviceBuffer<double> nodal_a_;
  DeviceBuffer<double> bx_;
  DeviceBuffer<double> by_;
  DeviceBuffer<double> source_load_per_amp_;
  DeviceBuffer<double> flux_linkage_;
};

} // namespace gpu_femm

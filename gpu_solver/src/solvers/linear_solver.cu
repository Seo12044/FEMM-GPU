#include "linear_solver.cuh"
#include <cmath>
#include <utility>

namespace gpu_femm
{

__global__ void ComputeFieldKernel(int triangle_count, const Node *nodes, const Triangle *triangles,
                                   const double *nodal_a, double *bx, double *by)
{
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= triangle_count)
    return;
  const Triangle triangle = triangles[element];
  const Node p0 = nodes[triangle.node[0]];
  const Node p1 = nodes[triangle.node[1]];
  const Node p2 = nodes[triangle.node[2]];
  const double determinant =
      (p1.x_m - p0.x_m) * (p2.y_m - p0.y_m) - (p2.x_m - p0.x_m) * (p1.y_m - p0.y_m);
  const double b0 = p1.y_m - p2.y_m;
  const double b1 = p2.y_m - p0.y_m;
  const double b2 = p0.y_m - p1.y_m;
  const double c0 = p2.x_m - p1.x_m;
  const double c1 = p0.x_m - p2.x_m;
  const double c2 = p1.x_m - p0.x_m;
  const double d_a_dx = (nodal_a[triangle.node[0]] * b0 + nodal_a[triangle.node[1]] * b1 +
                         nodal_a[triangle.node[2]] * b2) /
                        determinant;
  const double d_a_dy = (nodal_a[triangle.node[0]] * c0 + nodal_a[triangle.node[1]] * c1 +
                         nodal_a[triangle.node[2]] * c2) /
                        determinant;
  bx[element] = d_a_dy;
  by[element] = -d_a_dx;
}

__global__ void ComputeFluxKernel(int node_count, const double *source_load_per_amp,
                                  const double *nodal_a, double depth_m, double *flux_linkage)
{
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  double sum = 0.0;
  for (int i = 0; i < node_count; ++i)
    sum += source_load_per_amp[i] * nodal_a[i];
  *flux_linkage = depth_m * sum;
}

Status LinearP1Solver::Initialize(Model model)
{
  model_ = std::move(model);
  Status status = Assemble(model_, &assembly_);
  if (status != Status::kOk)
    return status;
  if ((status = solver_.Initialize(assembly_)) != Status::kOk ||
      (status = nodes_.allocate(model_.nodes.size())) != Status::kOk ||
      (status = triangles_.allocate(model_.triangles.size())) != Status::kOk ||
      (status = nodal_a_.allocate(model_.nodes.size())) != Status::kOk ||
      (status = bx_.allocate(model_.triangles.size())) != Status::kOk ||
      (status = by_.allocate(model_.triangles.size())) != Status::kOk ||
      (status = source_load_per_amp_.allocate(model_.nodes.size())) != Status::kOk ||
      (status = flux_linkage_.allocate(1)) != Status::kOk) {
    return status;
  }
  if ((status = CopyToDevice(nodes_.get(), model_.nodes.data(),
                             model_.nodes.size() * sizeof(Node))) != Status::kOk ||
      (status = CopyToDevice(triangles_.get(), model_.triangles.data(),
                             model_.triangles.size() * sizeof(Triangle))) != Status::kOk ||
      (status = CopyToDevice(source_load_per_amp_.get(), assembly_.source_load_per_amp.data(),
                             assembly_.source_load_per_amp.size() * sizeof(double))) !=
          Status::kOk) {
    return status;
  }
  initialized_ = true;
  return Status::kOk;
}

SolveResult LinearP1Solver::Solve(double current_a, double relative_tolerance, int max_iterations)
{
  SolveResult result;
  result.info.status = Status::kInvalidArgument;
  if (!initialized_ || !std::isfinite(current_a))
    return result;
  std::vector<double> rhs(assembly_.free_nodes.size());
  for (size_t i = 0; i < rhs.size(); ++i) {
    rhs[i] = current_a * assembly_.rhs_per_amp[i] + assembly_.rhs_offset[i];
  }
  std::vector<double> free_solution;
  result.info = solver_.Solve(rhs, relative_tolerance, max_iterations, &free_solution);
  if (result.info.status != Status::kOk)
    return result;

  result.a_wb_per_m = assembly_.boundary_values;
  for (size_t i = 0; i < assembly_.free_nodes.size(); ++i) {
    result.a_wb_per_m[assembly_.free_nodes[i]] = free_solution[i];
  }
  Status status = CopyToDevice(nodal_a_.get(), result.a_wb_per_m.data(),
                               result.a_wb_per_m.size() * sizeof(double));
  if (status != Status::kOk) {
    result.info.status = status;
    return result;
  }
  const int threads = 128;
  const int blocks = (static_cast<int>(model_.triangles.size()) + threads - 1) / threads;
  ComputeFieldKernel<<<blocks, threads>>>(static_cast<int>(model_.triangles.size()), nodes_.get(),
                                          triangles_.get(), nodal_a_.get(), bx_.get(), by_.get());
  ComputeFluxKernel<<<1, 1>>>(static_cast<int>(model_.nodes.size()), source_load_per_amp_.get(),
                              nodal_a_.get(), model_.depth_m, flux_linkage_.get());
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
    result.info.status = Status::kInternalError;
    return result;
  }
  result.bx_t.resize(model_.triangles.size());
  result.by_t.resize(model_.triangles.size());
  if (CopyToHost(result.bx_t.data(), bx_.get(), result.bx_t.size() * sizeof(double)) !=
          Status::kOk ||
      CopyToHost(result.by_t.data(), by_.get(), result.by_t.size() * sizeof(double)) !=
          Status::kOk ||
      CopyToHost(&result.flux_linkage_wb, flux_linkage_.get(), sizeof(double)) != Status::kOk) {
    result.info.status = Status::kInternalError;
    return result;
  }
  if (!std::isfinite(result.flux_linkage_wb)) {
    result.info.status = Status::kNumericalNonfinite;
  }
  return result;
}

} // namespace gpu_femm

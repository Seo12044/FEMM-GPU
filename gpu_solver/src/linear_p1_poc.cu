#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace gpu_femm {

enum class Status : int {
  kOk = 0,
  kInputIo = 1,
  kInvalidArgument = 2,
  kUnsupportedFeature = 3,
  kMeshInvalid = 4,
  kBoundaryInvalid = 5,
  kGpuUnavailable = 6,
  kGpuAllocationFailed = 7,
  kAssemblyFailed = 8,
  kLinearSolveNotConverged = 9,
  kLinearSolveBreakdown = 10,
  kNumericalNonfinite = 11,
  kOutputIo = 12,
  kInternalError = 13,
  kNonlinearSolveNotConverged = 14,
  kInvalidMaterial = 15,
};

const char* StatusName(Status status)
{
  switch (status) {
  case Status::kOk:
    return "OK";
  case Status::kInputIo:
    return "INPUT_IO";
  case Status::kInvalidArgument:
    return "INVALID_ARGUMENT";
  case Status::kUnsupportedFeature:
    return "UNSUPPORTED_FEATURE";
  case Status::kMeshInvalid:
    return "MESH_INVALID";
  case Status::kBoundaryInvalid:
    return "BOUNDARY_INVALID";
  case Status::kGpuUnavailable:
    return "GPU_UNAVAILABLE";
  case Status::kGpuAllocationFailed:
    return "GPU_ALLOCATION_FAILED";
  case Status::kAssemblyFailed:
    return "ASSEMBLY_FAILED";
  case Status::kLinearSolveNotConverged:
    return "LINEAR_SOLVE_NOT_CONVERGED";
  case Status::kLinearSolveBreakdown:
    return "LINEAR_SOLVE_BREAKDOWN";
  case Status::kNumericalNonfinite:
    return "NUMERICAL_NONFINITE";
  case Status::kOutputIo:
    return "OUTPUT_IO";
  case Status::kInternalError:
    return "INTERNAL_ERROR";
  case Status::kNonlinearSolveNotConverged:
    return "NONLINEAR_SOLVE_NOT_CONVERGED";
  case Status::kInvalidMaterial:
    return "INVALID_MATERIAL";
  }
  return "INTERNAL_ERROR";
}

struct Node {
  double x_m;
  double y_m;
};

struct Triangle {
  int32_t node[3];
  double reluctivity_m_per_h;
  double source_j_per_a;
};

struct Model {
  std::vector<Node> nodes;
  std::vector<Triangle> triangles;
  std::vector<int32_t> dirichlet_nodes;
  std::vector<double> dirichlet_a_wb_per_m;
  double depth_m = 0.0;
};

struct SolveInfo {
  Status status = Status::kInternalError;
  int iterations = 0;
  double residual_l2 = std::numeric_limits<double>::infinity();
};

struct SolveResult {
  SolveInfo info;
  std::vector<double> a_wb_per_m;
  std::vector<double> bx_t;
  std::vector<double> by_t;
  double flux_linkage_wb = std::numeric_limits<double>::quiet_NaN();
};

template <typename T>
class DeviceBuffer {
  public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  Status allocate(size_t count)
  {
    reset();
    count_ = count;
    if (count == 0)
      return Status::kOk;
    if (cudaMalloc(&ptr_, count * sizeof(T)) != cudaSuccess) {
      ptr_ = nullptr;
      count_ = 0;
      return Status::kGpuAllocationFailed;
    }
    return Status::kOk;
  }

  void reset()
  {
    if (ptr_ != nullptr)
      cudaFree(ptr_);
    ptr_ = nullptr;
    count_ = 0;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t count() const { return count_; }

  private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

Status CopyToDevice(void* destination, const void* source, size_t bytes)
{
  if (bytes == 0)
    return Status::kOk;
  return cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) == cudaSuccess
      ? Status::kOk
      : Status::kInternalError;
}

Status CopyToHost(void* destination, const void* source, size_t bytes)
{
  if (bytes == 0)
    return Status::kOk;
  return cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost) == cudaSuccess
      ? Status::kOk
      : Status::kInternalError;
}

__global__ void DeterministicPcgKernel(
    int n, const int32_t* row_offsets, const int32_t* column_indices,
    const double* values, const double* diagonal, const double* rhs, double* x,
    double* residual, double* direction, double* preconditioned,
    double* matrix_direction, double relative_tolerance, int max_iterations,
    SolveInfo* info)
{
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;

  info->status = Status::kInternalError;
  info->iterations = 0;
  info->residual_l2 = INFINITY;
  if (n <= 0 || max_iterations < 0 || !(relative_tolerance > 0.0) || !(relative_tolerance < 1.0) || !isfinite(relative_tolerance)) {
    info->status = Status::kInvalidArgument;
    return;
  }

  double rhs_scale = 0.0;
  for (int i = 0; i < n; ++i) {
    if (!isfinite(rhs[i]) || !isfinite(diagonal[i])) {
      info->status = Status::kNumericalNonfinite;
      return;
    }
    if (!(diagonal[i] > 0.0)) {
      info->status = Status::kLinearSolveBreakdown;
      return;
    }
    x[i] = 0.0;
    rhs_scale = fmax(rhs_scale, fabs(rhs[i]));
  }
  if (rhs_scale == 0.0) {
    info->residual_l2 = 0.0;
    info->status = Status::kOk;
    return;
  }

  double rhs_norm_sq = 0.0;
  double rho = 0.0;
  for (int i = 0; i < n; ++i) {
    residual[i] = rhs[i] / rhs_scale;
    preconditioned[i] = residual[i] / diagonal[i];
    direction[i] = preconditioned[i];
    rhs_norm_sq += residual[i] * residual[i];
    rho += residual[i] * preconditioned[i];
  }

  if (!isfinite(rhs_norm_sq) || !isfinite(rho)) {
    info->status = Status::kNumericalNonfinite;
    return;
  }
  double residual_norm_sq = rhs_norm_sq;
  const double rhs_norm = sqrt(rhs_norm_sq);
  const double threshold = relative_tolerance * rhs_norm;
  info->residual_l2 = sqrt(residual_norm_sq);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    for (int row = 0; row < n; ++row) {
      double sum = 0.0;
      for (int32_t entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry) {
        sum += values[entry] * direction[column_indices[entry]];
      }
      matrix_direction[row] = sum;
    }

    double denominator = 0.0;
    for (int i = 0; i < n; ++i)
      denominator += direction[i] * matrix_direction[i];
    if (!isfinite(denominator) || !isfinite(rho)) {
      info->status = Status::kNumericalNonfinite;
      return;
    }
    if (!(denominator > 0.0) || !(rho > 0.0)) {
      info->status = Status::kLinearSolveBreakdown;
      return;
    }

    const double alpha = rho / denominator;
    residual_norm_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      x[i] += alpha * direction[i];
      residual[i] -= alpha * matrix_direction[i];
      residual_norm_sq += residual[i] * residual[i];
    }

    info->iterations = iteration + 1;
    if (!isfinite(residual_norm_sq)) {
      info->status = Status::kNumericalNonfinite;
      return;
    }
    info->residual_l2 = sqrt(residual_norm_sq);
    if (!isfinite(info->residual_l2)) {
      info->status = Status::kNumericalNonfinite;
      return;
    }
    if (info->residual_l2 <= threshold) {
      double true_residual_norm_sq = 0.0;
      for (int row = 0; row < n; ++row) {
        double matrix_solution = 0.0;
        for (int32_t entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry) {
          matrix_solution += values[entry] * x[column_indices[entry]];
        }
        residual[row] = rhs[row] / rhs_scale - matrix_solution;
        true_residual_norm_sq += residual[row] * residual[row];
      }
      if (!isfinite(true_residual_norm_sq)) {
        info->status = Status::kNumericalNonfinite;
        return;
      }
      info->residual_l2 = sqrt(true_residual_norm_sq);
      if (info->residual_l2 <= threshold) {
        for (int i = 0; i < n; ++i) {
          x[i] *= rhs_scale;
          if (!isfinite(x[i])) {
            info->status = Status::kNumericalNonfinite;
            return;
          }
        }
        info->residual_l2 *= rhs_scale;
        if (!isfinite(info->residual_l2)) {
          info->status = Status::kNumericalNonfinite;
          return;
        }
        info->status = Status::kOk;
        return;
      }
      rho = 0.0;
      for (int i = 0; i < n; ++i) {
        preconditioned[i] = residual[i] / diagonal[i];
        direction[i] = preconditioned[i];
        rho += residual[i] * preconditioned[i];
      }
      if (!isfinite(rho) || !(rho > 0.0)) {
        info->status = Status::kLinearSolveBreakdown;
        return;
      }
      continue;
    }

    double next_rho = 0.0;
    for (int i = 0; i < n; ++i) {
      preconditioned[i] = residual[i] / diagonal[i];
      next_rho += residual[i] * preconditioned[i];
    }
    if (!isfinite(next_rho) || !(next_rho > 0.0)) {
      info->status = Status::kLinearSolveBreakdown;
      return;
    }
    const double beta = next_rho / rho;
    for (int i = 0; i < n; ++i) {
      direction[i] = preconditioned[i] + beta * direction[i];
    }
    rho = next_rho;
  }

  double true_residual_norm_sq = 0.0;
  for (int row = 0; row < n; ++row) {
    double matrix_solution = 0.0;
    for (int32_t entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry) {
      matrix_solution += values[entry] * x[column_indices[entry]];
    }
    const double true_residual = rhs[row] / rhs_scale - matrix_solution;
    true_residual_norm_sq += true_residual * true_residual;
  }
  if (!isfinite(true_residual_norm_sq)) {
    info->status = Status::kNumericalNonfinite;
    return;
  }
  for (int i = 0; i < n; ++i) {
    x[i] *= rhs_scale;
    if (!isfinite(x[i])) {
      info->status = Status::kNumericalNonfinite;
      return;
    }
  }
  info->residual_l2 = sqrt(true_residual_norm_sq) * rhs_scale;
  if (!isfinite(info->residual_l2)) {
    info->status = Status::kNumericalNonfinite;
    return;
  }
  info->status = Status::kLinearSolveNotConverged;
}

__global__ void ComputeFieldKernel(
    int triangle_count, const Node* nodes, const Triangle* triangles,
    const double* nodal_a, double* bx, double* by)
{
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= triangle_count)
    return;
  const Triangle triangle = triangles[element];
  const Node p0 = nodes[triangle.node[0]];
  const Node p1 = nodes[triangle.node[1]];
  const Node p2 = nodes[triangle.node[2]];
  const double determinant = (p1.x_m - p0.x_m) * (p2.y_m - p0.y_m) - (p2.x_m - p0.x_m) * (p1.y_m - p0.y_m);
  const double b0 = p1.y_m - p2.y_m;
  const double b1 = p2.y_m - p0.y_m;
  const double b2 = p0.y_m - p1.y_m;
  const double c0 = p2.x_m - p1.x_m;
  const double c1 = p0.x_m - p2.x_m;
  const double c2 = p1.x_m - p0.x_m;
  const double d_a_dx = (nodal_a[triangle.node[0]] * b0 + nodal_a[triangle.node[1]] * b1 + nodal_a[triangle.node[2]] * b2) / determinant;
  const double d_a_dy = (nodal_a[triangle.node[0]] * c0 + nodal_a[triangle.node[1]] * c1 + nodal_a[triangle.node[2]] * c2) / determinant;
  bx[element] = d_a_dy;
  by[element] = -d_a_dx;
}

__global__ void ComputeFluxKernel(
    int node_count, const double* source_load_per_amp, const double* nodal_a,
    double depth_m, double* flux_linkage)
{
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  double sum = 0.0;
  for (int i = 0; i < node_count; ++i)
    sum += source_load_per_amp[i] * nodal_a[i];
  *flux_linkage = depth_m * sum;
}

struct Assembly {
  std::vector<int32_t> row_offsets;
  std::vector<int32_t> column_indices;
  std::vector<double> values;
  std::vector<double> diagonal;
  std::vector<double> rhs_per_amp;
  std::vector<double> rhs_offset;
  std::vector<double> source_load_per_amp;
  std::vector<int32_t> free_nodes;
  std::vector<double> boundary_values;
};

Status ValidateModel(const Model& model)
{
  if (model.nodes.empty() || model.triangles.empty() || !(model.depth_m > 0.0) || !std::isfinite(model.depth_m)) {
    return Status::kInvalidArgument;
  }
  if (model.dirichlet_nodes.empty() || model.dirichlet_nodes.size() != model.dirichlet_a_wb_per_m.size()) {
    return Status::kBoundaryInvalid;
  }
  std::vector<bool> is_boundary(model.nodes.size(), false);
  for (size_t i = 0; i < model.dirichlet_nodes.size(); ++i) {
    const int32_t node = model.dirichlet_nodes[i];
    if (node < 0 || static_cast<size_t>(node) >= model.nodes.size() || is_boundary[node] || !std::isfinite(model.dirichlet_a_wb_per_m[i])) {
      return Status::kBoundaryInvalid;
    }
    is_boundary[node] = true;
  }
  if (model.dirichlet_nodes.size() == model.nodes.size())
    return Status::kBoundaryInvalid;

  for (const Node& node : model.nodes) {
    if (!std::isfinite(node.x_m) || !std::isfinite(node.y_m)) {
      return Status::kMeshInvalid;
    }
  }
  for (const Triangle& triangle : model.triangles) {
    for (int local = 0; local < 3; ++local) {
      if (triangle.node[local] < 0 || static_cast<size_t>(triangle.node[local]) >= model.nodes.size()) {
        return Status::kMeshInvalid;
      }
    }
    const Node p0 = model.nodes[triangle.node[0]];
    const Node p1 = model.nodes[triangle.node[1]];
    const Node p2 = model.nodes[triangle.node[2]];
    const double determinant = (p1.x_m - p0.x_m) * (p2.y_m - p0.y_m) - (p2.x_m - p0.x_m) * (p1.y_m - p0.y_m);
    if (!(determinant > 0.0) || !std::isfinite(determinant) || !(triangle.reluctivity_m_per_h > 0.0) || !std::isfinite(triangle.reluctivity_m_per_h) || !std::isfinite(triangle.source_j_per_a)) {
      return Status::kMeshInvalid;
    }
  }

  std::vector<std::vector<int32_t>> adjacency(model.nodes.size());
  for (const Triangle& triangle : model.triangles) {
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
    std::vector<int32_t> pending = { static_cast<int32_t>(start) };
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

Status Assemble(const Model& model, Assembly* assembly)
{
  const Status validation = ValidateModel(model);
  if (validation != Status::kOk)
    return validation;
  const size_t node_count = model.nodes.size();
  std::vector<double> stiffness(node_count * node_count, 0.0);
  assembly->source_load_per_amp.assign(node_count, 0.0);

  for (const Triangle& triangle : model.triangles) {
    const Node p[3] = { model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
      model.nodes[triangle.node[2]] };
    const double determinant = (p[1].x_m - p[0].x_m) * (p[2].y_m - p[0].y_m) - (p[2].x_m - p[0].x_m) * (p[1].y_m - p[0].y_m);
    const double area = 0.5 * determinant;
    const double b[3] = { p[1].y_m - p[2].y_m, p[2].y_m - p[0].y_m,
      p[0].y_m - p[1].y_m };
    const double c[3] = { p[2].x_m - p[1].x_m, p[0].x_m - p[2].x_m,
      p[1].x_m - p[0].x_m };
    for (int i = 0; i < 3; ++i) {
      const int global_i = triangle.node[i];
      assembly->source_load_per_amp[global_i] += triangle.source_j_per_a * area / 3.0;
      for (int j = 0; j < 3; ++j) {
        const int global_j = triangle.node[j];
        stiffness[global_i * node_count + global_j] += triangle.reluctivity_m_per_h * (b[i] * b[j] + c[i] * c[j]) / (4.0 * area);
      }
    }
  }
  for (const double value : stiffness) {
    if (!std::isfinite(value))
      return Status::kNumericalNonfinite;
  }
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
  for (size_t node = 0; node < node_count; ++node) {
    if (!is_boundary[node])
      assembly->free_nodes.push_back(static_cast<int32_t>(node));
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
    for (size_t boundary = 0; boundary < node_count; ++boundary) {
      if (is_boundary[boundary]) {
        assembly->rhs_offset[row] -= stiffness[global_row * node_count + boundary] * assembly->boundary_values[boundary];
      }
    }
    if (!std::isfinite(assembly->rhs_per_amp[row]) || !std::isfinite(assembly->rhs_offset[row])) {
      return Status::kNumericalNonfinite;
    }
    for (size_t column = 0; column < free_count; ++column) {
      const int global_column = assembly->free_nodes[column];
      const double value = stiffness[global_row * node_count + global_column];
      if (value != 0.0) {
        if (!std::isfinite(value))
          return Status::kNumericalNonfinite;
        assembly->column_indices.push_back(static_cast<int32_t>(column));
        assembly->values.push_back(value);
        if (row == column)
          assembly->diagonal[row] = value;
      }
    }
    if (!(assembly->diagonal[row] > 0.0) || !std::isfinite(assembly->diagonal[row])) {
      return Status::kAssemblyFailed;
    }
  }
  assembly->row_offsets[free_count] = static_cast<int32_t>(assembly->values.size());
  return Status::kOk;
}

class GpuCsrSolver {
  public:
  Status Initialize(const Assembly& assembly)
  {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
      return Status::kGpuUnavailable;
    }
    n_ = static_cast<int>(assembly.free_nodes.size());
    if (n_ <= 0 || assembly.row_offsets.size() != static_cast<size_t>(n_ + 1) || assembly.diagonal.size() != static_cast<size_t>(n_) || assembly.row_offsets.front() != 0 || assembly.row_offsets.back() != static_cast<int32_t>(assembly.values.size()) || assembly.column_indices.size() != assembly.values.size()) {
      return Status::kInvalidArgument;
    }
    for (int row = 0; row < n_; ++row) {
      if (assembly.row_offsets[row] > assembly.row_offsets[row + 1] || !std::isfinite(assembly.diagonal[row])) {
        return Status::kInvalidArgument;
      }
    }
    for (size_t entry = 0; entry < assembly.values.size(); ++entry) {
      if (assembly.column_indices[entry] < 0 || assembly.column_indices[entry] >= n_ || !std::isfinite(assembly.values[entry])) {
        return Status::kInvalidArgument;
      }
    }
    Status status = Status::kOk;
    if ((status = row_offsets_.allocate(assembly.row_offsets.size())) != Status::kOk || (status = column_indices_.allocate(assembly.column_indices.size())) != Status::kOk || (status = values_.allocate(assembly.values.size())) != Status::kOk || (status = diagonal_.allocate(assembly.diagonal.size())) != Status::kOk || (status = rhs_.allocate(n_)) != Status::kOk || (status = solution_.allocate(n_)) != Status::kOk || (status = residual_.allocate(n_)) != Status::kOk || (status = direction_.allocate(n_)) != Status::kOk || (status = preconditioned_.allocate(n_)) != Status::kOk || (status = matrix_direction_.allocate(n_)) != Status::kOk || (status = info_.allocate(1)) != Status::kOk) {
      return status;
    }
    if ((status = CopyToDevice(row_offsets_.get(), assembly.row_offsets.data(),
             assembly.row_offsets.size() * sizeof(int32_t)))
            != Status::kOk
        || (status = CopyToDevice(column_indices_.get(), assembly.column_indices.data(),
                assembly.column_indices.size() * sizeof(int32_t)))
            != Status::kOk
        || (status = CopyToDevice(values_.get(), assembly.values.data(),
                assembly.values.size() * sizeof(double)))
            != Status::kOk
        || (status = CopyToDevice(diagonal_.get(), assembly.diagonal.data(),
                assembly.diagonal.size() * sizeof(double)))
            != Status::kOk) {
      return status;
    }
    return Status::kOk;
  }

  SolveInfo Solve(const std::vector<double>& rhs, double relative_tolerance,
      int max_iterations, std::vector<double>* solution)
  {
    SolveInfo result;
    result.status = Status::kInvalidArgument;
    if (rhs.size() != static_cast<size_t>(n_) || solution == nullptr)
      return result;
    Status copy_status = CopyToDevice(rhs_.get(), rhs.data(), rhs.size() * sizeof(double));
    if (copy_status != Status::kOk) {
      result.status = copy_status;
      return result;
    }
    DeterministicPcgKernel<<<1, 1>>>(
        n_, row_offsets_.get(), column_indices_.get(), values_.get(), diagonal_.get(),
        rhs_.get(), solution_.get(), residual_.get(), direction_.get(),
        preconditioned_.get(), matrix_direction_.get(), relative_tolerance,
        max_iterations, info_.get());
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      result.status = Status::kInternalError;
      return result;
    }
    if (CopyToHost(&result, info_.get(), sizeof(result)) != Status::kOk) {
      result.status = Status::kInternalError;
      return result;
    }
    solution->resize(n_);
    if (CopyToHost(solution->data(), solution_.get(), n_ * sizeof(double)) != Status::kOk) {
      result.status = Status::kInternalError;
    }
    return result;
  }

  private:
  int n_ = 0;
  DeviceBuffer<int32_t> row_offsets_;
  DeviceBuffer<int32_t> column_indices_;
  DeviceBuffer<double> values_;
  DeviceBuffer<double> diagonal_;
  DeviceBuffer<double> rhs_;
  DeviceBuffer<double> solution_;
  DeviceBuffer<double> residual_;
  DeviceBuffer<double> direction_;
  DeviceBuffer<double> preconditioned_;
  DeviceBuffer<double> matrix_direction_;
  DeviceBuffer<SolveInfo> info_;
};

class LinearP1FixtureSolver {
  public:
  Status Initialize(Model model)
  {
    model_ = std::move(model);
    Status status = Assemble(model_, &assembly_);
    if (status != Status::kOk)
      return status;
    if ((status = solver_.Initialize(assembly_)) != Status::kOk || (status = nodes_.allocate(model_.nodes.size())) != Status::kOk || (status = triangles_.allocate(model_.triangles.size())) != Status::kOk || (status = nodal_a_.allocate(model_.nodes.size())) != Status::kOk || (status = bx_.allocate(model_.triangles.size())) != Status::kOk || (status = by_.allocate(model_.triangles.size())) != Status::kOk || (status = source_load_per_amp_.allocate(model_.nodes.size())) != Status::kOk || (status = flux_linkage_.allocate(1)) != Status::kOk) {
      return status;
    }
    if ((status = CopyToDevice(nodes_.get(), model_.nodes.data(),
             model_.nodes.size() * sizeof(Node)))
            != Status::kOk
        || (status = CopyToDevice(triangles_.get(), model_.triangles.data(),
                model_.triangles.size() * sizeof(Triangle)))
            != Status::kOk
        || (status = CopyToDevice(source_load_per_amp_.get(),
                assembly_.source_load_per_amp.data(),
                assembly_.source_load_per_amp.size() * sizeof(double)))
            != Status::kOk) {
      return status;
    }
    initialized_ = true;
    return Status::kOk;
  }

  SolveResult Solve(double current_a, double relative_tolerance = 1e-13,
      int max_iterations = 128)
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
    ComputeFieldKernel<<<blocks, threads>>>(
        static_cast<int>(model_.triangles.size()), nodes_.get(), triangles_.get(),
        nodal_a_.get(), bx_.get(), by_.get());
    ComputeFluxKernel<<<1, 1>>>(
        static_cast<int>(model_.nodes.size()), source_load_per_amp_.get(),
        nodal_a_.get(), model_.depth_m, flux_linkage_.get());
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      result.info.status = Status::kInternalError;
      return result;
    }
    result.bx_t.resize(model_.triangles.size());
    result.by_t.resize(model_.triangles.size());
    if (CopyToHost(result.bx_t.data(), bx_.get(), result.bx_t.size() * sizeof(double)) != Status::kOk || CopyToHost(result.by_t.data(), by_.get(), result.by_t.size() * sizeof(double)) != Status::kOk || CopyToHost(&result.flux_linkage_wb, flux_linkage_.get(), sizeof(double)) != Status::kOk) {
      result.info.status = Status::kInternalError;
      return result;
    }
    if (!std::isfinite(result.flux_linkage_wb)) {
      result.info.status = Status::kNumericalNonfinite;
    }
    return result;
  }

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

struct NonlinearBhEvaluation {
  double h_a_per_m = 0.0;
  double dh_db = 0.0;
  double reluctivity_m_per_h = 0.0;
  double dv_db2 = 0.0;
};

struct NonlinearBhCurve {
  std::vector<double> b_t;
  std::vector<double> h_a_per_m;
  std::vector<double> slope;
  double cold_secant_reluctivity_m_per_h = 0.0;
  int smoothing_passes = 0;
};

bool BuildNonlinearBhSpline(NonlinearBhCurve* curve)
{
  const int n = static_cast<int>(curve->b_t.size());
  if (n < 2 || curve->h_a_per_m.size() != curve->b_t.size()
      || curve->b_t.front() != 0.0 || curve->h_a_per_m.front() != 0.0)
    return false;
  if (!(curve->cold_secant_reluctivity_m_per_h > 0.0))
    curve->cold_secant_reluctivity_m_per_h = curve->h_a_per_m[1] / curve->b_t[1];
  for (int i = 1; i < n; ++i) {
    if (!(curve->b_t[i] > curve->b_t[i - 1]) || curve->h_a_per_m[i] < 0.0
        || !std::isfinite(curve->b_t[i]) || !std::isfinite(curve->h_a_per_m[i]))
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
    rhs[i] = 6.0 * (curve->h_a_per_m[i] - curve->h_a_per_m[i - 1]) / (left * left)
        + 6.0 * (curve->h_a_per_m[i + 1] - curve->h_a_per_m[i]) / (right * right);
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
    const double c1 = -2.0 * (2.0 * d0 * length + d1 * length + 3.0 * h0 - 3.0 * h1)
        / (length * length);
    const double c2 = 3.0 * (d0 * length + d1 * length + 2.0 * h0 - 2.0 * h1)
        / (length * length * length);
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

NonlinearBhEvaluation EvaluateNonlinearBh(const NonlinearBhCurve& curve, double field_b)
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
        result.h_a_per_m = (1.0 - 3.0 * z2 + 2.0 * z2 * z) * curve.h_a_per_m[i]
            + z * (1.0 - 2.0 * z + z2) * length * curve.slope[i]
            + z2 * (3.0 - 2.0 * z) * curve.h_a_per_m[i + 1]
            + z2 * (z - 1.0) * length * curve.slope[i + 1];
        result.dh_db = 6.0 * z * (z - 1.0) * curve.h_a_per_m[i] / length
            + (1.0 - 4.0 * z + 3.0 * z2) * curve.slope[i]
            + 6.0 * z * (1.0 - z) * curve.h_a_per_m[i + 1] / length
            + z * (3.0 * z - 2.0) * curve.slope[i + 1];
        break;
      }
    }
  }
  result.reluctivity_m_per_h = result.h_a_per_m / b;
  result.dv_db2 = 0.5 * (result.dh_db / (b * b) - result.h_a_per_m / (b * b * b));
  return result;
}

bool ReadNonlinearBhTable(const std::string& path, NonlinearBhCurve* curve)
{
  std::ifstream input(path);
  std::string header;
  if (!input || !std::getline(input, header) || header.find("H (A_per_meter)") == std::string::npos
      || header.find("B (tesla)") == std::string::npos)
    return false;
  double h = 0.0, b = 0.0;
  while (input >> h >> b) {
    curve->h_a_per_m.push_back(h);
    curve->b_t.push_back(b);
  }
  return input.eof() && BuildNonlinearBhSpline(curve);
}

// Package 3A keeps nonlinear assembly on the host for now, but routes every
// Newton correction through the existing deterministic CUDA CSR PCG backend.
// Each element explicitly names a material; an omitted label is a hard error.
struct NonlinearMaterial {
  // v(B) = v0 * (1 + alpha * |B|^2).  alpha==0 is a linear material.
  double reluctivity_zero_m_per_h = 0.0;
  double alpha_per_t2 = 0.0;
  double source_j_per_a = 0.0;
  double h_c_a_per_m = 0.0;
  double magnetization_deg = 0.0;
  int32_t bh_curve_index = -1;
};

struct NonlinearTriangle {
  int32_t node[3];
  int32_t material = -1;
};

struct NonlinearModel {
  std::vector<Node> nodes;
  std::vector<NonlinearMaterial> materials;
  std::vector<NonlinearBhCurve> bh_curves;
  std::vector<NonlinearTriangle> triangles;
  std::vector<int32_t> dirichlet_nodes;
  std::vector<double> dirichlet_a_wb_per_m;
  double depth_m = 0.0;
};

struct NonlinearOptions {
  double relative_tolerance = 1e-12;
  int max_newton_iterations = 32;
  int max_linear_iterations = 128;
  double linear_relative_tolerance = 1e-13;
};

struct NonlinearSolveResult {
  SolveInfo info;
  std::vector<double> a_wb_per_m;
  std::vector<double> bx_t;
  std::vector<double> by_t;
  std::vector<double> residual_history;
  double flux_linkage_wb = std::numeric_limits<double>::quiet_NaN();
};

struct NonlinearElementTerms {
  double area_m2 = 0.0;
  double b_x[3] = {}; // curl basis x component: dN/dy
  double b_y[3] = {}; // curl basis y component: -dN/dx
};

bool BuildElementTerms(const Node p[3], NonlinearElementTerms* terms)
{
  const double determinant = (p[1].x_m - p[0].x_m) * (p[2].y_m - p[0].y_m)
      - (p[2].x_m - p[0].x_m) * (p[1].y_m - p[0].y_m);
  if (!(determinant > 0.0) || !std::isfinite(determinant))
    return false;
  terms->area_m2 = 0.5 * determinant;
  const double dx[3] = { p[1].y_m - p[2].y_m, p[2].y_m - p[0].y_m,
    p[0].y_m - p[1].y_m };
  const double dy[3] = { p[2].x_m - p[1].x_m, p[0].x_m - p[2].x_m,
    p[1].x_m - p[0].x_m };
  for (int i = 0; i < 3; ++i) {
    terms->b_x[i] = dy[i] / determinant;
    terms->b_y[i] = -dx[i] / determinant;
  }
  return true;
}

Status ValidateNonlinearModel(const NonlinearModel& model)
{
  if (model.nodes.empty() || model.triangles.empty() || model.materials.empty()
      || !(model.depth_m > 0.0) || !std::isfinite(model.depth_m)
      || model.dirichlet_nodes.empty()
      || model.dirichlet_nodes.size() != model.dirichlet_a_wb_per_m.size()) {
    return Status::kInvalidArgument;
  }
  std::vector<bool> boundary(model.nodes.size(), false);
  for (size_t i = 0; i < model.dirichlet_nodes.size(); ++i) {
    const int32_t node = model.dirichlet_nodes[i];
    if (node < 0 || static_cast<size_t>(node) >= model.nodes.size() || boundary[node]
        || !std::isfinite(model.dirichlet_a_wb_per_m[i]))
      return Status::kBoundaryInvalid;
    boundary[node] = true;
  }
  if (model.dirichlet_nodes.size() == model.nodes.size())
    return Status::kBoundaryInvalid;
  for (const Node& node : model.nodes) {
    if (!std::isfinite(node.x_m) || !std::isfinite(node.y_m))
      return Status::kMeshInvalid;
  }
  for (const NonlinearMaterial& material : model.materials) {
    if (!(material.reluctivity_zero_m_per_h > 0.0) || material.alpha_per_t2 < 0.0
        || !std::isfinite(material.reluctivity_zero_m_per_h)
        || !std::isfinite(material.alpha_per_t2) || !std::isfinite(material.source_j_per_a)
        || !std::isfinite(material.h_c_a_per_m)
        || !std::isfinite(material.magnetization_deg))
      return Status::kInvalidMaterial;
    if (material.bh_curve_index < -1
        || (material.bh_curve_index >= 0
            && static_cast<size_t>(material.bh_curve_index) >= model.bh_curves.size()))
      return Status::kInvalidMaterial;
    if (material.bh_curve_index >= 0) {
      const NonlinearBhCurve& curve = model.bh_curves[material.bh_curve_index];
      if (curve.b_t.size() < 2 || curve.h_a_per_m.size() != curve.b_t.size()
          || curve.slope.size() != curve.b_t.size()
          || !(curve.cold_secant_reluctivity_m_per_h > 0.0))
        return Status::kInvalidMaterial;
    }
  }
  for (const NonlinearTriangle& triangle : model.triangles) {
    if (triangle.material < 0 || static_cast<size_t>(triangle.material) >= model.materials.size())
      return Status::kInvalidMaterial;
    Node p[3];
    for (int local = 0; local < 3; ++local) {
      if (triangle.node[local] < 0 || static_cast<size_t>(triangle.node[local]) >= model.nodes.size())
        return Status::kMeshInvalid;
      p[local] = model.nodes[triangle.node[local]];
    }
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms))
      return Status::kMeshInvalid;
  }
  return Status::kOk;
}

Status AssembleNonlinearNewton(const NonlinearModel& model,
    const std::vector<double>& a_old, double current_a, bool use_newton,
    Assembly* assembly)
{
  if (assembly == nullptr || a_old.size() != model.nodes.size() || !std::isfinite(current_a))
    return Status::kInvalidArgument;
  const Status validation = ValidateNonlinearModel(model);
  if (validation != Status::kOk)
    return validation;
  const size_t node_count = model.nodes.size();
  std::vector<double> jacobian(node_count * node_count, 0.0);
  std::vector<double> rhs(node_count, 0.0);
  assembly->source_load_per_amp.assign(node_count, 0.0);

  for (const NonlinearTriangle& triangle : model.triangles) {
    const NonlinearMaterial& material = model.materials[triangle.material];
    Node p[3] = { model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
      model.nodes[triangle.node[2]] };
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms))
      return Status::kMeshInvalid;
    double bx = 0.0;
    double by = 0.0;
    for (int i = 0; i < 3; ++i) {
      bx += a_old[triangle.node[i]] * terms.b_x[i];
      by += a_old[triangle.node[i]] * terms.b_y[i];
    }
    const double b_magnitude = std::hypot(bx, by);
    const NonlinearBhEvaluation bh = material.bh_curve_index >= 0
        ? EvaluateNonlinearBh(model.bh_curves[material.bh_curve_index], b_magnitude)
        : NonlinearBhEvaluation {};
    double reluctivity = material.bh_curve_index >= 0 ? bh.reluctivity_m_per_h
        : material.reluctivity_zero_m_per_h
            * (1.0 + material.alpha_per_t2 * b_magnitude * b_magnitude);
    // The Newton term uses d(v)/d(B^2), so a v0*(1+alpha*B^2)
    // material contributes v0*alpha without an extra |B| factor.
    const double reluctivity_derivative = use_newton
        ? (material.bh_curve_index >= 0 ? bh.dv_db2
                                        : material.reluctivity_zero_m_per_h * material.alpha_per_t2)
        : 0.0;
    if (!use_newton && material.bh_curve_index >= 0) {
      const NonlinearBhCurve& curve = model.bh_curves[material.bh_curve_index];
      reluctivity = curve.cold_secant_reluctivity_m_per_h;
    }
    if (!(reluctivity > 0.0) || !std::isfinite(reluctivity)
        || !std::isfinite(reluctivity_derivative))
      return Status::kNumericalNonfinite;
    const double theta_rad = material.magnetization_deg * 3.141592653589793238462643383279502884 / 180.0;
    const double hcx = material.h_c_a_per_m * std::cos(theta_rad);
    const double hcy = material.h_c_a_per_m * std::sin(theta_rad);
    for (int i = 0; i < 3; ++i) {
      const int global_i = triangle.node[i];
      const double coil = terms.area_m2 * material.source_j_per_a / 3.0;
      const double pm = terms.area_m2 * (hcx * terms.b_x[i] + hcy * terms.b_y[i]);
      assembly->source_load_per_amp[global_i] += coil;
      rhs[global_i] += current_a * coil + pm;
      const double b_dot_i = bx * terms.b_x[i] + by * terms.b_y[i];
      for (int j = 0; j < 3; ++j) {
        const int global_j = triangle.node[j];
        const double b_dot_j = bx * terms.b_x[j] + by * terms.b_y[j];
        const double k = terms.area_m2 * reluctivity
            * (terms.b_x[i] * terms.b_x[j] + terms.b_y[i] * terms.b_y[j]);
        const double c = use_newton ? 2.0 * terms.area_m2 * reluctivity_derivative
                * b_dot_i * b_dot_j
                                    : 0.0;
        jacobian[global_i * node_count + global_j] += k + c;
        if (use_newton)
          rhs[global_i] += c * a_old[global_j];
      }
    }
  }
  for (const double value : jacobian) {
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
  for (size_t node = 0; node < node_count; ++node) {
    if (!is_boundary[node])
      assembly->free_nodes.push_back(static_cast<int32_t>(node));
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
    assembly->rhs_per_amp[row] = rhs[global_row];
    for (size_t boundary = 0; boundary < node_count; ++boundary) {
      if (is_boundary[boundary])
        assembly->rhs_offset[row] -= jacobian[global_row * node_count + boundary]
            * assembly->boundary_values[boundary];
    }
    for (size_t column = 0; column < free_count; ++column) {
      const double value = jacobian[global_row * node_count + assembly->free_nodes[column]];
      if (value != 0.0) {
        assembly->column_indices.push_back(static_cast<int32_t>(column));
        assembly->values.push_back(value);
        if (row == column)
          assembly->diagonal[row] = value;
      }
    }
    if (!(assembly->diagonal[row] > 0.0) || !std::isfinite(assembly->diagonal[row])
        || !std::isfinite(assembly->rhs_per_amp[row]) || !std::isfinite(assembly->rhs_offset[row]))
      return Status::kAssemblyFailed;
  }
  assembly->row_offsets[free_count] = static_cast<int32_t>(assembly->values.size());
  return Status::kOk;
}

class NonlinearP1FixtureSolver {
  public:
  Status Initialize(NonlinearModel model)
  {
    const Status status = ValidateNonlinearModel(model);
    if (status != Status::kOk)
      return status;
    model_ = std::move(model);
    initialized_ = true;
    return Status::kOk;
  }

  NonlinearSolveResult Solve(double current_a, const NonlinearOptions& options = {},
      const std::vector<double>* warm_start = nullptr)
  {
    NonlinearSolveResult result;
    result.info.status = Status::kInvalidArgument;
    if (!initialized_ || !std::isfinite(current_a) || !(options.relative_tolerance > 0.0)
        || !std::isfinite(options.relative_tolerance) || options.max_newton_iterations < 0)
      return result;
    std::vector<double> a(model_.nodes.size(), 0.0);
    for (size_t i = 0; i < model_.dirichlet_nodes.size(); ++i)
      a[model_.dirichlet_nodes[i]] = model_.dirichlet_a_wb_per_m[i];
    const bool has_warm_start = warm_start != nullptr;
    if (has_warm_start) {
      if (warm_start->size() != a.size())
        return result;
      a = *warm_start;
      for (size_t i = 0; i < model_.dirichlet_nodes.size(); ++i)
        a[model_.dirichlet_nodes[i]] = model_.dirichlet_a_wb_per_m[i];
    }
    double relaxation = 1.0;
    double previous_residual = std::numeric_limits<double>::infinity();
    for (int iteration = 0; iteration < options.max_newton_iterations; ++iteration) {
      Assembly assembly;
      const bool use_newton = has_warm_start || iteration > 0;
      Status status = AssembleNonlinearNewton(model_, a, current_a, use_newton, &assembly);
      if (status != Status::kOk) {
        result.info.status = status;
        return result;
      }
      GpuCsrSolver solver;
      status = solver.Initialize(assembly);
      if (status != Status::kOk) {
        result.info.status = status;
        return result;
      }
      std::vector<double> free_solution;
      std::vector<double> rhs(assembly.free_nodes.size());
      for (size_t row = 0; row < rhs.size(); ++row)
        rhs[row] = assembly.rhs_per_amp[row] + assembly.rhs_offset[row];
      const SolveInfo linear_info = solver.Solve(rhs, options.linear_relative_tolerance,
          options.max_linear_iterations, &free_solution);
      if (linear_info.status != Status::kOk) {
        result.info = linear_info;
        return result;
      }
      std::vector<double> candidate = assembly.boundary_values;
      for (size_t row = 0; row < free_solution.size(); ++row)
        candidate[assembly.free_nodes[row]] = free_solution[row];
      double difference_sq = 0.0;
      double candidate_sq = 0.0;
      for (size_t i = 0; i < candidate.size(); ++i) {
        const double difference = candidate[i] - a[i];
        difference_sq += difference * difference;
        candidate_sq += candidate[i] * candidate[i];
      }
      const double residual = std::sqrt(difference_sq)
          / std::max(std::sqrt(candidate_sq), std::numeric_limits<double>::min());
      if (!std::isfinite(residual)) {
        result.info.status = Status::kNumericalNonfinite;
        return result;
      }
      result.residual_history.push_back(residual);
      if (iteration > 5) {
        if (residual > previous_residual && relaxation > 0.125)
          relaxation *= 0.5;
        else
          relaxation += 0.1 * (1.0 - relaxation);
      }
      for (size_t i = 0; i < a.size(); ++i)
        a[i] += relaxation * (candidate[i] - a[i]);
      result.info.iterations = iteration + 1;
      result.info.residual_l2 = residual;
      if (iteration > 0 && residual < 100.0 * options.relative_tolerance) {
        result.info.status = Status::kOk;
        result.a_wb_per_m = std::move(a);
        return FinalizeResult(&result);
      }
      previous_residual = residual;
    }
    result.info.status = Status::kNonlinearSolveNotConverged;
    result.a_wb_per_m = std::move(a);
    return FinalizeResult(&result);
  }

  private:
  NonlinearSolveResult FinalizeResult(NonlinearSolveResult* result) const
  {
    result->bx_t.assign(model_.triangles.size(), 0.0);
    result->by_t.assign(model_.triangles.size(), 0.0);
    double flux = 0.0;
    for (size_t element = 0; element < model_.triangles.size(); ++element) {
      const NonlinearTriangle& triangle = model_.triangles[element];
      Node p[3] = { model_.nodes[triangle.node[0]], model_.nodes[triangle.node[1]],
        model_.nodes[triangle.node[2]] };
      NonlinearElementTerms terms;
      if (!BuildElementTerms(p, &terms)) {
        result->info.status = Status::kMeshInvalid;
        return *result;
      }
      for (int i = 0; i < 3; ++i) {
        result->bx_t[element] += result->a_wb_per_m[triangle.node[i]] * terms.b_x[i];
        result->by_t[element] += result->a_wb_per_m[triangle.node[i]] * terms.b_y[i];
      }
      flux += terms.area_m2 * model_.materials[triangle.material].source_j_per_a / 3.0
          * (result->a_wb_per_m[triangle.node[0]] + result->a_wb_per_m[triangle.node[1]]
              + result->a_wb_per_m[triangle.node[2]]);
    }
    result->flux_linkage_wb = model_.depth_m * flux;
    if (!std::isfinite(result->flux_linkage_wb))
      result->info.status = Status::kNumericalNonfinite;
    return *result;
  }

  bool initialized_ = false;
  NonlinearModel model_;
};

constexpr double kMu0 = 4.0e-7 * 3.141592653589793238462643383279502884;
// fkn assembles magnetics in centimetres and writes A = (100 * mu0) V.
constexpr double kFemmInternalToPhysicalA = 100.0 * kMu0;

struct FemmReference {
  Model model;
  std::vector<double> expected_a;
  std::vector<double> expected_bx;
  std::vector<double> expected_by;
  double current_a = 0.0;
  double flux_linkage_wb = 0.0;
  Assembly dump;
  std::vector<double> dump_rhs;
};

bool NearMixed(double actual, double expected, double absolute_tolerance,
    double relative_tolerance)
{
  return std::isfinite(actual) && std::isfinite(expected)
      && std::abs(actual - expected)
          <= absolute_tolerance + relative_tolerance * std::abs(expected);
}

bool ReadFemmAns(const std::string& path, FemmReference* reference,
    std::string* error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::string line;
  bool found_solution = false;
  while (std::getline(input, line)) {
    if (line == "[Solution]" || line == "[Solution]\r") {
      found_solution = true;
      break;
    }
  }
  int node_count = 0;
  if (!found_solution || !(input >> node_count) || node_count <= 0) {
    *error = "invalid [Solution] node count";
    return false;
  }
  reference->model.nodes.resize(node_count);
  reference->expected_a.resize(node_count);
  for (int node = 0; node < node_count; ++node) {
    int boundary_code = 0;
    if (!(input >> reference->model.nodes[node].x_m
              >> reference->model.nodes[node].y_m
              >> reference->expected_a[node] >> boundary_code)
        || !std::isfinite(reference->expected_a[node])) {
      *error = "invalid FEMM solution node";
      return false;
    }
  }
  int element_count = 0;
  if (!(input >> element_count) || element_count <= 0) {
    *error = "invalid FEMM solution element count";
    return false;
  }
  reference->model.triangles.resize(element_count);
  std::vector<bool> boundary(node_count, false);
  for (int element = 0; element < element_count; ++element) {
    int label = 0;
    int edge[3] = { -1, -1, -1 };
    double element_current = 0.0;
    Triangle& triangle = reference->model.triangles[element];
    if (!(input >> triangle.node[0] >> triangle.node[1] >> triangle.node[2]
              >> label >> edge[0] >> edge[1] >> edge[2] >> element_current)
        || label != 0 || !std::isfinite(element_current)) {
      *error = "invalid or unsupported FEMM solution element";
      return false;
    }
    triangle.reluctivity_m_per_h = 1.0 / kMu0;
    triangle.source_j_per_a = 1.0;
    for (int side = 0; side < 3; ++side) {
      if (edge[side] >= 0) {
        const int first = triangle.node[side];
        const int second = triangle.node[(side + 1) % 3];
        if (first < 0 || first >= node_count || second < 0 || second >= node_count) {
          *error = "FEMM boundary node out of range";
          return false;
        }
        boundary[first] = true;
        boundary[second] = true;
      }
    }
  }
  for (int node = 0; node < node_count; ++node) {
    if (boundary[node]) {
      if (reference->expected_a[node] != 0.0) {
        *error = "fixture requires zero-A outer boundary";
        return false;
      }
      reference->model.dirichlet_nodes.push_back(node);
      reference->model.dirichlet_a_wb_per_m.push_back(0.0);
    }
  }
  reference->model.depth_m = 1.0;
  if (ValidateModel(reference->model) != Status::kOk) {
    *error = "FEMM mesh failed frozen-scope validation";
    return false;
  }

  reference->expected_bx.resize(element_count);
  reference->expected_by.resize(element_count);
  double area_sum = 0.0;
  for (int element = 0; element < element_count; ++element) {
    const Triangle& triangle = reference->model.triangles[element];
    const Node p0 = reference->model.nodes[triangle.node[0]];
    const Node p1 = reference->model.nodes[triangle.node[1]];
    const Node p2 = reference->model.nodes[triangle.node[2]];
    const double determinant = (p1.x_m - p0.x_m) * (p2.y_m - p0.y_m)
        - (p2.x_m - p0.x_m) * (p1.y_m - p0.y_m);
    area_sum += 0.5 * determinant;
    const double a0 = reference->expected_a[triangle.node[0]];
    const double a1 = reference->expected_a[triangle.node[1]];
    const double a2 = reference->expected_a[triangle.node[2]];
    reference->expected_bx[element] = (a0 * (p2.x_m - p1.x_m)
        + a1 * (p0.x_m - p2.x_m) + a2 * (p1.x_m - p0.x_m)) / determinant;
    reference->expected_by[element] = -(a0 * (p1.y_m - p2.y_m)
        + a1 * (p2.y_m - p0.y_m) + a2 * (p0.y_m - p1.y_m)) / determinant;
  }
  if (!NearMixed(area_sum, 1.0, 1e-12, 1e-12)) {
    *error = "fixture source normalization requires one square metre";
    return false;
  }
  return true;
}

bool ReadCircuitReference(const std::string& path, FemmReference* reference,
    std::string* error)
{
  std::ifstream input(path);
  std::string current_name;
  std::string voltage_name;
  std::string flux_name;
  double voltage = 0.0;
  if (!input || !(input >> current_name >> reference->current_a
                    >> voltage_name >> voltage
                    >> flux_name >> reference->flux_linkage_wb)
      || current_name != "current_A" || voltage_name != "voltage_drop_V"
      || flux_name != "flux_linkage_Wb" || reference->current_a != 12.0
      || voltage != 0.0 || !std::isfinite(reference->flux_linkage_wb)) {
    *error = "invalid frozen circuit reference";
    return false;
  }
  return true;
}

struct NonlinearFemmReference {
  NonlinearModel model;
  std::vector<double> expected_a;
  std::vector<double> expected_bx;
  std::vector<double> expected_by;
  double current_a = 0.0;
  double flux_linkage_wb = 0.0;
};

bool ReadNonlinearFemmReference(const std::string& stem, const std::string& curve_directory,
    NonlinearFemmReference* reference, std::string* error)
{
  std::ifstream input(stem + ".ans");
  std::string line;
  while (std::getline(input, line) && line != "[Solution]") {}
  int node_count = 0;
  if (!input || !(input >> node_count) || node_count <= 0) {
    *error = "invalid nonlinear [Solution] node count";
    return false;
  }
  reference->model.nodes.resize(node_count);
  reference->expected_a.resize(node_count);
  for (int node = 0; node < node_count; ++node) {
    int ignored = 0;
    if (!(input >> reference->model.nodes[node].x_m >> reference->model.nodes[node].y_m
              >> reference->expected_a[node] >> ignored)) {
      *error = "invalid nonlinear solution node";
      return false;
    }
    reference->model.nodes[node].x_m *= 1e-3;
    reference->model.nodes[node].y_m *= 1e-3;
  }
  int element_count = 0;
  if (!(input >> element_count) || element_count <= 0) {
    *error = "invalid nonlinear solution element count";
    return false;
  }
  reference->model.triangles.resize(element_count);
  std::vector<int> label(element_count, -1);
  bool label_seen[4] = { false, false, false, false };
  std::vector<bool> boundary(node_count, false);
  for (int element = 0; element < element_count; ++element) {
    int edge[3] = { -1, -1, -1 };
    double element_current = 0.0;
    NonlinearTriangle& triangle = reference->model.triangles[element];
    if (!(input >> triangle.node[0] >> triangle.node[1] >> triangle.node[2] >> label[element]
              >> edge[0] >> edge[1] >> edge[2] >> element_current)
        || label[element] < 0 || label[element] > 3 || !std::isfinite(element_current)) {
      *error = "missing or out-of-range nonlinear block label";
      return false;
    }
    triangle.material = label[element];
    label_seen[label[element]] = true;
    for (int side = 0; side < 3; ++side) {
      if (edge[side] >= 0) {
        const int first = triangle.node[side];
        const int second = triangle.node[(side + 1) % 3];
        if (first < 0 || first >= node_count || second < 0 || second >= node_count) {
          *error = "nonlinear boundary node out of range";
          return false;
        }
        boundary[first] = boundary[second] = true;
      }
    }
  }
  for (bool seen : label_seen) {
    if (!seen) {
      *error = "nonlinear fixture is missing an expected material region";
      return false;
    }
  }
  double coil_area = 0.0;
  for (int element = 0; element < element_count; ++element) {
    const NonlinearTriangle& triangle = reference->model.triangles[element];
    Node p[3] = { reference->model.nodes[triangle.node[0]], reference->model.nodes[triangle.node[1]],
      reference->model.nodes[triangle.node[2]] };
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms)) {
      *error = "invalid nonlinear element geometry";
      return false;
    }
    if (label[element] == 2)
      coil_area += terms.area_m2;
  }
  if (!(coil_area > 0.0)) {
    *error = "nonlinear fixture has no coil material";
    return false;
  }
  NonlinearBhCurve steel_curve;
  if (!ReadNonlinearBhTable(curve_directory + "/35PN230.tab", &steel_curve)) {
    *error = "cannot read 35PN230 B-H table";
    return false;
  }
  reference->model.bh_curves = { std::move(steel_curve) };
  reference->model.materials = {
    { 1.0, 0.0, 0.0, 0.0, 0.0, 0 }, // label 0: Steel35PN230
    { 1.0 / (kMu0 * 1.05), 0.0, 0.0, 900000.0, 0.0 }, // label 1: PM
    { 1.0 / kMu0, 0.0, 50.0 / coil_area, 0.0, 0.0 }, // label 2: driven coil-air
    { 1.0 / kMu0, 0.0, 0.0, 0.0, 0.0 }, // label 3: default air
  };
  reference->model.depth_m = 0.020;
  for (int node = 0; node < node_count; ++node) {
    if (boundary[node]) {
      if (reference->expected_a[node] != 0.0) {
        *error = "fixture requires zero-A outer boundary";
        return false;
      }
      reference->model.dirichlet_nodes.push_back(node);
      reference->model.dirichlet_a_wb_per_m.push_back(0.0);
    }
  }
  if (ValidateNonlinearModel(reference->model) != Status::kOk) {
    *error = "nonlinear frozen mesh validation failed";
    return false;
  }
  std::ifstream summary(stem + ".summary.txt");
  std::string key;
  while (summary >> key) {
    if (key == "circuit_current_A=")
      summary >> reference->current_a;
    else if (key == "flux_linkage_Wb=")
      summary >> reference->flux_linkage_wb;
    else
      summary.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  // Summary is key=value, not whitespace-separated; parse it linewise when needed.
  if (!std::isfinite(reference->current_a) || reference->current_a == 0.0
      || !std::isfinite(reference->flux_linkage_wb) || reference->flux_linkage_wb == 0.0) {
    summary.clear();
    summary.seekg(0);
    while (std::getline(summary, line)) {
      const size_t equals = line.find('=');
      if (equals == std::string::npos)
        continue;
      if (line.substr(0, equals) == "circuit_current_A")
        reference->current_a = std::stod(line.substr(equals + 1));
      if (line.substr(0, equals) == "flux_linkage_Wb")
        reference->flux_linkage_wb = std::stod(line.substr(equals + 1));
    }
  }
  if (!std::isfinite(reference->current_a) || !std::isfinite(reference->flux_linkage_wb)) {
    *error = "invalid nonlinear fixture summary";
    return false;
  }
  reference->expected_bx.resize(element_count);
  reference->expected_by.resize(element_count);
  for (int element = 0; element < element_count; ++element) {
    const NonlinearTriangle& triangle = reference->model.triangles[element];
    Node p[3] = { reference->model.nodes[triangle.node[0]], reference->model.nodes[triangle.node[1]],
      reference->model.nodes[triangle.node[2]] };
    NonlinearElementTerms terms;
    BuildElementTerms(p, &terms);
    for (int i = 0; i < 3; ++i) {
      reference->expected_bx[element] += reference->expected_a[triangle.node[i]] * terms.b_x[i];
      reference->expected_by[element] += reference->expected_a[triangle.node[i]] * terms.b_y[i];
    }
  }
  return true;
}

bool ReadDumpRhs(const std::string& path, std::vector<double>* rhs,
    std::string* error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::string line;
  bool reading = false;
  while (std::getline(input, line)) {
    if (!reading) {
      const size_t start = line.find("b = [");
      if (start == std::string::npos)
        continue;
      line = line.substr(start + 5);
      reading = true;
    }
    const bool end = line.find("];") != std::string::npos;
    line.erase(std::remove(line.begin(), line.end(), ';'), line.end());
    line.erase(std::remove(line.begin(), line.end(), ']'), line.end());
    std::istringstream value_input(line);
    double value = 0.0;
    if (value_input >> value)
      rhs->push_back(value);
    if (end)
      break;
  }
  if (!reading || rhs->empty()) {
    *error = "invalid FEMM dump RHS";
    return false;
  }
  return true;
}

bool ReadDumpMatrix(const std::string& path, size_t dimension, Assembly* dump,
    std::string* error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::map<std::pair<int, int>, std::vector<double>> entries;
  int one_based_row = 0;
  int one_based_column = 0;
  double value = 0.0;
  while (input >> one_based_row >> one_based_column >> value) {
    const int row = one_based_row - 1;
    const int column = one_based_column - 1;
    if (row < 0 || column < 0 || static_cast<size_t>(row) >= dimension
        || static_cast<size_t>(column) >= dimension || !std::isfinite(value)) {
      *error = "invalid FEMM dump matrix entry";
      return false;
    }
    entries[{ std::min(row, column), std::max(row, column) }].push_back(value);
  }
  std::vector<std::map<int, double>> rows(dimension);
  for (const auto& item : entries) {
    const std::vector<double>& duplicates = item.second;
    if (duplicates.size() != 2 || duplicates[0] != duplicates[1]) {
      *error = "FEMM dump duplicate/mirror invariant failed";
      return false;
    }
    const int row = item.first.first;
    const int column = item.first.second;
    rows[row][column] = duplicates[0];
    if (row != column)
      rows[column][row] = duplicates[0];
  }
  dump->row_offsets.assign(dimension + 1, 0);
  dump->diagonal.assign(dimension, 0.0);
  dump->free_nodes.resize(dimension);
  for (size_t row = 0; row < dimension; ++row) {
    dump->free_nodes[row] = static_cast<int32_t>(row);
    dump->row_offsets[row] = static_cast<int32_t>(dump->values.size());
    for (const auto& entry : rows[row]) {
      dump->column_indices.push_back(entry.first);
      dump->values.push_back(entry.second);
      if (entry.first == static_cast<int>(row))
        dump->diagonal[row] = entry.second;
    }
    if (!(dump->diagonal[row] > 0.0)) {
      *error = "FEMM dump missing positive diagonal";
      return false;
    }
  }
  dump->row_offsets[dimension] = static_cast<int32_t>(dump->values.size());
  return true;
}

bool LoadFemmReference(const std::string& stem, FemmReference* reference,
    std::string* error)
{
  return ReadFemmAns(stem + ".ans", reference, error)
      && ReadCircuitReference(stem + ".circuit.txt", reference, error)
      && ReadDumpRhs(stem + ".m", &reference->dump_rhs, error)
      && reference->dump_rhs.size() == reference->expected_a.size()
      && ReadDumpMatrix(stem + ".dat", reference->dump_rhs.size(),
          &reference->dump, error);
}

int FemmReferenceTest(const std::string& stem)
{
  FemmReference reference;
  std::string error;
  if (!LoadFemmReference(stem, &reference, &error)) {
    std::cerr << "FAIL: " << (error.empty() ? "reference dimension mismatch" : error) << '\n';
    return 1;
  }
  int failures = 0;
  double max_a_error = 0.0;
  double max_b_error = 0.0;
  auto expect = [&failures](bool condition, const std::string& message) {
    if (!condition) {
      std::cerr << "FAIL: " << message << '\n';
      ++failures;
    }
  };

  LinearP1FixtureSolver physical_solver;
  const Status initialize_status = physical_solver.Initialize(reference.model);
  expect(initialize_status == Status::kOk,
      std::string("FEMM physical fixture initialization: ") + StatusName(initialize_status));
  if (initialize_status == Status::kOk) {
    const SolveResult result = physical_solver.Solve(reference.current_a, 1e-13, 256);
    expect(result.info.status == Status::kOk,
        std::string("FEMM physical fixture solve: ") + StatusName(result.info.status));
    expect(result.a_wb_per_m.size() == reference.expected_a.size(), "FEMM nodal A dimension");
    expect(result.bx_t.size() == reference.expected_bx.size(), "FEMM Bx dimension");
    expect(result.by_t.size() == reference.expected_by.size(), "FEMM By dimension");
    if (result.a_wb_per_m.size() == reference.expected_a.size()) {
      for (size_t i = 0; i < result.a_wb_per_m.size(); ++i) {
        max_a_error = std::max(max_a_error,
            std::abs(result.a_wb_per_m[i] - reference.expected_a[i]));
        expect(NearMixed(result.a_wb_per_m[i], reference.expected_a[i], 1e-12, 1e-4),
            "FEMM nodal A parity");
      }
    }
    if (result.bx_t.size() == reference.expected_bx.size()
        && result.by_t.size() == reference.expected_by.size()) {
      for (size_t i = 0; i < result.bx_t.size(); ++i) {
        max_b_error = std::max(max_b_error,
            std::max(std::abs(result.bx_t[i] - reference.expected_bx[i]),
                std::abs(result.by_t[i] - reference.expected_by[i])));
        expect(NearMixed(result.bx_t[i], reference.expected_bx[i], 1e-9, 1e-4),
            "FEMM Bx parity");
        expect(NearMixed(result.by_t[i], reference.expected_by[i], 1e-9, 1e-4),
            "FEMM By parity");
      }
    }
    expect(NearMixed(result.flux_linkage_wb, reference.flux_linkage_wb, 1e-12, 1e-4),
        "FEMM circuit flux-linkage parity");
  }

  GpuCsrSolver dump_solver;
  const Status dump_status = dump_solver.Initialize(reference.dump);
  expect(dump_status == Status::kOk,
      std::string("FEMM dump initialization: ") + StatusName(dump_status));
  if (dump_status == Status::kOk) {
    std::vector<double> internal_solution;
    const SolveInfo info = dump_solver.Solve(reference.dump_rhs, 1e-13, 256,
        &internal_solution);
    expect(info.status == Status::kOk,
        std::string("FEMM dump solve: ") + StatusName(info.status));
    expect(internal_solution.size() == reference.expected_a.size(), "FEMM dump dimension");
    if (internal_solution.size() == reference.expected_a.size()) {
      for (size_t i = 0; i < internal_solution.size(); ++i) {
        const double physical_a = kFemmInternalToPhysicalA * internal_solution[i];
        expect(NearMixed(physical_a, reference.expected_a[i], 1e-12, 1e-4),
            "FEMM dump-to-ans A parity");
      }
    }
  }

  if (failures == 0) {
    std::cout << "PASS gpu_linear_p1_femm_reference\n"
              << "  nodes=" << reference.model.nodes.size()
              << " triangles=" << reference.model.triangles.size()
              << " max_A_error=" << max_a_error
              << " max_B_error=" << max_b_error << '\n'
              << "  flux_reference_Wb=" << reference.flux_linkage_wb << '\n';
  }
  return failures == 0 ? 0 : 1;
}

int NonlinearFemmReferenceTest(const std::string& stem, const std::string& curve_directory)
{
  NonlinearFemmReference reference;
  std::string error;
  if (!ReadNonlinearFemmReference(stem, curve_directory, &reference, &error)) {
    std::cerr << "FAIL nonlinear FEMM reference input: " << error << '\n';
    return 1;
  }
  NonlinearP1FixtureSolver solver;
  const Status initialize = solver.Initialize(reference.model);
  if (initialize != Status::kOk) {
    std::cerr << "FAIL nonlinear FEMM initialization: " << StatusName(initialize) << '\n';
    return 1;
  }
  NonlinearOptions options;
  options.relative_tolerance = 1e-8;
  options.max_newton_iterations = 128;
  const NonlinearSolveResult actual = solver.Solve(reference.current_a, options);
  if (actual.info.status != Status::kOk) {
    std::cerr << "FAIL nonlinear FEMM solve: " << StatusName(actual.info.status) << '\n';
    return 1;
  }
  double max_a_error = 0.0, max_b_error = 0.0;
  for (size_t i = 0; i < actual.a_wb_per_m.size(); ++i)
    max_a_error = std::max(max_a_error, std::abs(actual.a_wb_per_m[i] - reference.expected_a[i]));
  for (size_t i = 0; i < actual.bx_t.size(); ++i) {
    max_b_error = std::max(max_b_error, std::abs(actual.bx_t[i] - reference.expected_bx[i]));
    max_b_error = std::max(max_b_error, std::abs(actual.by_t[i] - reference.expected_by[i]));
  }
  const bool parity = NearMixed(actual.flux_linkage_wb, reference.flux_linkage_wb, 1e-10, 1e-6)
      && max_a_error <= 1e-9 && max_b_error <= 1e-6;
  if (!parity) {
    std::cerr << "FAIL nonlinear FEMM parity A=" << max_a_error << " B=" << max_b_error
              << " flux=" << actual.flux_linkage_wb << " expected_flux="
              << reference.flux_linkage_wb << '\n';
    return 1;
  }
  std::cout << "PASS gpu_nonlinear_p1_femm_reference\n"
            << "  nodes=" << reference.model.nodes.size() << " triangles="
            << reference.model.triangles.size() << " iterations=" << actual.info.iterations
            << " max_A_error=" << max_a_error << " max_B_error=" << max_b_error
            << " flux_Wb=" << actual.flux_linkage_wb << '\n';
  return 0;
}

Model UnitSquareFixture()
{
  Model model;
  model.nodes = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }, { 0.0, 1.0 },
    { 0.5, 0.5 } };
  model.triangles = {
    { { 0, 1, 4 }, 1.0, 1.0 }, { { 1, 2, 4 }, 1.0, 1.0 },
    { { 2, 3, 4 }, 1.0, 1.0 }, { { 3, 0, 4 }, 1.0, 1.0 }
  };
  model.dirichlet_nodes = { 0, 1, 2, 3 };
  model.dirichlet_a_wb_per_m = { 0.0, 0.0, 0.0, 0.0 };
  model.depth_m = 1.0;
  return model;
}

NonlinearModel NonlinearThreeRegionFixture()
{
  NonlinearModel model;
  model.nodes = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }, { 0.0, 1.0 },
    { 0.5, 0.5 } };
  // PM, driven coil, and nonlinear steel all occur in the same small model.
  model.materials = {
    { 1.0, 0.0, 0.0, 10.0, 0.0 },
    { 1.0, 0.0, 12.0, 0.0, 0.0 },
    { 0.5, 1.0, 0.0, 0.0, 0.0, 0 },
  };
  model.bh_curves = { { { 0.0, 1.0, 2.0 }, { 0.0, 1.0, 4.0 }, {} } };
  BuildNonlinearBhSpline(&model.bh_curves[0]);
  model.triangles = { { { 0, 1, 4 }, 0 }, { { 1, 2, 4 }, 1 },
    { { 2, 3, 4 }, 2 }, { { 3, 0, 4 }, 2 } };
  model.dirichlet_nodes = { 0, 1, 2, 3 };
  model.dirichlet_a_wb_per_m = { 0.0, 0.0, 0.0, 0.0 };
  model.depth_m = 1.0;
  return model;
}

bool Near(double actual, double expected, double tolerance = 1e-12)
{
  return std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
}

bool SameBits(const SolveResult& left, const SolveResult& right)
{
  if (left.info.status != right.info.status || left.info.iterations != right.info.iterations || std::memcmp(&left.info.residual_l2, &right.info.residual_l2, sizeof(double)) != 0 || std::memcmp(&left.flux_linkage_wb, &right.flux_linkage_wb, sizeof(double)) != 0 || left.a_wb_per_m.size() != right.a_wb_per_m.size() || left.bx_t.size() != right.bx_t.size() || left.by_t.size() != right.by_t.size()) {
    return false;
  }
  return std::memcmp(left.a_wb_per_m.data(), right.a_wb_per_m.data(),
             left.a_wb_per_m.size() * sizeof(double))
      == 0
      && std::memcmp(left.bx_t.data(), right.bx_t.data(), left.bx_t.size() * sizeof(double)) == 0 && std::memcmp(left.by_t.data(), right.by_t.data(), left.by_t.size() * sizeof(double)) == 0;
}

int SelfTest()
{
  int failures = 0;
  auto expect = [&failures](bool condition, const std::string& message) {
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
  floating_component.nodes.push_back({ 2.0, 2.0 });
  expect(ValidateModel(floating_component) == Status::kBoundaryInvalid,
      "floating component returns BOUNDARY_INVALID");
  expect(std::string(StatusName(Status::kLinearSolveNotConverged)) == "LINEAR_SOLVE_NOT_CONVERGED",
      "stable error-code name");
  expect(std::string(StatusName(Status::kNonlinearSolveNotConverged)) == "NONLINEAR_SOLVE_NOT_CONVERGED",
      "stable nonlinear error-code name");

  // Analytic P1 contract: b_i=(dN_i/dy,-dN_i/dx), f_PM=S Hc.b_i.
  const Node analytic_nodes[3] = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 } };
  NonlinearElementTerms analytic_terms;
  expect(BuildElementTerms(analytic_nodes, &analytic_terms), "analytic element terms");
  const double expected_pm[3] = { -5.0, 0.0, 5.0 };
  for (int i = 0; i < 3; ++i) {
    expect(Near(analytic_terms.area_m2 * 10.0 * analytic_terms.b_x[i], expected_pm[i]),
        "PM source sign");
  }
  // At A=[0,0,1], v=1 and dv/d(B^2)=.5.  This is K+C exactly.
  const double bx = analytic_terms.b_x[2];
  const double by = analytic_terms.b_y[2];
  const double expected_jacobian[3][3] = {
    { 1.5, -0.5, -1.0 }, { -0.5, 0.5, 0.0 }, { -1.0, 0.0, 1.0 }
  };
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double b_dot_i = bx * analytic_terms.b_x[i] + by * analytic_terms.b_y[i];
      const double b_dot_j = bx * analytic_terms.b_x[j] + by * analytic_terms.b_y[j];
      const double actual = analytic_terms.area_m2
              * (analytic_terms.b_x[i] * analytic_terms.b_x[j]
                  + analytic_terms.b_y[i] * analytic_terms.b_y[j])
          + 2.0 * analytic_terms.area_m2 * 0.5 * b_dot_i * b_dot_j;
      expect(Near(actual, expected_jacobian[i][j]), "analytic Newton Jacobian");
    }
  }
  NonlinearModel missing_material = NonlinearThreeRegionFixture();
  missing_material.triangles[0].material = -1;
  expect(ValidateNonlinearModel(missing_material) == Status::kInvalidMaterial,
      "unassigned nonlinear material is rejected");

  Assembly three_by_three;
  three_by_three.row_offsets = { 0, 2, 5, 7 };
  three_by_three.column_indices = { 0, 1, 0, 1, 2, 1, 2 };
  three_by_three.values = { 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0 };
  three_by_three.diagonal = { 4.0, 4.0, 3.0 };
  three_by_three.free_nodes = { 0, 1, 2 };
  GpuCsrSolver csr_solver;
  Status csr_status = csr_solver.Initialize(three_by_three);
  expect(csr_status == Status::kOk, "3x3 CSR initialization");
  std::vector<double> csr_solution;
  const SolveInfo csr_info = csr_solver.Solve({ 2.0, 4.0, 7.0 }, 1e-13, 64, &csr_solution);
  expect(csr_info.status == Status::kOk, "3x3 PCG convergence");
  expect(csr_solution.size() == 3 && Near(csr_solution[0], 1.0) && Near(csr_solution[1], 2.0) && Near(csr_solution[2], 3.0),
      "3x3 PCG reference parity");

  Assembly indefinite;
  indefinite.row_offsets = { 0, 1 };
  indefinite.column_indices = { 0 };
  indefinite.values = { -1.0 };
  indefinite.diagonal = { -1.0 };
  indefinite.free_nodes = { 0 };
  GpuCsrSolver breakdown_solver;
  Status breakdown_status = breakdown_solver.Initialize(indefinite);
  expect(breakdown_status == Status::kOk, "breakdown fixture initialization");
  std::vector<double> breakdown_solution;
  const SolveInfo breakdown_info = breakdown_solver.Solve({ 1.0 }, 1e-13, 8, &breakdown_solution);
  expect(breakdown_info.status == Status::kLinearSolveBreakdown,
      "indefinite matrix returns LINEAR_SOLVE_BREAKDOWN");

  LinearP1FixtureSolver solver;
  Status initialize_status = solver.Initialize(UnitSquareFixture());
  expect(initialize_status == Status::kOk,
      std::string("fixture initialization: ") + StatusName(initialize_status));
  if (initialize_status != Status::kOk)
    return failures + 1;

  const SolveResult positive = solver.Solve(12.0);
  expect(positive.info.status == Status::kOk,
      std::string("positive solve: ") + StatusName(positive.info.status));
  const double expected_a[5] = { 0.0, 0.0, 0.0, 0.0, 1.0 };
  const double expected_bx[4] = { 2.0, 0.0, -2.0, 0.0 };
  const double expected_by[4] = { 0.0, 2.0, 0.0, -2.0 };
  expect(positive.a_wb_per_m.size() == 5 && positive.bx_t.size() == 4 && positive.by_t.size() == 4,
      "positive result dimensions");
  if (positive.a_wb_per_m.size() == 5 && positive.bx_t.size() == 4 && positive.by_t.size() == 4) {
    for (int i = 0; i < 5; ++i) {
      expect(Near(positive.a_wb_per_m[i], expected_a[i]), "nodal A parity");
    }
    for (int i = 0; i < 4; ++i) {
      expect(Near(positive.bx_t[i], expected_bx[i]), "Bx sign/parity");
      expect(Near(positive.by_t[i], expected_by[i]), "By sign/parity");
    }
  }
  expect(Near(positive.flux_linkage_wb, 1.0 / 3.0), "flux-linkage parity");
  expect(positive.info.residual_l2 <= 1e-13, "linear residual tolerance");

  const SolveResult negative = solver.Solve(-12.0);
  expect(negative.info.status == Status::kOk, "negative-current solve");
  expect(negative.a_wb_per_m.size() == 5 && negative.bx_t.size() == 4 && negative.by_t.size() == 4,
      "negative result dimensions");
  if (negative.a_wb_per_m.size() == 5 && negative.bx_t.size() == 4 && negative.by_t.size() == 4) {
    for (int i = 0; i < 5; ++i) {
      expect(Near(negative.a_wb_per_m[i], -expected_a[i]), "negative A sign");
    }
    for (int i = 0; i < 4; ++i) {
      expect(Near(negative.bx_t[i], -expected_bx[i]), "negative Bx sign");
      expect(Near(negative.by_t[i], -expected_by[i]), "negative By sign");
    }
  }
  expect(Near(negative.flux_linkage_wb, -1.0 / 3.0), "negative flux sign");

  const SolveResult tiny = solver.Solve(1e-14);
  expect(tiny.info.status == Status::kOk && tiny.info.iterations > 0,
      "small nonzero RHS is solved, not accepted at x=0");
  expect(tiny.a_wb_per_m.size() == 5 && Near(tiny.a_wb_per_m[4], 1e-14 / 12.0, 1e-28),
      "small nonzero RHS parity");
  const SolveResult subnormal_square = solver.Solve(1e-200);
  expect(subnormal_square.info.status == Status::kOk && subnormal_square.info.iterations > 0,
      "subnormal-square RHS is solved, not accepted at x=0");
  expect(subnormal_square.a_wb_per_m.size() == 5 && Near(subnormal_square.a_wb_per_m[4], 1e-200 / 12.0, 1e-214),
      "subnormal-square RHS parity");
  const SolveResult invalid_tolerance = solver.Solve(12.0, std::numeric_limits<double>::infinity(), 8);
  expect(invalid_tolerance.info.status == Status::kInvalidArgument,
      "infinite tolerance returns INVALID_ARGUMENT");

  const SolveResult not_converged = solver.Solve(12.0, 1e-13, 0);
  expect(not_converged.info.status == Status::kLinearSolveNotConverged,
      "iteration limit returns LINEAR_SOLVE_NOT_CONVERGED");

  NonlinearP1FixtureSolver nonlinear_solver;
  const Status nonlinear_initialize = nonlinear_solver.Initialize(NonlinearThreeRegionFixture());
  expect(nonlinear_initialize == Status::kOk,
      std::string("nonlinear fixture initialization: ") + StatusName(nonlinear_initialize));
  if (nonlinear_initialize == Status::kOk) {
    const NonlinearSolveResult nonlinear = nonlinear_solver.Solve(2.0);
    expect(nonlinear.info.status == Status::kOk,
        std::string("nonlinear PM+coil solve: ") + StatusName(nonlinear.info.status));
    expect(nonlinear.a_wb_per_m.size() == 5 && nonlinear.bx_t.size() == 4
            && nonlinear.by_t.size() == 4 && nonlinear.residual_history.size() >= 2,
        "nonlinear result dimensions and history");
    expect(std::isfinite(nonlinear.flux_linkage_wb), "nonlinear circuit flux");
    if (nonlinear.info.status == Status::kOk) {
      const NonlinearSolveResult warm = nonlinear_solver.Solve(2.0, {}, &nonlinear.a_wb_per_m);
      expect(warm.info.status == Status::kOk, "nonlinear warm-start solve");
      expect(warm.info.iterations <= nonlinear.info.iterations, "warm start does not add Newton steps");
      expect(warm.a_wb_per_m.size() == nonlinear.a_wb_per_m.size(), "warm result dimensions");
      if (warm.a_wb_per_m.size() == nonlinear.a_wb_per_m.size()) {
        for (size_t i = 0; i < warm.a_wb_per_m.size(); ++i)
          expect(Near(warm.a_wb_per_m[i], nonlinear.a_wb_per_m[i], 1e-9),
              "warm-start solution parity");
      }
    }
    NonlinearOptions no_iterations;
    no_iterations.max_newton_iterations = 0;
    const NonlinearSolveResult nonlinear_limited = nonlinear_solver.Solve(2.0, no_iterations);
    expect(nonlinear_limited.info.status == Status::kNonlinearSolveNotConverged,
        "nonlinear iteration cap returns NONLINEAR_SOLVE_NOT_CONVERGED");
  }

  solver.Solve(12.0); // warm up all runtime paths before memory accounting.
  size_t free_before = 0;
  size_t total_before = 0;
  const cudaError_t memory_before_status = cudaMemGetInfo(&free_before, &total_before);
  expect(memory_before_status == cudaSuccess, "pre-repeat cudaMemGetInfo");
  for (int repeat = 0; repeat < 100; ++repeat) {
    const SolveResult repeated = solver.Solve(12.0);
    expect(SameBits(positive, repeated), "100-run bitwise determinism");
  }
  size_t free_after = 0;
  size_t total_after = 0;
  const cudaError_t memory_after_status = cudaMemGetInfo(&free_after, &total_after);
  expect(memory_after_status == cudaSuccess, "post-repeat cudaMemGetInfo");
  if (memory_before_status == cudaSuccess && memory_after_status == cudaSuccess) {
    expect(free_after >= free_before, "no GPU-memory growth across 100 solves");
  }

  if (failures == 0) {
    std::cout << "PASS gpu_linear_p1_selftest\n"
              << "  status=" << StatusName(positive.info.status)
              << " iterations=" << positive.info.iterations
              << " residual=" << positive.info.residual_l2 << '\n'
              << "  A_center=" << positive.a_wb_per_m.at(4)
              << " flux_Wb=" << positive.flux_linkage_wb << '\n'
              << "  deterministic_repeats=100 gpu_memory_delta_bytes="
              << static_cast<long long>(free_before) - static_cast<long long>(free_after)
              << '\n';
  }
  return failures == 0 ? 0 : 1;
}

} // namespace gpu_femm

int main(int argc, char** argv)
{
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    return gpu_femm::SelfTest();
  }
  if (argc == 3 && std::string(argv[1]) == "--femm-reference") {
    return gpu_femm::FemmReferenceTest(argv[2]);
  }
  if (argc == 4 && std::string(argv[1]) == "--nonlinear-reference") {
    return gpu_femm::NonlinearFemmReferenceTest(argv[2], argv[3]);
  }
  std::cerr << "Usage: gpu_linear_p1_poc --self-test | --femm-reference <stem>"
            << " | --nonlinear-reference <stem> <curve_dir>\n";
  return 2;
}

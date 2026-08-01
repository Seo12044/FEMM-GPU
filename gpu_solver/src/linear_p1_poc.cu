#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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
  std::cerr << "Usage: gpu_linear_p1_poc --self-test\n";
  return 2;
}

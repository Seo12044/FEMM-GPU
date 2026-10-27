#include "physics/nonlinear_assembly.cuh"
#include "core/profile.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace gpu_femm
{

__device__ bool DeviceBhEvaluation(const DeviceNonlinearTriangle &triangle, const double *curve_b,
                                   const double *curve_h, const double *curve_slope, double field_b,
                                   double *reluctivity, double *derivative)
{
  if (reluctivity == nullptr || derivative == nullptr || triangle.curve_count < 2)
    return false;
  const double b = fabs(field_b);
  const int offset = triangle.curve_offset;
  if (b == 0.0) {
    *reluctivity = curve_slope[offset];
    *derivative = 0.0;
    return isfinite(*reluctivity);
  }
  double h = 0.0, dh = 0.0;
  const int last = offset + triangle.curve_count - 1;
  if (b > curve_b[last]) {
    h = curve_h[last] + curve_slope[last] * (b - curve_b[last]);
    dh = curve_slope[last];
  } else {
    bool found = false;
    for (int index = offset; index < last; ++index) {
      if (b >= curve_b[index] && b <= curve_b[index + 1]) {
        const double length = curve_b[index + 1] - curve_b[index];
        const double z = (b - curve_b[index]) / length, z2 = z * z;
        h = (1.0 - 3.0 * z2 + 2.0 * z2 * z) * curve_h[index] +
            z * (1.0 - 2.0 * z + z2) * length * curve_slope[index] +
            z2 * (3.0 - 2.0 * z) * curve_h[index + 1] +
            z2 * (z - 1.0) * length * curve_slope[index + 1];
        dh = 6.0 * z * (z - 1.0) * curve_h[index] / length +
             (1.0 - 4.0 * z + 3.0 * z2) * curve_slope[index] +
             6.0 * z * (1.0 - z) * curve_h[index + 1] / length +
             z * (3.0 * z - 2.0) * curve_slope[index + 1];
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  *reluctivity = h / b;
  *derivative = 0.5 * (dh / (b * b) - h / (b * b * b));
  return isfinite(*reluctivity) && isfinite(*derivative);
}

__global__ void DeviceNonlinearElementKernel(const DeviceNonlinearTriangle *triangles,
                                             size_t triangle_count, const double *old_a,
                                             size_t node_count, const double *curve_b,
                                             const double *curve_h, const double *curve_slope,
                                             bool use_newton, double *bx_values, double *by_values,
                                             double *nu_values, double *derivative_values,
                                             size_t quadrature_stride, int *valid)
{
  const size_t triangle = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t item = blockIdx.y;
  if (triangle >= triangle_count)
    return;
  const DeviceNonlinearTriangle *properties = triangles + triangle;
  const double *a = old_a + item * node_count;
  for (int point = 0; point < properties->quadrature_count; ++point) {
    double bx = 0.0, by = 0.0;
    for (int local = 0; local < 3; ++local) {
      bx += a[properties->node[local]] * properties->b_x[point][local];
      by += a[properties->node[local]] * properties->b_y[point][local];
    }
    const double magnitude = hypot(bx, by);
    double nu = 0.0, derivative = 0.0;
    bool ok = isfinite(magnitude);
    if (ok && properties->curve_count >= 0) {
      if (!use_newton) {
        nu = properties->cold_secant;
        derivative = 0.0;
      } else
        ok = DeviceBhEvaluation(*properties, curve_b, curve_h, curve_slope, magnitude, &nu,
                                &derivative);
    } else if (ok) {
      nu = properties->reluctivity_zero * (1.0 + properties->alpha * magnitude * magnitude);
      derivative = use_newton ? properties->reluctivity_zero * properties->alpha : 0.0;
    }
    const size_t index = (item * triangle_count + triangle) * quadrature_stride + point;
    bx_values[index] = bx;
    by_values[index] = by;
    nu_values[index] = nu;
    derivative_values[index] = derivative;
    if (!ok || !(nu > 0.0) || !isfinite(derivative))
      atomicExch(valid, 0);
  }
}

__global__ void DeviceNonlinearValuesKernel(const DeviceNonlinearTriangle *triangles,
                                            const int32_t *contributor_offsets,
                                            const DeviceMatrixContributor *contributors, size_t nnz,
                                            const double *base_values, const double *bx_values,
                                            const double *by_values, const double *nu_values,
                                            const double *derivative_values, size_t triangle_count,
                                            size_t quadrature_stride, bool use_newton,
                                            double *values)
{
  const size_t slot = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t item = blockIdx.y;
  if (slot >= nnz)
    return;
  double sum = 0.0;
  const size_t element_base = item * triangle_count;
  // The contributor list is fixed in host element/i/j emission order. One
  // lane owns a slot, so no atomic or schedule-dependent reduction is used.
  for (int32_t position = contributor_offsets[slot]; position < contributor_offsets[slot + 1];
       ++position) {
    const DeviceMatrixContributor contributor = contributors[position];
    const size_t triangle_index = static_cast<size_t>(contributor.triangle);
    const int local_i = contributor.local_row, local_j = contributor.local_column;
    const DeviceNonlinearTriangle *triangle = triangles + triangle_index;
    for (int point = 0; point < triangle->quadrature_count; ++point) {
      const size_t index = (element_base + triangle_index) * quadrature_stride + point;
      const double bx = bx_values[index], by = by_values[index];
      const double gradient = triangle->b_x[point][local_i] * triangle->b_x[point][local_j] +
                              triangle->b_y[point][local_i] * triangle->b_y[point][local_j];
      const double k = triangle->weight[point] * nu_values[index] * gradient;
      const double b_i = bx * triangle->b_x[point][local_i] + by * triangle->b_y[point][local_i];
      const double b_j = bx * triangle->b_x[point][local_j] + by * triangle->b_y[point][local_j];
      const double c =
          use_newton ? 2.0 * triangle->weight[point] * derivative_values[index] * b_i * b_j : 0.0;
      sum += static_cast<double>(contributor.sign) * (k + c);
    }
  }
  // Host assembly emits every triangle before AGE contributions.  Preserve
  // that phase ordering even though the fixed AGE terms are pre-accumulated.
  sum += base_values[slot];
  values[item * nnz + slot] = sum;
}

__global__ void DeviceNonlinearRhsKernel(
    const DeviceNonlinearTriangle *triangles, const int32_t *contributor_offsets,
    const DeviceRhsContributor *contributors, size_t free_count, const double *pm_rhs,
    const double *circuit_rhs, int circuit_count, const double *currents, const double *age_rhs,
    const double *old_a, const double *boundary_values, const double *bx_values,
    const double *by_values, const double *nu_values, const double *derivative_values,
    size_t triangle_count, size_t quadrature_stride, size_t node_count, bool use_newton,
    double *rhs)
{
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t item = blockIdx.y;
  if (row >= free_count)
    return;
  double sum = pm_rhs[row] + age_rhs[row];
  for (int circuit = 0; circuit < circuit_count; ++circuit)
    sum += currents[item * static_cast<size_t>(circuit_count) + circuit] *
           circuit_rhs[static_cast<size_t>(circuit) * free_count + row];
  const size_t element_base = item * triangle_count;
  for (int32_t position = contributor_offsets[row]; position < contributor_offsets[row + 1];
       ++position) {
    const DeviceRhsContributor contributor = contributors[position];
    const size_t triangle_index = static_cast<size_t>(contributor.triangle);
    const int local_i = contributor.local_row;
    const DeviceNonlinearTriangle *triangle = triangles + triangle_index;
    for (int point = 0; point < triangle->quadrature_count; ++point) {
      const size_t index = (element_base + triangle_index) * quadrature_stride + point;
      const double bx = bx_values[index], by = by_values[index];
      const double b_i = bx * triangle->b_x[point][local_i] + by * triangle->b_y[point][local_i];
      for (int local_j = 0; local_j < 3; ++local_j) {
        const double b_j = bx * triangle->b_x[point][local_j] + by * triangle->b_y[point][local_j];
        const double k = triangle->weight[point] * nu_values[index] *
                         (triangle->b_x[point][local_i] * triangle->b_x[point][local_j] +
                          triangle->b_y[point][local_i] * triangle->b_y[point][local_j]);
        const double c =
            use_newton ? 2.0 * triangle->weight[point] * derivative_values[index] * b_i * b_j : 0.0;
        const int32_t node = triangle->node[local_j];
        if (use_newton)
          sum += static_cast<double>(contributor.sign) * c * old_a[item * node_count + node];
        if (boundary_values[node] != 0.0)
          sum -= static_cast<double>(contributor.sign) * (k + c) * boundary_values[node];
      }
    }
  }
  rhs[item * free_count + row] = sum;
}

__global__ void DeviceNonlinearDiagonalKernel(const int32_t *diagonal_slots, size_t free_count,
                                              size_t nnz, const double *values, double *diagonal,
                                              double *inverse_diagonal, int *valid)
{
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t item = blockIdx.y;
  if (row >= free_count)
    return;
  const int32_t slot = diagonal_slots[row];
  const double value =
      slot >= 0 && static_cast<size_t>(slot) < nnz ? values[item * nnz + slot] : 0.0;
  diagonal[item * free_count + row] = value;
  if (isfinite(value) && value > 0.0)
    inverse_diagonal[item * free_count + row] = 1.0 / value;
  else {
    inverse_diagonal[item * free_count + row] = 0.0;
    atomicExch(valid, 0);
  }
}

Status DeviceNonlinearAssemblyPlan::EnsureStateCapacity(size_t count)
{
  if (state_capacity_ >= count)
    return Status::kOk;
  const size_t triangles = d_triangles_.count();
  Status status = Status::kOk;
  if ((status = d_old_.allocate(count * node_count_)) != Status::kOk ||
      (status = d_currents_.allocate(count * static_cast<size_t>(circuit_count_))) != Status::kOk ||
      (status = d_bx_.allocate(count * triangles * quadrature_stride_)) != Status::kOk ||
      (status = d_by_.allocate(count * triangles * quadrature_stride_)) != Status::kOk ||
      (status = d_nu_.allocate(count * triangles * quadrature_stride_)) != Status::kOk ||
      (status = d_derivative_.allocate(count * triangles * quadrature_stride_)) != Status::kOk ||
      (status = h_old_.allocate(count * node_count_)) != Status::kOk ||
      (status = h_currents_.allocate(count * static_cast<size_t>(circuit_count_))) != Status::kOk ||
      (status = h_valid_.allocate(1)) != Status::kOk)
    return status;
  uploaded_currents_.clear();
  state_capacity_ = count;
  return Status::kOk;
}

Status DeviceNonlinearAssemblyPlan::Assemble(const std::vector<std::vector<double>> &states,
                                             const std::vector<std::vector<double>> &currents,
                                             bool use_newton, GpuCsrSolver *solver, double *seconds)
{
  if (!initialized_ || solver == nullptr || states.empty() || states.size() != currents.size())
    return Status::kInvalidArgument;
  const size_t count = states.size(), triangles = d_triangles_.count();
  for (size_t item = 0; item < count; ++item) {
    if (states[item].size() != node_count_ ||
        currents[item].size() != static_cast<size_t>(circuit_count_))
      return Status::kInvalidArgument;
  }
  Status status = EnsureStateCapacity(count);
  if (status != Status::kOk)
    return status;
  for (size_t item = 0; item < count; ++item) {
    std::copy(states[item].begin(), states[item].end(), h_old_.get() + item * node_count_);
    std::copy(currents[item].begin(), currents[item].end(),
              h_currents_.get() + item * static_cast<size_t>(circuit_count_));
  }
  const size_t current_count = count * static_cast<size_t>(circuit_count_);
  const bool currents_changed = uploaded_currents_.size() != current_count ||
                                std::memcmp(uploaded_currents_.data(), h_currents_.get(),
                                            current_count * sizeof(double)) != 0;
  const ProfileClock::time_point start = ProfileClock::now();
  if ((status = solver->PrepareDeviceBatch(count)) != Status::kOk)
    return status;
  bool drained = false;
  struct DrainOnExit {
    bool &drained;
    ~DrainOnExit()
    {
      if (!drained)
        cudaStreamSynchronize(nullptr);
    }
  } drain{drained};
  *h_valid_.get() = 1;
  if (cudaMemcpyAsync(d_old_.get(), h_old_.get(), count * node_count_ * sizeof(double),
                      cudaMemcpyHostToDevice) != cudaSuccess ||
      (currents_changed &&
       cudaMemcpyAsync(d_currents_.get(), h_currents_.get(), current_count * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) ||
      cudaMemcpyAsync(d_valid_.get(), h_valid_.get(), sizeof(int), cudaMemcpyHostToDevice) !=
          cudaSuccess)
    return Status::kInternalError;
  const dim3 blocks_elements(static_cast<unsigned int>((triangles + 255) / 256),
                             static_cast<unsigned int>(count));
  const dim3 blocks_values(static_cast<unsigned int>((nnz_ + 255) / 256),
                           static_cast<unsigned int>(count));
  const dim3 blocks_rows(static_cast<unsigned int>((free_count_ + 255) / 256),
                         static_cast<unsigned int>(count));
  DeviceNonlinearElementKernel<<<blocks_elements, 256>>>(
      d_triangles_.get(), triangles, d_old_.get(), node_count_, d_curve_b_.get(), d_curve_h_.get(),
      d_curve_slope_.get(), use_newton, d_bx_.get(), d_by_.get(), d_nu_.get(), d_derivative_.get(),
      quadrature_stride_, d_valid_.get());
  DeviceNonlinearValuesKernel<<<blocks_values, 256>>>(
      d_triangles_.get(), d_matrix_offsets_.get(), d_matrix_entries_.get(), nnz_,
      d_base_values_.get(), d_bx_.get(), d_by_.get(), d_nu_.get(), d_derivative_.get(), triangles,
      quadrature_stride_, use_newton, solver->device_batch_values());
  DeviceNonlinearRhsKernel<<<blocks_rows, 256>>>(
      d_triangles_.get(), d_rhs_offsets_.get(), d_rhs_entries_.get(), free_count_, d_pm_rhs_.get(),
      d_circuit_rhs_.get(), circuit_count_, d_currents_.get(), d_age_rhs_.get(), d_old_.get(),
      d_boundary_values_.get(), d_bx_.get(), d_by_.get(), d_nu_.get(), d_derivative_.get(),
      triangles, quadrature_stride_, node_count_, use_newton, solver->device_batch_rhs());
  DeviceNonlinearDiagonalKernel<<<blocks_rows, 256>>>(
      d_diagonal_slots_.get(), free_count_, nnz_, solver->device_batch_values(),
      solver->device_batch_diagonal(), solver->device_batch_inverse_diagonal(), d_valid_.get());
  if (cudaGetLastError() != cudaSuccess ||
      cudaMemcpyAsync(h_valid_.get(), d_valid_.get(), sizeof(int), cudaMemcpyDeviceToHost) !=
          cudaSuccess ||
      cudaStreamSynchronize(nullptr) != cudaSuccess)
    return Status::kInternalError;
  drained = true;
  if (seconds != nullptr)
    *seconds += ProfileSecondsSince(start);
  const bool valid = *h_valid_.get() != 0;
  solver->set_device_batch_inverse_diagonal_usable(valid);
  if (valid && currents_changed)
    uploaded_currents_.assign(h_currents_.get(), h_currents_.get() + current_count);
  return valid ? Status::kOk : Status::kAssemblyFailed;
}

bool DeviceNonlinearAssemblyPlan::initialized() const { return initialized_; }

const Assembly &DeviceNonlinearAssemblyPlan::symbolic_assembly() const { return symbolic_; }

Status DeviceNonlinearAssemblyPlan::Initialize(const NonlinearModel &model)
{
  PreparedNonlinearAssembly prepared;
  const Status status = PrepareNonlinearAssembly(model, &prepared);
  return status == Status::kOk ? Initialize(std::move(prepared)) : status;
}

Status DeviceNonlinearAssemblyPlan::Initialize(PreparedNonlinearAssembly prepared)
{
  node_count_ = prepared.node_count;
  circuit_count_ = prepared.circuit_count;
  quadrature_stride_ = prepared.quadrature_stride;
  free_count_ = prepared.symbolic.free_nodes.size();
  nnz_ = prepared.symbolic.values.size();
  state_capacity_ = 0;
  const auto &triangles = prepared.triangles;
  const auto &curve_b = prepared.curve_b;
  const auto &curve_h = prepared.curve_h;
  const auto &curve_slope = prepared.curve_slope;
  const auto &matrix_offsets = prepared.matrix_offsets;
  const auto &matrix_entries = prepared.matrix_entries;
  const auto &rhs_offsets = prepared.rhs_offsets;
  const auto &rhs_entries = prepared.rhs_entries;
  const auto &age_values = prepared.age_values;
  const auto &age_rhs = prepared.age_rhs;
  const auto &pm_rhs = prepared.pm_rhs;
  const auto &circuit_rhs = prepared.circuit_rhs;
  const auto &boundary_values = prepared.boundary_values;
  const auto &diagonal_slots = prepared.diagonal_slots;
  Status status = Status::kOk;
  if ((status = d_triangles_.allocate(triangles.size())) != Status::kOk ||
      (status = d_curve_b_.allocate(curve_b.size())) != Status::kOk ||
      (status = d_curve_h_.allocate(curve_h.size())) != Status::kOk ||
      (status = d_curve_slope_.allocate(curve_slope.size())) != Status::kOk ||
      (status = d_matrix_offsets_.allocate(matrix_offsets.size())) != Status::kOk ||
      (status = d_matrix_entries_.allocate(matrix_entries.size())) != Status::kOk ||
      (status = d_rhs_offsets_.allocate(rhs_offsets.size())) != Status::kOk ||
      (status = d_rhs_entries_.allocate(rhs_entries.size())) != Status::kOk ||
      (status = d_base_values_.allocate(age_values.size())) != Status::kOk ||
      (status = d_age_rhs_.allocate(age_rhs.size())) != Status::kOk ||
      (status = d_pm_rhs_.allocate(pm_rhs.size())) != Status::kOk ||
      (status = d_circuit_rhs_.allocate(circuit_rhs.size())) != Status::kOk ||
      (status = d_boundary_values_.allocate(boundary_values.size())) != Status::kOk ||
      (status = d_diagonal_slots_.allocate(diagonal_slots.size())) != Status::kOk ||
      (status = d_valid_.allocate(1)) != Status::kOk)
    return status;
  if ((status = CopyToDevice(d_triangles_.get(), triangles.data(),
                             triangles.size() * sizeof(DeviceNonlinearTriangle))) != Status::kOk ||
      (status = CopyToDevice(d_curve_b_.get(), curve_b.data(), curve_b.size() * sizeof(double))) !=
          Status::kOk ||
      (status = CopyToDevice(d_curve_h_.get(), curve_h.data(), curve_h.size() * sizeof(double))) !=
          Status::kOk ||
      (status = CopyToDevice(d_curve_slope_.get(), curve_slope.data(),
                             curve_slope.size() * sizeof(double))) != Status::kOk ||
      (status = CopyToDevice(d_matrix_offsets_.get(), matrix_offsets.data(),
                             matrix_offsets.size() * sizeof(int32_t))) != Status::kOk ||
      (status = CopyToDevice(d_matrix_entries_.get(), matrix_entries.data(),
                             matrix_entries.size() * sizeof(DeviceMatrixContributor))) !=
          Status::kOk ||
      (status = CopyToDevice(d_rhs_offsets_.get(), rhs_offsets.data(),
                             rhs_offsets.size() * sizeof(int32_t))) != Status::kOk ||
      (status = CopyToDevice(d_rhs_entries_.get(), rhs_entries.data(),
                             rhs_entries.size() * sizeof(DeviceRhsContributor))) != Status::kOk ||
      (status = CopyToDevice(d_base_values_.get(), age_values.data(),
                             age_values.size() * sizeof(double))) != Status::kOk ||
      (status = CopyToDevice(d_age_rhs_.get(), age_rhs.data(), age_rhs.size() * sizeof(double))) !=
          Status::kOk ||
      (status = CopyToDevice(d_pm_rhs_.get(), pm_rhs.data(), pm_rhs.size() * sizeof(double))) !=
          Status::kOk ||
      (status = CopyToDevice(d_circuit_rhs_.get(), circuit_rhs.data(),
                             circuit_rhs.size() * sizeof(double))) != Status::kOk ||
      (status = CopyToDevice(d_boundary_values_.get(), boundary_values.data(),
                             boundary_values.size() * sizeof(double))) != Status::kOk ||
      (status = CopyToDevice(d_diagonal_slots_.get(), diagonal_slots.data(),
                             diagonal_slots.size() * sizeof(int32_t))) != Status::kOk)
    return status;
  symbolic_ = std::move(prepared.symbolic);
  initialized_ = true;
  return Status::kOk;
}

} // namespace gpu_femm

#include "physics/nonlinear_assembly.cuh"
#include "core/profile.h"
#include "axisymmetric_p1.h"
#include "native_airgap_matrix.h"

#include <algorithm>
#include <cmath>
#include <map>
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
      (status = d_derivative_.allocate(count * triangles * quadrature_stride_)) != Status::kOk)
    return status;
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
  std::vector<double> packed_states(count * node_count_),
      packed_currents(count * static_cast<size_t>(circuit_count_));
  for (size_t item = 0; item < count; ++item) {
    if (states[item].size() != node_count_ ||
        currents[item].size() != static_cast<size_t>(circuit_count_))
      return Status::kInvalidArgument;
    std::copy(states[item].begin(), states[item].end(), packed_states.begin() + item * node_count_);
    std::copy(currents[item].begin(), currents[item].end(),
              packed_currents.begin() + item * static_cast<size_t>(circuit_count_));
  }
  Status status = EnsureStateCapacity(count);
  if (status != Status::kOk)
    return status;
  const ProfileClock::time_point start = ProfileClock::now();
  if ((status = solver->PrepareDeviceBatch(count)) != Status::kOk ||
      (status = CopyToDevice(d_old_.get(), packed_states.data(),
                             packed_states.size() * sizeof(double))) != Status::kOk ||
      (status = CopyToDevice(d_currents_.get(), packed_currents.data(),
                             packed_currents.size() * sizeof(double))) != Status::kOk)
    return status;
  int valid = 1;
  if (cudaMemcpy(d_valid_.get(), &valid, sizeof(valid), cudaMemcpyHostToDevice) != cudaSuccess)
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
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(&valid, d_valid_.get(), sizeof(valid), cudaMemcpyDeviceToHost) != cudaSuccess)
    return Status::kInternalError;
  if (seconds != nullptr)
    *seconds += ProfileSecondsSince(start);
  solver->set_device_batch_inverse_diagonal_usable(valid != 0);
  return valid != 0 ? Status::kOk : Status::kAssemblyFailed;
}

bool DeviceNonlinearAssemblyPlan::initialized() const { return initialized_; }

const Assembly &DeviceNonlinearAssemblyPlan::symbolic_assembly() const { return symbolic_; }

Status DeviceNonlinearAssemblyPlan::Initialize(const NonlinearModel &model)
{
  if (ValidateNonlinearModel(model) != Status::kOk)
    return Status::kInvalidArgument;
  node_count_ = model.nodes.size();
  circuit_count_ = model.circuit_count;
  quadrature_stride_ = model.problem_type == MagneticProblemType::kPlanar ? 1 : 7;
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
  free_count_ = dofs.free_roots.size();
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
  nnz_ = symbolic.values.size();
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
  symbolic_ = std::move(symbolic);
  initialized_ = true;
  return Status::kOk;
}

} // namespace gpu_femm

#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace gpu_femm
{

bool BuildAirGapStencil(const AirGapElement &age, size_t k, int32_t node[10], double weight[10])
{
  const size_t elements = age.quad_points.size() - 1;
  if (elements < 2 || k >= elements)
    return false;
  const size_t previous = k == 0 ? elements - 1 : k - 1;
  const size_t next = k + 1;
  const size_t next2 = k + 2 > elements ? 1 : k + 2;
  const AirGapQuadPoint &before = age.quad_points[previous];
  const AirGapQuadPoint &current = age.quad_points[k];
  const AirGapQuadPoint &after = age.quad_points[next];
  const AirGapQuadPoint &after2 = age.quad_points[next2];
  const int local_node[10] = {before.node[0], current.node[0], current.node[1], after.node[1],
                              after2.node[1], before.node[2],  current.node[2], current.node[3],
                              after.node[3],  after2.node[3]};
  const double local_weight[10] = {
      before.weight[0], current.weight[0], current.weight[1], after.weight[1], after2.weight[1],
      before.weight[2], current.weight[2], current.weight[3], after.weight[3], after2.weight[3]};
  std::copy(local_node, local_node + 10, node);
  std::copy(local_weight, local_weight + 10, weight);
  if (age.antiperiodic && k == 0) {
    weight[0] = -weight[0];
    weight[5] = -weight[5];
  }
  if (age.antiperiodic && k + 1 == elements) {
    weight[4] = -weight[4];
    weight[9] = -weight[9];
  }
  return true;
}

FrozenPostprocessResult ComputeAirGapElementPostprocess(const NonlinearModel &model,
                                                        const NonlinearSolveResult &solution,
                                                        const FrozenPostprocessOptions &options)
{
  FrozenPostprocessResult result;
  result.status = Status::kOk;
  result.force_x_n = 0.0;
  result.force_y_n = 0.0;
  result.torque_nm = 0.0;
  if (model.air_gap_elements.empty() || solution.a_wb_per_m.size() != model.nodes.size()) {
    result.status = Status::kInvalidArgument;
    return result;
  }
  struct AirGapField {
    const AirGapElement *age = nullptr;
    std::vector<double> br, bt;
    std::vector<int32_t> harmonic_order;
    std::vector<double> br_cos, br_sin;
  };
  std::vector<AirGapField> fields;
  for (const AirGapElement &age : model.air_gap_elements) {
    const size_t elements = age.quad_points.size() - 1;
    const double dt = (3.141592653589793238462643383279502884 / 180.0) * age.arc_length_deg /
                      static_cast<double>(elements);
    const double radius = 0.5 * (age.inner_radius_m + age.outer_radius_m);
    const double dr = age.outer_radius_m - age.inner_radius_m;
    if (!(dt > 0.0) || !(radius > 0.0) || !(dr > 0.0)) {
      result.status = Status::kMeshInvalid;
      return result;
    }
    AirGapField field;
    field.age = &age;
    field.br.resize(elements);
    field.bt.resize(elements);
    for (size_t k = 0; k < elements; ++k) {
      int32_t node[10];
      double weight[10], a[10];
      if (!BuildAirGapStencil(age, k, node, weight)) {
        result.status = Status::kMeshInvalid;
        return result;
      }
      for (int local = 0; local < 10; ++local)
        a[local] = solution.a_wb_per_m[node[local]] * weight[local];
      const double ci = age.inner_shift, co = age.outer_shift;
      field.br[k] = (-(ci * a[1]) - 2 * a[2] + 2 * a[3] + ci * (a[2] + a[3] - a[4]) -
                     ci * ci * ci * (a[0] - 4 * a[1] + 6 * a[2] - 4 * a[3] + a[4]) +
                     ci * ci * (a[0] - 5 * a[1] + 9 * a[2] - 7 * a[3] + 2 * a[4]) - 2 * a[7] +
                     2 * a[8] + co * (-a[6] + a[7] + a[8] - a[9]) -
                     co * co * co * (a[5] - 4 * a[6] + 6 * a[7] - 4 * a[8] + a[9]) +
                     co * co * (a[5] - 5 * a[6] + 9 * a[7] - 7 * a[8] + 2 * a[9])) /
                    (4 * dt * radius);
      field.bt[k] =
          (ci * a[1] + 2 * a[2] + 2 * a[3] -
           ci * ci * (a[0] - 3 * a[1] + a[2] + 3 * a[3] - 2 * a[4]) + ci * (a[2] - a[3] - a[4]) +
           ci * ci * ci * (a[0] - 2 * a[1] + 2 * a[3] - a[4]) - co * a[6] +
           (-2 + co) * (1 + co) * a[7] - 2 * a[8] +
           co * (a[8] + co * (a[5] - 3 * a[6] + 3 * a[8] - 2 * a[9]) + a[9] +
                 co * co * (-a[5] + 2 * a[6] - 2 * a[8] + a[9]))) /
          (4 * dr);
      if (!std::isfinite(field.br[k]) || !std::isfinite(field.bt[k])) {
        result.status = Status::kNumericalNonfinite;
        return result;
      }
      const double theta = (static_cast<double>(k) + 0.5) * dt;
      const double normal = field.br[k] * field.br[k] - field.bt[k] * field.bt[k];
      const double shear = 2.0 * field.br[k] * field.bt[k];
      const double scale = model.depth_m * radius * dt / (2.0 * kMu0);
      result.force_x_n += scale * (normal * std::cos(theta) - shear * std::sin(theta));
      result.force_y_n += scale * (normal * std::sin(theta) + shear * std::cos(theta));
      result.torque_nm += model.depth_m * radius * radius * dt * field.br[k] * field.bt[k] / kMu0;
    }
    // FEMM's mo_getgapb does not return the nearest AGE element value.  It
    // reconstructs the radial field from the same discrete Fourier series
    // built by femmviewDoc when it reads an .ans file.
    const size_t harmonic_count = age.antiperiodic ? (elements + 1) / 2 : elements / 2 + 1;
    const int32_t harmonic_base =
        static_cast<int32_t>(std::lround((age.antiperiodic ? 180.0 : 360.0) / age.arc_length_deg));
    field.harmonic_order.resize(harmonic_count);
    field.br_cos.assign(harmonic_count, 0.0);
    field.br_sin.assign(harmonic_count, 0.0);
    for (size_t harmonic = 0; harmonic < harmonic_count; ++harmonic) {
      const int32_t order = age.antiperiodic
                                ? harmonic_base * static_cast<int32_t>(2 * harmonic + 1)
                                : harmonic_base * static_cast<int32_t>(harmonic);
      field.harmonic_order[harmonic] = order;
      for (size_t k = 0; k < elements; ++k) {
        const double phase = (static_cast<double>(k) + 0.5) * dt * static_cast<double>(order);
        field.br_cos[harmonic] += field.br[k] * std::cos(phase);
        field.br_sin[harmonic] += field.br[k] * std::sin(phase);
      }
      const bool dc_or_nyquist =
          order == 0 ||
          (!age.antiperiodic && harmonic + 1 == harmonic_count && (elements % 2) == 0);
      const double normalization =
          dc_or_nyquist ? static_cast<double>(elements) : static_cast<double>(elements) / 2.0;
      field.br_cos[harmonic] /= normalization;
      field.br_sin[harmonic] /= normalization;
    }
    fields.push_back(std::move(field));
  }
  for (const double angle : options.airgap_angles_rad) {
    if (!std::isfinite(angle)) {
      result.status = Status::kInvalidArgument;
      return result;
    }
    const AirGapField &field = fields.front();
    double radial_b_t = 0.0;
    for (size_t harmonic = 0; harmonic < field.harmonic_order.size(); ++harmonic) {
      const double phase = static_cast<double>(field.harmonic_order[harmonic]) * angle;
      radial_b_t +=
          field.br_cos[harmonic] * std::cos(phase) + field.br_sin[harmonic] * std::sin(phase);
    }
    result.airgap_samples.push_back({angle, radial_b_t});
  }
  if (!std::isfinite(result.force_x_n) || !std::isfinite(result.force_y_n) ||
      !std::isfinite(result.torque_nm)) {
    result.status = Status::kNumericalNonfinite;
  }
  return result;
}

TangentPostprocessResult ComputeAirGapElementTangentPostprocess(
    const NonlinearModel &base_model, const NonlinearModel &tangent_model,
    const NonlinearSolveResult &base, const NonlinearSolveResult &derivative,
    const FrozenPostprocessOptions &options)
{
  TangentPostprocessResult result;
  if (base_model.air_gap_elements.empty() ||
      base_model.air_gap_elements.size() != tangent_model.air_gap_elements.size() ||
      base.a_wb_per_m.size() != base_model.nodes.size() ||
      derivative.a_wb_per_m.size() != tangent_model.nodes.size() ||
      base_model.nodes.size() != tangent_model.nodes.size()) {
    result.status = Status::kInvalidArgument;
    return result;
  }
  std::vector<double> first_dbr;
  const AirGapElement *first_tangent_age = nullptr;
  for (size_t age_index = 0; age_index < base_model.air_gap_elements.size(); ++age_index) {
    const AirGapElement &age = base_model.air_gap_elements[age_index];
    const AirGapElement &tangent_age = tangent_model.air_gap_elements[age_index];
    const size_t elements = age.quad_points.size() - 1;
    if (tangent_age.quad_points.size() != age.quad_points.size() ||
        tangent_age.arc_length_deg != age.arc_length_deg ||
        tangent_age.inner_radius_m != age.inner_radius_m ||
        tangent_age.outer_radius_m != age.outer_radius_m ||
        tangent_age.inner_shift != age.inner_shift || tangent_age.outer_shift != age.outer_shift) {
      result.status = Status::kInvalidArgument;
      return result;
    }
    const double dt = (3.141592653589793238462643383279502884 / 180.0) * age.arc_length_deg /
                      static_cast<double>(elements);
    const double radius = 0.5 * (age.inner_radius_m + age.outer_radius_m);
    const double dr = age.outer_radius_m - age.inner_radius_m;
    std::vector<double> dbr(elements, 0.0);
    for (size_t k = 0; k < elements; ++k) {
      int32_t node[10], tangent_node[10];
      double weight[10], tangent_weight[10], a[10], da[10];
      if (!BuildAirGapStencil(age, k, node, weight) ||
          !BuildAirGapStencil(tangent_age, k, tangent_node, tangent_weight)) {
        result.status = Status::kMeshInvalid;
        return result;
      }
      for (int local = 0; local < 10; ++local) {
        a[local] = base.a_wb_per_m[node[local]] * weight[local];
        da[local] = derivative.a_wb_per_m[tangent_node[local]] * tangent_weight[local];
      }
      const auto age_field = [&](const double value[10], double *br, double *bt) {
        const double ci = age.inner_shift, co = age.outer_shift;
        *br = (-(ci * value[1]) - 2 * value[2] + 2 * value[3] +
               ci * (value[2] + value[3] - value[4]) -
               ci * ci * ci * (value[0] - 4 * value[1] + 6 * value[2] - 4 * value[3] + value[4]) +
               ci * ci * (value[0] - 5 * value[1] + 9 * value[2] - 7 * value[3] + 2 * value[4]) -
               2 * value[7] + 2 * value[8] + co * (-value[6] + value[7] + value[8] - value[9]) -
               co * co * co * (value[5] - 4 * value[6] + 6 * value[7] - 4 * value[8] + value[9]) +
               co * co * (value[5] - 5 * value[6] + 9 * value[7] - 7 * value[8] + 2 * value[9])) /
              (4 * dt * radius);
        *bt = (ci * value[1] + 2 * value[2] + 2 * value[3] -
               ci * ci * (value[0] - 3 * value[1] + value[2] + 3 * value[3] - 2 * value[4]) +
               ci * (value[2] - value[3] - value[4]) +
               ci * ci * ci * (value[0] - 2 * value[1] + 2 * value[3] - value[4]) - co * value[6] +
               (-2 + co) * (1 + co) * value[7] - 2 * value[8] +
               co * (value[8] + co * (value[5] - 3 * value[6] + 3 * value[8] - 2 * value[9]) +
                     value[9] + co * co * (-value[5] + 2 * value[6] - 2 * value[8] + value[9]))) /
              (4 * dr);
      };
      double br = 0.0, bt = 0.0, dbt = 0.0;
      age_field(a, &br, &bt);
      age_field(da, &dbr[k], &dbt);
      const double theta = (static_cast<double>(k) + 0.5) * dt;
      const double dnormal = 2.0 * (br * dbr[k] - bt * dbt);
      const double dshear = 2.0 * (dbr[k] * bt + br * dbt);
      const double scale = base_model.depth_m * radius * dt / (2.0 * kMu0);
      result.force_x_n_per_a += scale * (dnormal * std::cos(theta) - dshear * std::sin(theta));
      result.force_y_n_per_a += scale * (dnormal * std::sin(theta) + dshear * std::cos(theta));
      result.torque_nm_per_a +=
          base_model.depth_m * radius * radius * dt * (dbr[k] * bt + br * dbt) / kMu0;
    }
    if (first_tangent_age == nullptr) {
      first_tangent_age = &tangent_age;
      first_dbr = std::move(dbr);
    }
  }
  if (first_tangent_age != nullptr) {
    const size_t elements = first_dbr.size();
    const size_t harmonic_count =
        first_tangent_age->antiperiodic ? (elements + 1) / 2 : elements / 2 + 1;
    const int32_t harmonic_base = static_cast<int32_t>(std::lround(
        (first_tangent_age->antiperiodic ? 180.0 : 360.0) / first_tangent_age->arc_length_deg));
    const double dt = (3.141592653589793238462643383279502884 / 180.0) *
                      first_tangent_age->arc_length_deg / static_cast<double>(elements);
    std::vector<int32_t> order(harmonic_count);
    std::vector<double> cosine(harmonic_count, 0.0), sine(harmonic_count, 0.0);
    for (size_t harmonic = 0; harmonic < harmonic_count; ++harmonic) {
      order[harmonic] = first_tangent_age->antiperiodic
                            ? harmonic_base * static_cast<int32_t>(2 * harmonic + 1)
                            : harmonic_base * static_cast<int32_t>(harmonic);
      for (size_t k = 0; k < elements; ++k) {
        const double phase =
            (static_cast<double>(k) + 0.5) * dt * static_cast<double>(order[harmonic]);
        cosine[harmonic] += first_dbr[k] * std::cos(phase);
        sine[harmonic] += first_dbr[k] * std::sin(phase);
      }
      const bool dc_or_nyquist =
          order[harmonic] == 0 || (!first_tangent_age->antiperiodic &&
                                   harmonic + 1 == harmonic_count && (elements % 2) == 0);
      const double normalization =
          dc_or_nyquist ? static_cast<double>(elements) : static_cast<double>(elements) / 2.0;
      cosine[harmonic] /= normalization;
      sine[harmonic] /= normalization;
    }
    for (const double angle : options.airgap_angles_rad) {
      double radial = 0.0;
      for (size_t harmonic = 0; harmonic < harmonic_count; ++harmonic) {
        const double phase = static_cast<double>(order[harmonic]) * angle;
        radial += cosine[harmonic] * std::cos(phase) + sine[harmonic] * std::sin(phase);
      }
      result.airgap_radial_flux_density_t_per_a.push_back(radial);
    }
  }
  result.status = std::isfinite(result.force_x_n_per_a) && std::isfinite(result.force_y_n_per_a) &&
                          std::isfinite(result.torque_nm_per_a)
                      ? Status::kOk
                      : Status::kNumericalNonfinite;
  return result;
}

} // namespace gpu_femm

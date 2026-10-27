#pragma once

#include "physics/nonlinear_model.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace gpu_femm
{

struct AirgapSample {
  double angle_rad = 0.0;
  double radial_b_t = std::numeric_limits<double>::quiet_NaN();
};

struct FrozenPostprocessOptions {
  // Fixture mapping is explicit: label 1 is PM/selected rotor, label 3 is
  // default air. This avoids treating an unassigned material as air.
  int32_t selected_material_label = 1;
  int32_t air_material_label = 3;
  // Artifact mode selects complete FEMM label groups; -1 retains material labels.
  int32_t selected_group_number = -1;
  int32_t air_group_number = -1;
  double airgap_radius_m = 0.0;
  std::vector<double> airgap_angles_rad;
  int max_mask_iterations = 256;
  double mask_relative_tolerance = 1e-12;
};

struct FrozenPostprocessResult {
  Status status = Status::kInternalError;
  double force_x_n = std::numeric_limits<double>::quiet_NaN();
  double force_y_n = std::numeric_limits<double>::quiet_NaN();
  double torque_nm = std::numeric_limits<double>::quiet_NaN();
  std::vector<AirgapSample> airgap_samples;
};

using EdgeKey = std::pair<int32_t, int32_t>;

struct FrozenAirgapProbe {
  int32_t containing_element = -1;
  double lambda[3] = {};
  double cosine = 0.0;
  double sine = 0.0;
};

struct FrozenPostprocessPlan {
  Status status = Status::kInternalError;
  size_t node_count = 0;
  size_t triangle_count = 0;
  int32_t selected_group_number = -1;
  int32_t air_group_number = -1;
  int32_t selected_material_label = -1;
  int32_t air_material_label = -1;
  double airgap_radius_m = 0.0;
  std::vector<double> airgap_angles_rad;
  std::vector<double> mask;
  std::map<EdgeKey, std::vector<int32_t>> edge_elements;
  std::vector<std::vector<int32_t>> incident;
  std::vector<FrozenAirgapProbe> probes;
  std::vector<double> mask_gradient_x;
  std::vector<double> mask_gradient_y;
  std::vector<double> element_weight;
  std::vector<double> element_center_x;
  std::vector<double> element_center_y;
};

struct TangentPostprocessResult {
  Status status = Status::kInternalError;
  double force_x_n_per_a = 0.0;
  double force_y_n_per_a = 0.0;
  double torque_nm_per_a = 0.0;
  std::vector<double> airgap_radial_flux_density_t_per_a;
};

bool IsSelectedPostprocessMaterial(const NonlinearMaterial &material,
                                   const FrozenPostprocessOptions &options, int32_t material_index);
bool IsAirPostprocessMaterial(const NonlinearMaterial &material,
                              const FrozenPostprocessOptions &options, int32_t material_index);

EdgeKey CanonicalEdge(int32_t first, int32_t second);
bool TriangleBarycentric(const Node p[3], double x_m, double y_m, double lambda[3]);

Status BuildWeightedStressMask(const NonlinearModel &model, const FrozenPostprocessOptions &options,
                               std::vector<double> *mask);

bool SmoothedElementB(const NonlinearModel &model, const std::vector<double> &a,
                      const std::vector<double> &bx, const std::vector<double> &by,
                      const std::vector<std::vector<int32_t>> &incident,
                      const std::map<EdgeKey, std::vector<int32_t>> &edge_elements, int32_t element,
                      double nodal_bx[3], double nodal_by[3]);

bool FrozenPostprocessPlanMatches(const FrozenPostprocessPlan &plan, const NonlinearModel &model,
                                  const FrozenPostprocessOptions &options);
Status BuildFrozenPostprocessPlan(const NonlinearModel &model,
                                  const FrozenPostprocessOptions &options,
                                  FrozenPostprocessPlan *plan);
FrozenPostprocessResult
ComputeFrozenPostprocess(const NonlinearModel &model, const NonlinearSolveResult &solution,
                         const FrozenPostprocessOptions &options,
                         const FrozenPostprocessPlan *reusable_plan = nullptr);

bool BuildAirGapStencil(const AirGapElement &age, size_t k, int32_t node[10], double weight[10]);
FrozenPostprocessResult ComputeAirGapElementPostprocess(const NonlinearModel &model,
                                                        const NonlinearSolveResult &solution,
                                                        const FrozenPostprocessOptions &options);

TangentPostprocessResult ComputeFrozenTangentPostprocess(
    const NonlinearModel &model, const NonlinearSolveResult &base,
    const NonlinearSolveResult &derivative, const FrozenPostprocessOptions &options,
    const std::vector<double> &mask, const FrozenPostprocessPlan *reusable_plan = nullptr);
TangentPostprocessResult ComputeAirGapElementTangentPostprocess(
    const NonlinearModel &base_model, const NonlinearModel &tangent_model,
    const NonlinearSolveResult &base, const NonlinearSolveResult &derivative,
    const FrozenPostprocessOptions &options);

} // namespace gpu_femm

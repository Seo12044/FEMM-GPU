#include "app/motor_batch.h"
#include "app/sample_adapter.h"
#include "core/profile.h"

#include <utility>

namespace gpu_femm
{
MotorBatchPostprocessResult PostprocessMotorBatchGroup(
    const MotorBatchRequest &request, const std::vector<MotorBatchPreparedItem> &prepared,
    const std::vector<size_t> &group, std::vector<NonlinearSolveResult> chunk_solutions,
    FrozenPostprocessPlan prepared_plan, NonlinearP1Solver &cached_solver,
    const NonlinearOptions &options, int *launches)
{
  MotorBatchPostprocessResult result;
  result.responses.resize(request.items.size());
  auto &timing = result.timing;
  auto &responses = result.responses;
  for (size_t index : group)
    result.indices.push_back(prepared[index].request_index);
  for (size_t local = 0; local < group.size(); ++local) {
    const MotorBatchPreparedItem &prepared_item = prepared[group[local]];
    const size_t index = prepared_item.request_index;
    const MotorSampleRequest &item = request.items[index].request;
    NonlinearSolveResult &solution = chunk_solutions[local];
    timing.nonlinear_pcg_iterations += solution.linear_iterations;
    if (solution.info.status != Status::kOk) {
      responses[index] = MakeMotorBatchResponseDto(solution.info.status, item, &solution, nullptr);
      continue;
    }
    FrozenPostprocessOptions post_options;
    post_options.selected_group_number = item.selected_group_number;
    post_options.air_group_number = item.air_group_number;
    post_options.max_mask_iterations = 4096;
    post_options.airgap_radius_m = item.airgap_radius_mm * 1e-3;
    for (double angle : item.airgap_angles_deg)
      post_options.airgap_angles_rad.push_back(angle * 3.141592653589793238462643383279502884 /
                                               180.0);
    FrozenPostprocessOptions base_post_options = post_options;
    std::vector<int8_t> base_airgap_repeat_signs;
    if (item.sector_performance.enabled && prepared_item.artifact->has_native_age_sector) {
      base_airgap_repeat_signs = MapNativeAgeSectorAirgapAnglesToRepresentative(
          *prepared_item.artifact, &base_post_options);
    } else if (item.sector_performance.enabled && !prepared_item.artifact->has_sliding_band) {
      base_airgap_repeat_signs = MapSectorAirgapAnglesToRepresentative(
          prepared_item.artifact->sector_start_angle_deg * 3.141592653589793238462643383279502884 /
              180.0,
          prepared_item.artifact->sector_boundary_sign, &base_post_options);
    }
    const ProfileClock::time_point postprocess_start = ProfileClock::now();
    FrozenPostprocessPlan item_frozen_plan;
    const FrozenPostprocessPlan *frozen_plan = nullptr;
    if (!prepared_item.artifact->has_sliding_band) {
      if (FrozenPostprocessPlanMatches(prepared_plan, prepared_item.model, base_post_options)) {
        frozen_plan = &prepared_plan;
      } else {
        const ProfileClock::time_point plan_start = ProfileClock::now();
        BuildFrozenPostprocessPlan(prepared_item.model, base_post_options, &item_frozen_plan);
        const double plan_seconds = ProfileSecondsSince(plan_start);
        timing.postprocess_plan_compute_seconds += plan_seconds;
        timing.postprocess_plan_seconds += plan_seconds;
        frozen_plan = &item_frozen_plan;
      }
    }
    const ProfileClock::time_point base_postprocess_start = ProfileClock::now();
    FrozenPostprocessResult postprocess;
    if (prepared_item.artifact->has_sliding_band) {
      postprocess =
          ComputeAirGapElementPostprocess(prepared_item.model, solution, base_post_options);
    } else if (frozen_plan->status != Status::kOk) {
      postprocess.status = frozen_plan->status;
    } else {
      postprocess =
          ComputeFrozenPostprocess(prepared_item.model, solution, base_post_options, frozen_plan);
    }
    timing.base_postprocess_seconds += ProfileSecondsSince(base_postprocess_start);
    if (postprocess.status == Status::kOk && !base_airgap_repeat_signs.empty() &&
        !ApplySectorAirgapRepeatSigns(base_airgap_repeat_signs, &postprocess))
      postprocess.status = Status::kInternalError;
    timing.postprocess_seconds += ProfileSecondsSince(postprocess_start);
    responses[index] =
        MakeMotorBatchResponseDto(postprocess.status, item, &solution,
                                  postprocess.status == Status::kOk ? &postprocess : nullptr);
    if (item.sector_performance.enabled) {
      ApplyBaseSectorReconstruction(*prepared_item.artifact, item, &responses[index]);
      responses[index].sector_reconstruction_status = responses[index].status;
    }
    if (postprocess.status == Status::kOk && item.tangent_requested) {
      responses[index].tangent_status = Status::kOk;
      FrozenPostprocessOptions tangent_post_options = post_options;
      std::vector<int8_t> tangent_airgap_repeat_signs;
      if (item.sector_performance.enabled && !prepared_item.artifact->has_sliding_band) {
        tangent_airgap_repeat_signs = MapSectorAirgapAnglesToRepresentative(
            prepared_item.artifact->sector_start_angle_deg *
                3.141592653589793238462643383279502884 / 180.0,
            static_cast<int8_t>((prepared_item.artifact->suspension_spatial_order % 2) == 0 ? 1
                                                                                            : -1),
            &tangent_post_options);
      }
      FrozenPostprocessPlan tangent_frozen_plan;
      const FrozenPostprocessPlan *tangent_plan = frozen_plan;
      if (!prepared_item.artifact->has_sliding_band &&
          !FrozenPostprocessPlanMatches(*tangent_plan, prepared_item.model, tangent_post_options)) {
        responses[index].tangent_status = BuildFrozenPostprocessPlan(
            prepared_item.model, tangent_post_options, &tangent_frozen_plan);
        tangent_plan = &tangent_frozen_plan;
      }
      NonlinearP1Solver sector_tangent_solver;
      NonlinearP1Solver *tangent_solver = &cached_solver;
      if (item.sector_performance.enabled) {
        if (prepared_item.tangent_model.nodes.empty()) {
          responses[index].tangent_status = Status::kInvalidArgument;
        } else {
          responses[index].tangent_status =
              sector_tangent_solver.Initialize(prepared_item.tangent_model);
          tangent_solver = &sector_tangent_solver;
        }
      }
      const ProfileClock::time_point tangent_operator_start = ProfileClock::now();
      std::vector<TangentSolveResult> tangent_results =
          responses[index].tangent_status == Status::kOk
              ? tangent_solver->SolveTangentMultiRhs(solution, item.tangent_multi_rhs.rhs, options,
                                                     launches)
              : std::vector<TangentSolveResult>{};
      timing.tangent_operator_seconds += ProfileSecondsSince(tangent_operator_start);
      if (responses[index].tangent_status == Status::kOk &&
          tangent_results.size() != item.tangent_multi_rhs.rhs.size())
        responses[index].tangent_status = Status::kInternalError;
      if (item.sector_performance.enabled && !tangent_results.empty())
        responses[index].tangent_operator_evidence = tangent_results.front().operator_evidence;
      for (TangentSolveResult &tangent : tangent_results) {
        if (tangent.info.status != Status::kOk) {
          responses[index].tangent_status = tangent.info.status;
          continue;
        }
        const ProfileClock::time_point tangent_postprocess_start = ProfileClock::now();
        TangentPostprocessResult tangent_postprocess =
            prepared_item.artifact->has_sliding_band
                ? ComputeAirGapElementTangentPostprocess(
                      prepared_item.model,
                      item.sector_performance.enabled ? prepared_item.tangent_model
                                                      : prepared_item.model,
                      solution, tangent.field_derivative, tangent_post_options)
                : ComputeFrozenTangentPostprocess(prepared_item.model, solution,
                                                  tangent.field_derivative, tangent_post_options,
                                                  tangent_plan->mask, tangent_plan);
        const double tangent_postprocess_seconds = ProfileSecondsSince(tangent_postprocess_start);
        timing.postprocess_seconds += tangent_postprocess_seconds;
        timing.tangent_postprocess_seconds += tangent_postprocess_seconds;
        if (tangent_postprocess.status != Status::kOk) {
          responses[index].tangent_status = tangent_postprocess.status;
          continue;
        }
        if (!tangent_airgap_repeat_signs.empty() &&
            !ApplySectorTangentAirgapRepeatSigns(tangent_airgap_repeat_signs,
                                                 &tangent_postprocess)) {
          responses[index].tangent_status = Status::kInternalError;
          continue;
        }
        TangentRhsResponseDto dto;
        dto.name = tangent.name;
        dto.circuit_current_derivative_a_per_a = tangent.circuit_current_derivative_a_per_a;
        dto.force_x_n_per_a = tangent_postprocess.force_x_n_per_a;
        dto.force_y_n_per_a = tangent_postprocess.force_y_n_per_a;
        dto.torque_nm_per_a = tangent_postprocess.torque_nm_per_a;
        dto.circuit_flux_linkage_derivative_wb_per_a =
            tangent.field_derivative.circuit_flux_linkage_wb;
        dto.airgap_radial_flux_density_derivative_t_per_a =
            std::move(tangent_postprocess.airgap_radial_flux_density_t_per_a);
        dto.linear_convergence = tangent.info;
        if (item.sector_performance.enabled &&
            !ApplyTangentSectorReconstruction(*prepared_item.artifact, responses[index], &dto)) {
          responses[index].tangent_status = Status::kInternalError;
          continue;
        }
        responses[index].tangent_rhs.push_back(std::move(dto));
      }
      if (responses[index].tangent_status == Status::kOk &&
          responses[index].tangent_rhs.size() != item.tangent_multi_rhs.rhs.size())
        responses[index].tangent_status = Status::kInternalError;
      if (item.sector_performance.enabled)
        responses[index].sector_reconstruction_status = responses[index].tangent_status;
    }
  }
  return result;
}
} // namespace gpu_femm

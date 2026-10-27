#include "app/motor_batch.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

#include "io/mesh_artifact.h"
#include "io/sample_request.h"
#include "postprocess/postprocess.h"
#include "solvers/options.h"

namespace gpu_femm
{

std::vector<double> ReconstructSectorCircuitFluxLinkage(const std::vector<double> &sector_flux,
                                                        const GpuFemmMeshArtifact &artifact,
                                                        int8_t field_repeat_sign)
{
  if (sector_flux.size() != artifact.circuit_partner_indices.size() ||
      sector_flux.size() != artifact.circuit_partner_orientations.size())
    return {};
  std::vector<double> full = sector_flux;
  for (size_t source = 0; source < sector_flux.size(); ++source) {
    const int32_t target = artifact.circuit_partner_indices[source];
    full[target] +=
        static_cast<double>(artifact.circuit_partner_orientations[source] * field_repeat_sign) *
        sector_flux[source];
  }
  return full;
}

std::vector<double>
ReconstructCyclicSectorCircuitFluxLinkage(const std::vector<double> &sector_flux,
                                          const GpuFemmMeshArtifact &artifact)
{
  if (sector_flux.size() != artifact.circuit_partner_indices.size() ||
      sector_flux.size() != artifact.circuit_partner_orientations.size())
    return {};
  std::vector<double> full(sector_flux.size(), 0.0);
  for (size_t source = 0; source < sector_flux.size(); ++source) {
    int32_t target = static_cast<int32_t>(source);
    int sign = 1;
    for (int32_t copy = 0; copy < artifact.full_machine_sector_count; ++copy) {
      full[target] += static_cast<double>(sign) * sector_flux[source];
      sign *= artifact.circuit_partner_orientations[target] * artifact.sector_boundary_sign;
      target = artifact.circuit_partner_indices[target];
    }
  }
  return full;
}

void ApplyBaseSectorReconstruction(const GpuFemmMeshArtifact &artifact,
                                   const MotorSampleRequest &request, MotorBatchResponseDto *dto)
{
  if (dto == nullptr || !artifact.has_motor_sector)
    return;
  dto->sector_reconstruction_enabled = true;
  dto->sector_start_angle_deg = artifact.sector_start_angle_deg;
  dto->sector_angle_deg = artifact.sector_angle_deg;
  dto->full_machine_sector_count = artifact.full_machine_sector_count;
  dto->electrical_pole_pairs = artifact.electrical_pole_pairs;
  dto->suspension_spatial_order = artifact.suspension_spatial_order;
  dto->base_field_repeat_sign = artifact.sector_boundary_sign;
  dto->tangent_field_repeat_sign =
      static_cast<int8_t>((artifact.suspension_spatial_order % 2) == 0 ? 1 : -1);
  dto->base_torque_multiplier = artifact.has_native_age_sector
                                    ? static_cast<double>(artifact.full_machine_sector_count)
                                    : 2.0;
  const int8_t cross_repeat_sign =
      static_cast<int8_t>(dto->base_field_repeat_sign * dto->tangent_field_repeat_sign);
  dto->tangent_force_multiplier = 1.0 - cross_repeat_sign;
  dto->tangent_torque_multiplier = 1.0 + cross_repeat_sign;
  dto->sector_uses_native_age = artifact.has_sliding_band;
  (void)request;
  dto->circuit_partner_indices = artifact.circuit_partner_indices;
  dto->circuit_partner_orientations = artifact.circuit_partner_orientations;
  dto->apex_self_pair_nodes = artifact.apex_self_pair_nodes;
  dto->model_invariance_certificate_sha256 = artifact.model_invariance_certificate_sha256;
  if (dto->status == Status::kOk) {
    // A complete rotational orbit cancels the vector load while its scalar
    // axial torque adds. v4 is AGE torque-only; no sector force formula is
    // exposed as a physical result.
    dto->force_x_n = 0.0;
    dto->force_y_n = 0.0;
    dto->torque_nm *= dto->base_torque_multiplier;
    dto->circuit_flux_linkage_wb =
        artifact.has_native_age_sector
            ? ReconstructCyclicSectorCircuitFluxLinkage(dto->circuit_flux_linkage_wb, artifact)
            : ReconstructSectorCircuitFluxLinkage(dto->circuit_flux_linkage_wb, artifact,
                                                  dto->base_field_repeat_sign);
    if (dto->circuit_flux_linkage_wb.empty())
      dto->status = Status::kInternalError;
  }
}

bool ApplyTangentSectorReconstruction(const GpuFemmMeshArtifact &artifact,
                                      const MotorBatchResponseDto &response,
                                      TangentRhsResponseDto *rhs)
{
  if (rhs == nullptr || !artifact.has_motor_sector)
    return false;
  rhs->force_x_n_per_a *= response.tangent_force_multiplier;
  rhs->force_y_n_per_a *= response.tangent_force_multiplier;
  rhs->torque_nm_per_a *= response.tangent_torque_multiplier;
  rhs->circuit_flux_linkage_derivative_wb_per_a = ReconstructSectorCircuitFluxLinkage(
      rhs->circuit_flux_linkage_derivative_wb_per_a, artifact, response.tangent_field_repeat_sign);
  return !rhs->circuit_flux_linkage_derivative_wb_per_a.empty();
}

MotorBatchResponseDto MakeMotorBatchResponseDto(Status status, const MotorSampleRequest &request,
                                                const NonlinearSolveResult *solution,
                                                const FrozenPostprocessResult *postprocess)
{
  MotorBatchResponseDto dto;
  dto.status = status;
  dto.tangent_requested = request.tangent_requested;
  if (solution != nullptr) {
    dto.circuit_currents_a = solution->circuit_currents_a;
    dto.iterations = solution->info.iterations;
    dto.residual_l2 = solution->info.residual_l2;
  }
  if (status == Status::kOk && solution != nullptr && postprocess != nullptr) {
    dto.force_x_n = postprocess->force_x_n;
    dto.force_y_n = postprocess->force_y_n;
    dto.torque_nm = postprocess->torque_nm;
    dto.circuit_flux_linkage_wb = solution->circuit_flux_linkage_wb;
    dto.airgap_angles_deg = request.airgap_angles_deg;
    for (const AirgapSample &sample : postprocess->airgap_samples)
      dto.airgap_radial_flux_density_t.push_back(sample.radial_b_t);
    dto.mesh_element_count = solution->bx_t.size();
  }
  return dto;
}

static void WriteMotorBatchItemResponseJson(std::ostream &output, const MotorSampleRequest &request,
                                            const MotorBatchResponseDto &dto)
{
  output << std::setprecision(17) << "{\n  \"protocol\": \"" << request.protocol << "\",\n"
         << "  \"mesh_artifact_sha256\": \"" << request.mesh_artifact_sha256 << "\",\n"
         << "  \"base_motor_fem_sha256\": \"" << request.base_motor_fem_sha256 << "\",\n"
         << "  \"source_fem_sha256\": \"" << request.source_fem_sha256 << "\",\n"
         << "  \"status\": \"" << (dto.status == Status::kOk ? "PASS" : "FAIL") << "\",\n"
         << "  \"solve_status\": \"" << StatusName(dto.status) << "\",\n"
         << "  \"error_identifier\": \""
         << (dto.status == Status::kOk ? "" : std::string("GPU_FEMM_") + StatusName(dto.status))
         << "\",\n"
         << "  \"error_message\": \"" << (dto.status == Status::kOk ? "" : StatusName(dto.status))
         << "\",\n";
  if (dto.status == Status::kOk) {
    output << "  \"Fx_N\": " << dto.force_x_n << ",\n  \"Fy_N\": " << dto.force_y_n
           << ",\n  \"torque_Nm\": " << dto.torque_nm << ",\n  \"actual_circuit_currents_A\": [";
    for (size_t i = 0; i < dto.circuit_currents_a.size(); ++i)
      output << (i ? ", " : "") << dto.circuit_currents_a[i];
    output << "],\n  \"circuit_flux_linkage_Wb\": [";
    for (size_t i = 0; i < dto.circuit_flux_linkage_wb.size(); ++i)
      output << (i ? ", " : "") << dto.circuit_flux_linkage_wb[i];
    output << "],\n  \"airgap_sample_angles_deg\": [";
    for (size_t i = 0; i < dto.airgap_angles_deg.size(); ++i)
      output << (i ? ", " : "") << dto.airgap_angles_deg[i];
    output << "],\n  \"airgap_radial_flux_density_T\": [";
    for (size_t i = 0; i < dto.airgap_radial_flux_density_t.size(); ++i)
      output << (i ? ", " : "") << dto.airgap_radial_flux_density_t[i];
    output << "],\n  \"mesh_element_count\": " << dto.mesh_element_count << ",\n";
    if (dto.sector_reconstruction_enabled) {
      output
          << "  \"sector_reconstruction\": {\"schema_version\": "
          << (dto.sector_uses_native_age ? "\"gpu_femm_motor_sector_reconstruction_v2\", "
                                         : "\"gpu_femm_motor_sector_reconstruction_response_v1\", ")
          << "\"status\": \"" << (dto.sector_reconstruction_status == Status::kOk ? "PASS" : "FAIL")
          << "\", \"error_identifier\": \""
          << (dto.sector_reconstruction_status == Status::kOk
                  ? ""
                  : std::string("GPU_FEMM_SECTOR_") + StatusName(dto.sector_reconstruction_status))
          << "\", \"sector_start_angle_deg\": " << dto.sector_start_angle_deg
          << ", \"sector_angle_deg\": " << dto.sector_angle_deg
          << ", \"full_machine_sector_count\": " << dto.full_machine_sector_count
          << ", \"electrical_pole_pairs\": " << dto.electrical_pole_pairs
          << ", \"suspension_spatial_order\": " << dto.suspension_spatial_order
          << ", \"base_boundary_relation\": \""
          << (dto.base_field_repeat_sign > 0 ? "periodic" : "antiperiodic")
          << "\", \"tangent_boundary_relation\": \""
          << (dto.tangent_field_repeat_sign > 0 ? "periodic" : "antiperiodic")
          << "\", \"base_field_repeat_sign\": " << static_cast<int>(dto.base_field_repeat_sign)
          << ", \"tangent_field_repeat_sign\": " << static_cast<int>(dto.tangent_field_repeat_sign)
          << ", \"output_normalization\": \"full_machine\""
          << ", \"load_extraction_method\": \""
          << (dto.sector_uses_native_age ? "native_air_gap_element_signed_sector"
                                         : "weighted_stress_tensor_periodic_pair_mask")
          << "\", \"periodic_pair_aware_weighted_stress_mask\": "
          << (!dto.sector_uses_native_age ? "true" : "false")
          << ", \"constraint_source\": \"cropped_sector_artifact\""
          << ", \"operator_construction\": "
          << (dto.sector_uses_native_age ? "\"native_age_sector_base_only\""
                                         : "\"same_topology_dual_signed_reduction\"")
          << ", \"tangent_operator_storage\": \"shared_single_upload\""
          << ", \"tangent_physical_operator_count\": "
          << dto.tangent_operator_evidence.physical_operator_count
          << ", \"tangent_operator_upload_count\": "
          << dto.tangent_operator_evidence.operator_upload_count
          << ", \"tangent_rhs_count\": " << dto.tangent_operator_evidence.rhs_count
          << ", \"tangent_operator_bytes\": " << dto.tangent_operator_evidence.operator_bytes
          << ", \"tangent_rhs_workspace_bytes\": "
          << dto.tangent_operator_evidence.rhs_workspace_bytes
          << ", \"base_force_rule\": \"sum_rotated_sector_vectors\""
          << ", \"base_torque_multiplier\": " << dto.base_torque_multiplier
          << ", \"tangent_force_multiplier\": " << dto.tangent_force_multiplier
          << ", \"tangent_torque_multiplier\": " << dto.tangent_torque_multiplier
          << ", \"circuit_partner_indices\": [";
      for (size_t i = 0; i < dto.circuit_partner_indices.size(); ++i)
        output << (i ? ", " : "") << dto.circuit_partner_indices[i];
      output << "], \"circuit_partner_orientations\": [";
      for (size_t i = 0; i < dto.circuit_partner_orientations.size(); ++i)
        output << (i ? ", " : "") << static_cast<int>(dto.circuit_partner_orientations[i]);
      output << "], \"apex_self_pair_nodes\": [";
      for (size_t i = 0; i < dto.apex_self_pair_nodes.size(); ++i)
        output << (i ? ", " : "") << dto.apex_self_pair_nodes[i];
      output << "], \"model_invariance_certificate_sha256\": \""
             << dto.model_invariance_certificate_sha256 << "\"},\n";
    }
    if (dto.tangent_requested) {
      output << "  \"tangent_multi_rhs\": {\"schema_version\": "
             << "\"gpu_femm_tangent_multi_rhs_response_v1\", \"status\": \""
             << (dto.tangent_status == Status::kOk ? "PASS" : "FAIL") << "\", \"solve_status\": \""
             << StatusName(dto.tangent_status) << "\", \"error_identifier\": \""
             << (dto.tangent_status == Status::kOk
                     ? ""
                     : std::string("GPU_FEMM_TANGENT_") + StatusName(dto.tangent_status))
             << "\", \"linearization\": "
             << "\"consistent_newton_jacobian_at_converged_operating_point\", "
             << "\"base_circuit_currents_A\": [";
      for (size_t i = 0; i < dto.circuit_currents_a.size(); ++i)
        output << (i ? ", " : "") << dto.circuit_currents_a[i];
      output << "], \"rhs\": [";
      for (size_t rhs_index = 0; rhs_index < dto.tangent_rhs.size(); ++rhs_index) {
        const TangentRhsResponseDto &rhs = dto.tangent_rhs[rhs_index];
        output << (rhs_index ? ", " : "") << "{\"name\": \"" << rhs.name
               << "\", \"circuit_current_derivative_A_per_A\": [";
        for (size_t i = 0; i < rhs.circuit_current_derivative_a_per_a.size(); ++i)
          output << (i ? ", " : "") << rhs.circuit_current_derivative_a_per_a[i];
        output << "], \"dFx_N_per_A\": " << rhs.force_x_n_per_a
               << ", \"dFy_N_per_A\": " << rhs.force_y_n_per_a
               << ", \"dtorque_Nm_per_A\": " << rhs.torque_nm_per_a
               << ", \"circuit_flux_linkage_derivative_Wb_per_A\": [";
        for (size_t i = 0; i < rhs.circuit_flux_linkage_derivative_wb_per_a.size(); ++i)
          output << (i ? ", " : "") << rhs.circuit_flux_linkage_derivative_wb_per_a[i];
        output << "], \"airgap_radial_flux_density_derivative_T_per_A\": [";
        for (size_t i = 0; i < rhs.airgap_radial_flux_density_derivative_t_per_a.size(); ++i)
          output << (i ? ", " : "") << rhs.airgap_radial_flux_density_derivative_t_per_a[i];
        output << "], \"linear_convergence\": {\"iterations\": "
               << rhs.linear_convergence.iterations << ", \"residual_l2\": ";
        if (std::isfinite(rhs.linear_convergence.residual_l2))
          output << rhs.linear_convergence.residual_l2;
        else
          output << "null";
        output << "}}";
      }
      output << "]},\n";
    }
    output << "  \"convergence\": {\"iterations\": " << dto.iterations
           << ", \"residual_l2\": " << dto.residual_l2 << "}\n";
  } else {
    output << "  \"Fx_N\": null,\n  \"Fy_N\": null,\n  \"torque_Nm\": null,\n"
           << "  \"actual_circuit_currents_A\": [],\n  \"circuit_flux_linkage_Wb\": [],\n"
           << "  \"airgap_sample_angles_deg\": [],\n  \"airgap_radial_flux_density_T\": [],\n"
           << "  \"mesh_element_count\": 0,\n  \"convergence\": {\"iterations\": " << dto.iterations
           << ", \"residual_l2\": ";
    if (std::isfinite(dto.residual_l2))
      output << dto.residual_l2;
    else
      output << "null";
    output << "}\n";
  }
  output << "}";
}

bool WriteMotorBatchResponse(const std::string &path, const MotorBatchRequest *request,
                             const std::vector<MotorBatchResponseDto> &responses,
                             int32_t requested_chunk, int32_t effective_chunk, int32_t cache_hits,
                             int32_t chunk_count, size_t csr_symbolic_cache_hits,
                             int actual_parallel_width, int batched_pcg_launches,
                             const MotorBatchTiming *timing)
{
  std::ofstream output(path, std::ios::trunc);
  if (!output)
    return false;
  bool pass = request != nullptr && request->items.size() == responses.size();
  for (const MotorBatchResponseDto &response : responses)
    pass = pass && response.status == Status::kOk;
  output << std::setprecision(17) << "{\n  \"protocol\": \""
         << (request == nullptr ? "gpu_femm_motor_batch_v1" : request->protocol) << "\",\n"
         << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
         << "  \"solve_status\": \"" << (pass ? "PASS" : "PARTIAL_FAILURE") << "\",\n"
         << "  \"error_identifier\": \"" << (pass ? "" : "GPU_FEMM_BATCH_PARTIAL_FAILURE")
         << "\",\n"
         << "  \"error_message\": \"" << (pass ? "" : "one or more batch items failed") << "\",\n"
         << "  \"cache\": {\"mesh_cache_hits\": " << cache_hits
         << ", \"chunk_count\": " << chunk_count
         << ", \"requested_chunk_size\": " << requested_chunk
         << ", \"effective_chunk_size\": " << effective_chunk
         << ", \"actual_parallel_width\": " << actual_parallel_width
         << ", \"csr_symbolic_cache_hits\": " << csr_symbolic_cache_hits
         << ", \"batched_pcg_launches\": " << batched_pcg_launches << "},\n";
  if (timing != nullptr) {
    output << "  \"timing\": {\"schema_version\": \"gpu_femm_motor_batch_timing_v1\""
           << ", \"request_read_parse_seconds\": " << timing->request_read_parse_seconds
           << ", \"preflight_seconds\": " << timing->preflight_seconds
           << ", \"artifact_read_validate_seconds\": " << timing->artifact_read_validate_seconds
           << ", \"solver_initialize_seconds\": " << timing->solver_initialize_seconds
           << ", \"host_assembly_seconds\": " << timing->nonlinear.host_assembly_seconds
           << ", \"device_assembly_seconds\": " << timing->nonlinear.device_assembly_seconds
           << ", \"gpu_upload_seconds\": " << timing->nonlinear.gpu_upload_seconds
           << ", \"gpu_kernel_sync_seconds\": " << timing->nonlinear.gpu_kernel_sync_seconds
           << ", \"gpu_download_seconds\": " << timing->nonlinear.gpu_download_seconds
           << ", \"linear_solver\": \""
           << (selected_linear_solver == LinearSolverBackend::kCudss ? "cudss" : "pcg") << "\""
           << ", \"direct_analyses\": " << direct_solver_stats.analyses
           << ", \"direct_factorizations\": " << direct_solver_stats.factorizations
           << ", \"direct_solves\": " << direct_solver_stats.solves
           << ", \"direct_factor_reuses\": " << direct_solver_stats.factor_reuses
           << ", \"direct_analysis_seconds\": " << direct_solver_stats.analysis_seconds
           << ", \"direct_factor_seconds\": " << direct_solver_stats.factor_seconds
           << ", \"direct_solve_seconds\": " << direct_solver_stats.solve_seconds
           << ", \"direct_peak_plan_device_bytes\": "
           << direct_solver_stats.peak_device_estimate_bytes << ", \"preconditioner\": \""
           << (selected_linear_solver == LinearSolverBackend::kCudss   ? "none"
               : selected_preconditioner == PcgPreconditioner::kJacobi ? "jacobi"
                                                                       : "block-jacobi")
           << "\""
           << ", \"preconditioner_setup_seconds\": "
           << timing->nonlinear.preconditioner.setup_seconds
           << ", \"interleaved_spmv_solves\": " << timing->nonlinear.interleaved_spmv_solves
           << ", \"preconditioner_builds\": " << timing->nonlinear.preconditioner.builds
           << ", \"preconditioner_block_size\": "
           << (selected_preconditioner == PcgPreconditioner::kJacobi ? 1 : kJacobiBlockSize)
           << ", \"nonlinear_pcg_iterations\": " << timing->nonlinear_pcg_iterations
           << ", \"preconditioner_reuses\": " << timing->nonlinear.preconditioner.reuses
           << ", \"preconditioner_fallback_blocks\": "
           << timing->nonlinear.preconditioner.fallback_blocks
           << ", \"state_update_seconds\": " << timing->nonlinear.state_update_seconds
           << ", \"finalize_seconds\": " << timing->nonlinear.finalize_seconds
           << ", \"nonlinear_recovery_seconds\": " << timing->nonlinear.recovery_seconds
           << ", \"postprocess_plan_compute_seconds\": " << timing->postprocess_plan_compute_seconds
           << ", \"postprocess_plan_seconds\": " << timing->postprocess_plan_seconds
           << ", \"base_postprocess_seconds\": " << timing->base_postprocess_seconds
           << ", \"tangent_operator_seconds\": " << timing->tangent_operator_seconds
           << ", \"tangent_postprocess_seconds\": " << timing->tangent_postprocess_seconds
           << ", \"postprocess_seconds\": " << timing->postprocess_seconds
           << ", \"measured_before_response_write_seconds\": "
           << timing->measured_before_response_write_seconds << "},\n";
  }
  output << "  \"items\": [\n";
  if (request != nullptr)
    for (size_t index = 0; index < request->items.size(); ++index) {
      output << "    {\"task_id\": \"" << request->items[index].task_id << "\", \"response\": ";
      WriteMotorBatchItemResponseJson(output, request->items[index].request, responses[index]);
      output << "}" << (index + 1 == request->items.size() ? "\n" : ",\n");
    }
  output << "  ]\n}\n";
  return static_cast<bool>(output);
}

} // namespace gpu_femm

#pragma once

#include "core/types.h"
#include "io/mesh_artifact.h"
#include "io/sample_request.h"
#include "postprocess/postprocess.h"
#include "solvers/nonlinear_solver.cuh"
#include "solvers/options.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace gpu_femm
{

struct MotorBatchItem {
  std::string task_id;
  MotorSampleRequest request;
};

struct MotorBatchRequest {
  std::string protocol = "gpu_femm_motor_batch_v1";
  int32_t max_items_per_chunk = 1;
  std::vector<MotorBatchItem> items;
};

bool ReadMotorBatchRequestJson(const std::string &json, MotorBatchRequest *request,
                               std::string *error);

// Keep artifact parsing and SHA verification tied to a resolved filesystem
// identity, rather than to the spelling used by an individual request.  The
// original request path is retained as the read path so failed opens preserve
// the established per-request INPUT_IO behavior.
struct MotorBatchArtifactLoadPlan {
  std::vector<std::string> canonical_paths;
  std::vector<std::string> representative_paths;
  std::vector<size_t> item_slots;
  std::vector<size_t> slot_use_counts;
};

MotorBatchArtifactLoadPlan MakeMotorBatchArtifactLoadPlan(const MotorBatchRequest &request);

struct MotorBatchArtifactLoad {
  Status status = Status::kInternalError;
  std::string artifact_sha256;
  std::string fingerprint;
  GpuFemmMeshArtifact artifact;
};

// CPU-only input snapshot owned by the bounded streaming producer. Each
// submission reads and hashes its files afresh; no timestamp-based cache.
struct MotorBatchPrefetch {
  MotorBatchRequest request;
  MotorBatchArtifactLoadPlan load_plan;
  std::vector<std::unique_ptr<MotorBatchArtifactLoad>> loads;
  struct MaskPreparation {
    std::string operator_key;
    int32_t selected_group_number = -1, air_group_number = -1;
    Status status = Status::kInternalError;
    std::shared_ptr<const PreparedWeightedStressMask> mask;
    double compute_seconds = 0;
  };
  std::vector<MaskPreparation> masks;
  double prepare_seconds = 0.0;
};

// Strictly CPU-only, bounded to the first four distinct non-AGE operators.
void PrefetchMotorBatchMasks(MotorBatchPrefetch *input);

void LoadMotorBatchArtifact(const std::string &path, MotorBatchArtifactLoad *load);

// SolveBatch can share its CUDA CSR operator only when every item has the
// exact same assembled model.  For v2 this deliberately keys the immutable
// artifact bytes and the requested mechanical pose, rather than treating all
// sliding-band samples as one geometry.  Different current vectors and
// postprocess requests do not change that operator and may therefore batch.
std::string MotorBatchOperatorKey(const MotorSampleRequest &request,
                                  const MotorBatchArtifactLoad &load, const NonlinearModel &model);

struct MotorBatchPreparedItem {
  size_t request_index = 0;
  const GpuFemmMeshArtifact *artifact = nullptr;
  NonlinearModel model;
  NonlinearModel tangent_model;
  std::vector<double> currents;
  std::string operator_key;
};

// Return stable first-seen groups so the execution order remains predictable;
// response placement always uses request_index and therefore stays in caller
// order even when compatible samples were interleaved in the input.
std::vector<std::vector<size_t>>
GroupMotorBatchPreparedItemsByOperator(const std::vector<MotorBatchPreparedItem> &prepared);

// Keep only protocol output after each item/chunk.  In particular, do not
// retain the nodal A or per-element B vectors for a whole batch.
struct TangentRhsResponseDto {
  std::string name;
  std::vector<double> circuit_current_derivative_a_per_a;
  double force_x_n_per_a = 0.0;
  double force_y_n_per_a = 0.0;
  double torque_nm_per_a = 0.0;
  std::vector<double> circuit_flux_linkage_derivative_wb_per_a;
  std::vector<double> airgap_radial_flux_density_derivative_t_per_a;
  SolveInfo linear_convergence;
};

struct MotorBatchResponseDto {
  Status status = Status::kInvalidArgument;
  double force_x_n = 0.0;
  double force_y_n = 0.0;
  double torque_nm = 0.0;
  std::vector<double> circuit_currents_a;
  std::vector<double> circuit_flux_linkage_wb;
  std::vector<double> airgap_angles_deg;
  std::vector<double> airgap_radial_flux_density_t;
  size_t mesh_element_count = 0;
  int iterations = 0;
  double residual_l2 = std::numeric_limits<double>::infinity();
  bool tangent_requested = false;
  Status tangent_status = Status::kInvalidArgument;
  std::vector<TangentRhsResponseDto> tangent_rhs;
  bool sector_reconstruction_enabled = false;
  Status sector_reconstruction_status = Status::kInvalidArgument;
  double sector_start_angle_deg = 0.0;
  double sector_angle_deg = 360.0;
  int32_t full_machine_sector_count = 1;
  int32_t electrical_pole_pairs = 0;
  int32_t suspension_spatial_order = 0;
  int8_t base_field_repeat_sign = 1;
  int8_t tangent_field_repeat_sign = 1;
  double base_torque_multiplier = 1.0;
  double tangent_force_multiplier = 1.0;
  double tangent_torque_multiplier = 1.0;
  bool sector_uses_native_age = false;
  SharedOperatorBatchEvidence tangent_operator_evidence;
  std::vector<int32_t> circuit_partner_indices;
  std::vector<int8_t> circuit_partner_orientations;
  std::vector<int32_t> apex_self_pair_nodes;
  std::string model_invariance_certificate_sha256;
};

std::vector<double> ReconstructSectorCircuitFluxLinkage(const std::vector<double> &sector_flux,
                                                        const GpuFemmMeshArtifact &artifact,
                                                        int8_t field_repeat_sign);

std::vector<double>
ReconstructCyclicSectorCircuitFluxLinkage(const std::vector<double> &sector_flux,
                                          const GpuFemmMeshArtifact &artifact);

void ApplyBaseSectorReconstruction(const GpuFemmMeshArtifact &artifact,
                                   const MotorSampleRequest &request, MotorBatchResponseDto *dto);

bool ApplyTangentSectorReconstruction(const GpuFemmMeshArtifact &artifact,
                                      const MotorBatchResponseDto &response,
                                      TangentRhsResponseDto *rhs);

struct MotorBatchTiming {
  size_t nonlinear_pcg_iterations = 0;
  double request_read_parse_seconds = 0.0;
  double preflight_seconds = 0.0;
  double artifact_read_validate_seconds = 0.0;
  double solver_initialize_seconds = 0.0;
  double nonlinear_prepare_compute_seconds = 0.0;
  double nonlinear_prepare_wait_seconds = 0.0;
  NonlinearBatchTiming nonlinear;
  double postprocess_plan_compute_seconds = 0.0;
  double postprocess_plan_seconds = 0.0;
  double base_postprocess_seconds = 0.0;
  double tangent_operator_seconds = 0.0;
  double tangent_postprocess_seconds = 0.0;
  double postprocess_seconds = 0.0;
  double measured_before_response_write_seconds = 0.0;
};

struct MotorBatchPostprocessResult {
  std::vector<MotorBatchResponseDto> responses;
  std::vector<size_t> indices;
  MotorBatchTiming timing;
};

MotorBatchPostprocessResult PostprocessMotorBatchGroup(
    const MotorBatchRequest &request, const std::vector<MotorBatchPreparedItem> &prepared,
    const std::vector<size_t> &group, std::vector<NonlinearSolveResult> chunk_solutions,
    FrozenPostprocessPlan prepared_plan, NonlinearP1Solver &solver, const NonlinearOptions &options,
    int *launches);

MotorBatchResponseDto MakeMotorBatchResponseDto(Status status, const MotorSampleRequest &request,
                                                const NonlinearSolveResult *solution,
                                                const FrozenPostprocessResult *postprocess);

bool WriteMotorBatchResponse(const std::string &path, const MotorBatchRequest *request,
                             const std::vector<MotorBatchResponseDto> &responses,
                             int32_t requested_chunk, int32_t effective_chunk, int32_t cache_hits,
                             int32_t chunk_count, size_t csr_symbolic_cache_hits,
                             int actual_parallel_width, int batched_pcg_launches,
                             const MotorBatchTiming *timing);

int MotorBatchAdapter(const std::string &request_path, const std::string &response_path,
                      bool include_timing = false,
                      NonlinearAssemblyMode assembly_mode = NonlinearAssemblyMode::kAuto,
                      std::unique_ptr<MotorBatchPrefetch> prefetch = nullptr);

} // namespace gpu_femm

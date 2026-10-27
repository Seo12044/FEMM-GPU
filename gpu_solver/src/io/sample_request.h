#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace gpu_femm
{

struct SingleSampleRequest {
  std::string stem;
  std::string curve_directory;
  std::string source_motor_fem_sha256;
  double current_a = 0.0;
  int32_t selected_material_label = -1;
  double airgap_radius_mm = 0.0;
  std::vector<double> airgap_angles_deg;
};

struct TangentMultiRhsRequest {
  std::vector<std::pair<std::string, std::vector<double>>> rhs;
};

struct MotorSectorPerformanceRequest {
  bool enabled = false;
  std::string model_invariance_certificate_sha256;
};

struct MotorSampleRequest {
  std::string protocol = "gpu_femm_motor_sample_v1";
  std::string mesh_artifact_path;
  std::string mesh_artifact_sha256;
  std::string base_motor_fem_sha256;
  std::string source_fem_sha256;
  std::vector<double> circuit_currents_a;
  int32_t selected_group_number = -1;
  int32_t air_group_number = -1;
  double airgap_radius_mm = 0.0;
  std::vector<double> airgap_angles_deg;
  double rotor_angle_deg = 0.0;
  double displacement_mm[2] = {};
  bool tangent_requested = false;
  TangentMultiRhsRequest tangent_multi_rhs;
  MotorSectorPerformanceRequest sector_performance;
};

// Project-neutral planar-DC entry point.  The production motor protocol above
// remains frozen for the MATLAB adapter; this request exposes the same solver
// without requiring motor identity, pose, or force-integration fields.
struct PlanarDcSampleRequest {
  std::string protocol = "gpu_femm_planar_dc_sample_v1";
  std::string mesh_artifact_path;
  std::string mesh_artifact_sha256;
  std::vector<double> circuit_currents_a;
  int32_t force_group_number = -1;
  int32_t stress_air_group_number = -1;
  double airgap_radius_mm = 0.0;
  std::vector<double> airgap_angles_deg;
  double sliding_band_angle_deg = 0.0;
  bool compute_force_torque = false;
  bool include_field_solution = false;
};

struct StrictJson;
struct GpuFemmMeshArtifact;
struct NonlinearModel;
struct FrozenPostprocessOptions;
struct FrozenPostprocessResult;
struct TangentPostprocessResult;

bool ReadSingleSampleRequest(const std::string &path, SingleSampleRequest *request,
                             std::string *error);
bool ReadPlanarDcSampleRequestJson(const std::string &json, PlanarDcSampleRequest *request,
                                   std::string *error);
bool ReadMotorSampleRequestValue(const StrictJson &root, MotorSampleRequest *request,
                                 std::string *error);
bool ReadMotorSampleRequestJson(const std::string &json, MotorSampleRequest *request,
                                std::string *error);
bool RequestMatchesArtifact(const MotorSampleRequest &request, const GpuFemmMeshArtifact &artifact);
bool ApplySlidingBandAngle(double requested_angle_deg, const GpuFemmMeshArtifact &artifact,
                           NonlinearModel *model);
bool ApplySlidingBandRotorAngle(const MotorSampleRequest &request,
                                const GpuFemmMeshArtifact &artifact, NonlinearModel *model);
bool MakeMotorSectorTangentModel(const GpuFemmMeshArtifact &artifact,
                                 const NonlinearModel &posed_base_model,
                                 NonlinearModel *tangent_model);
std::vector<int8_t> MapSectorAirgapAnglesToRepresentative(double sector_start_angle_rad,
                                                          int8_t repeat_sign,
                                                          FrozenPostprocessOptions *options);
std::vector<int8_t>
MapNativeAgeSectorAirgapAnglesToRepresentative(const GpuFemmMeshArtifact &artifact,
                                               FrozenPostprocessOptions *options);
bool ApplySectorAirgapRepeatSigns(const std::vector<int8_t> &signs,
                                  FrozenPostprocessResult *postprocess);
bool ApplySectorTangentAirgapRepeatSigns(const std::vector<int8_t> &signs,
                                         TangentPostprocessResult *postprocess);

} // namespace gpu_femm

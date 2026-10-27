#pragma once

#include "physics/nonlinear_model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpu_femm
{

struct GpuFemmMeshArtifact {
  NonlinearModel model;
  std::vector<double> circuit_currents_a;
  std::vector<int32_t> triangle_group_numbers;
  std::string schema_version;
  std::string pose_fem_sha256;
  std::string base_motor_fem_sha256;
  std::string canonical_identity_sha256;
  bool has_sliding_band = false;
  bool has_motor_sector = false;
  // v4 is a cyclic, native-AGE torque-only sector. Keep it separate from
  // v3's conforming-mesh/WST sector so old protocol semantics cannot drift.
  bool has_native_age_sector = false;
  double sector_start_angle_deg = 0.0;
  double sector_angle_deg = 360.0;
  int32_t full_machine_sector_count = 1;
  int32_t electrical_pole_pairs = 0;
  int32_t suspension_spatial_order = 0;
  int8_t sector_boundary_sign = 1;
  std::vector<int32_t> circuit_partner_indices;
  std::vector<int8_t> circuit_partner_orientations;
  std::vector<int32_t> apex_self_pair_nodes;
  std::string model_invariance_certificate_sha256;
  double rotor_angle_deg = 0.0;
  double displacement_mm[2] = {};
};

bool ParseGpuFemmMeshArtifactJson(const std::string &text, GpuFemmMeshArtifact *artifact,
                                  std::string *error);
bool ReadGpuFemmMeshArtifact(const std::string &path, GpuFemmMeshArtifact *artifact,
                             std::string *error);

} // namespace gpu_femm

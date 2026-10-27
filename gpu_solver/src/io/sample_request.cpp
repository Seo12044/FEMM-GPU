#include "io/sample_request.h"
#include "io/mesh_artifact.h"
#include "io/strict_json.h"
#include "postprocess/postprocess.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace gpu_femm
{

namespace
{

bool JsonValueStart(const std::string &json, const std::string &key, size_t *position)
{
  const std::string quoted = "\"" + key + "\"";
  const size_t key_position = json.find(quoted);
  if (key_position == std::string::npos)
    return false;
  const size_t colon = json.find(':', key_position + quoted.size());
  if (colon == std::string::npos)
    return false;
  *position = colon + 1;
  while (*position < json.size() && std::isspace(static_cast<unsigned char>(json[*position])))
    ++*position;
  return *position < json.size();
}

bool JsonString(const std::string &json, const std::string &key, std::string *value)
{
  size_t position = 0;
  if (!JsonValueStart(json, key, &position) || json[position] != '\"')
    return false;
  const size_t end = json.find('\"', position + 1);
  if (end == std::string::npos)
    return false;
  *value = json.substr(position + 1, end - position - 1);
  return value->find('\\') == std::string::npos;
}

bool JsonDouble(const std::string &json, const std::string &key, double *value)
{
  size_t position = 0;
  if (!JsonValueStart(json, key, &position))
    return false;
  try {
    size_t parsed = 0;
    *value = std::stod(json.substr(position), &parsed);
    return parsed > 0 && std::isfinite(*value);
  } catch (...) {
    return false;
  }
}

bool JsonDoubleArray(const std::string &json, const std::string &key, std::vector<double> *values)
{
  size_t position = 0;
  if (!JsonValueStart(json, key, &position) || json[position] != '[')
    return false;
  ++position;
  values->clear();
  while (position < json.size()) {
    while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position])))
      ++position;
    if (position >= json.size())
      return false;
    if (json[position] == ']')
      return true;
    try {
      size_t parsed = 0;
      const double value = std::stod(json.substr(position), &parsed);
      if (parsed == 0 || !std::isfinite(value))
        return false;
      values->push_back(value);
      position += parsed;
    } catch (...) {
      return false;
    }
    while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position])))
      ++position;
    if (position >= json.size() || (json[position] != ',' && json[position] != ']'))
      return false;
    if (json[position] == ']')
      return true;
    ++position;
  }
  return false;
}

} // namespace

bool ReadSingleSampleRequest(const std::string &path, SingleSampleRequest *request,
                             std::string *error)
{
  std::ifstream input(path);
  std::stringstream contents;
  contents << input.rdbuf();
  const std::string json = contents.str();
  std::string protocol;
  double selected = 0.0;
  if (!input || !JsonString(json, "protocol", &protocol) ||
      protocol != "gpu_femm_single_sample_v1" || !JsonString(json, "stem", &request->stem) ||
      !JsonString(json, "curve_directory", &request->curve_directory) ||
      !JsonString(json, "source_motor_fem_sha256", &request->source_motor_fem_sha256) ||
      !JsonDouble(json, "current_A", &request->current_a) ||
      !JsonDouble(json, "selected_material_label", &selected) ||
      !JsonDouble(json, "airgap_radius_mm", &request->airgap_radius_mm) ||
      !JsonDoubleArray(json, "airgap_angles_deg", &request->airgap_angles_deg) ||
      request->stem.empty() || request->curve_directory.empty() ||
      request->source_motor_fem_sha256.size() != 64 || !(request->airgap_radius_mm >= 0.0) ||
      (!request->airgap_angles_deg.empty() && !(request->airgap_radius_mm > 0.0))) {
    *error = "invalid gpu_femm_single_sample_v1 request";
    return false;
  }
  request->selected_material_label = static_cast<int32_t>(selected);
  if (selected != static_cast<double>(request->selected_material_label) ||
      request->selected_material_label != 1) {
    *error = "frozen adapter requires selected_material_label=1";
    return false;
  }
  if (!std::all_of(request->source_motor_fem_sha256.begin(), request->source_motor_fem_sha256.end(),
                   [](unsigned char c) { return std::isxdigit(c); })) {
    *error = "source_motor_fem_sha256 must be hexadecimal";
    return false;
  }
  return true;
}

namespace
{

bool StrictNumberArray(const StrictJson &value, std::vector<double> *output)
{
  if (value.type != StrictJson::Type::kArray || output == nullptr)
    return false;
  output->clear();
  for (const StrictJson &member : value.array) {
    if (member.type != StrictJson::Type::kNumber)
      return false;
    output->push_back(member.number);
  }
  return true;
}

} // namespace

bool ReadPlanarDcSampleRequestJson(const std::string &json, PlanarDcSampleRequest *request,
                                   std::string *error)
{
  StrictJson root;
  StrictJsonParser parser(json);
  if (request == nullptr || error == nullptr || !parser.Parse(&root, error) ||
      !ExactObject(root,
                   {"protocol", "mesh_artifact_path", "mesh_artifact_sha256", "circuit_currents_A",
                    "force_group_number", "stress_air_group_number", "airgap_radius_mm",
                    "airgap_angles_deg", "sliding_band_angle_deg", "compute_force_torque",
                    "include_field_solution"},
                   error))
    return false;
  const StrictJson *protocol = JsonMember(root, "protocol", StrictJson::Type::kString, error);
  const StrictJson *path = JsonMember(root, "mesh_artifact_path", StrictJson::Type::kString, error);
  const StrictJson *artifact_sha =
      JsonMember(root, "mesh_artifact_sha256", StrictJson::Type::kString, error);
  const StrictJson *currents =
      JsonMember(root, "circuit_currents_A", StrictJson::Type::kArray, error);
  const StrictJson *force_group =
      JsonMember(root, "force_group_number", StrictJson::Type::kNumber, error);
  const StrictJson *air_group =
      JsonMember(root, "stress_air_group_number", StrictJson::Type::kNumber, error);
  const StrictJson *radius = JsonMember(root, "airgap_radius_mm", StrictJson::Type::kNumber, error);
  const StrictJson *angles = JsonMember(root, "airgap_angles_deg", StrictJson::Type::kArray, error);
  const StrictJson *sliding_angle =
      JsonMember(root, "sliding_band_angle_deg", StrictJson::Type::kNumber, error);
  const StrictJson *compute_force =
      JsonMember(root, "compute_force_torque", StrictJson::Type::kBool, error);
  const StrictJson *include_field =
      JsonMember(root, "include_field_solution", StrictJson::Type::kBool, error);
  int32_t selected = -2, air = -2;
  if (protocol == nullptr || path == nullptr || artifact_sha == nullptr || currents == nullptr ||
      force_group == nullptr || air_group == nullptr || radius == nullptr || angles == nullptr ||
      sliding_angle == nullptr || compute_force == nullptr || include_field == nullptr ||
      (protocol->string != "gpu_femm_planar_dc_sample_v1" &&
       protocol->string != "gpu_femm_magnetostatic_sample_v1") ||
      path->string.empty() || !JsonLowerSha256(artifact_sha->string) ||
      !JsonInteger(*force_group, &selected) || !JsonInteger(*air_group, &air) || selected < -1 ||
      air < -1 || ((selected == -1) != (air == -1)) || (selected >= 0 && selected == air) ||
      !(radius->number >= 0.0) || !StrictNumberArray(*currents, &request->circuit_currents_a) ||
      !StrictNumberArray(*angles, &request->airgap_angles_deg) ||
      (!request->airgap_angles_deg.empty() && !(radius->number > 0.0))) {
    if (error->empty())
      *error = "invalid gpu_femm_planar_dc_sample_v1 request";
    return false;
  }
  request->mesh_artifact_path = path->string;
  request->protocol = protocol->string;
  request->mesh_artifact_sha256 = artifact_sha->string;
  request->force_group_number = selected;
  request->stress_air_group_number = air;
  request->airgap_radius_mm = radius->number;
  request->sliding_band_angle_deg = sliding_angle->number;
  request->compute_force_torque = compute_force->boolean;
  request->include_field_solution = include_field->boolean;
  return true;
}

bool ReadMotorSampleRequestValue(const StrictJson &root, MotorSampleRequest *request,
                                 std::string *error)
{
  const std::initializer_list<const char *> legacy_fields = {
      "protocol",          "mesh_artifact_path", "mesh_artifact_sha256",  "base_motor_fem_sha256",
      "source_fem_sha256", "circuit_currents_A", "selected_group_number", "air_group_number",
      "airgap_radius_mm",  "airgap_angles_deg",  "rotor_angle_deg",       "displacement_mm"};
  const auto protocol_field =
      root.type == StrictJson::Type::kObject ? root.object.find("protocol") : root.object.end();
  const bool v1 = protocol_field != root.object.end() &&
                  protocol_field->second.type == StrictJson::Type::kString &&
                  protocol_field->second.string == "gpu_femm_motor_sample_v1";
  const bool v2 = protocol_field != root.object.end() &&
                  protocol_field->second.type == StrictJson::Type::kString &&
                  protocol_field->second.string == "gpu_femm_motor_sample_v2";
  const bool v3 = protocol_field != root.object.end() &&
                  protocol_field->second.type == StrictJson::Type::kString &&
                  protocol_field->second.string == "gpu_femm_motor_sample_v3";
  const bool v4 = protocol_field != root.object.end() &&
                  protocol_field->second.type == StrictJson::Type::kString &&
                  protocol_field->second.string == "gpu_femm_motor_sample_v4";
  const bool has_tangent_field = root.type == StrictJson::Type::kObject &&
                                 root.object.find("tangent_multi_rhs") != root.object.end();
  if (request == nullptr || error == nullptr || (!v1 && !v2 && !v3 && !v4) ||
      (v1 && !ExactObject(root, legacy_fields, error)) || (v2 && !has_tangent_field) ||
      (v2 && !ExactObject(root,
                          {"protocol", "mesh_artifact_path", "mesh_artifact_sha256",
                           "base_motor_fem_sha256", "source_fem_sha256", "circuit_currents_A",
                           "selected_group_number", "air_group_number", "airgap_radius_mm",
                           "airgap_angles_deg", "rotor_angle_deg", "displacement_mm",
                           "tangent_multi_rhs"},
                          error)) ||
      ((v3 || v4) &&
       !ExactObject(root,
                    {"protocol", "mesh_artifact_path", "mesh_artifact_sha256",
                     "base_motor_fem_sha256", "source_fem_sha256", "circuit_currents_A",
                     "selected_group_number", "air_group_number", "airgap_radius_mm",
                     "airgap_angles_deg", "rotor_angle_deg", "displacement_mm", "tangent_multi_rhs",
                     "sector_performance"},
                    error)))
    return false;
  const StrictJson *protocol = JsonMember(root, "protocol", StrictJson::Type::kString, error);
  const StrictJson *path = JsonMember(root, "mesh_artifact_path", StrictJson::Type::kString, error);
  const StrictJson *artifact_sha =
      JsonMember(root, "mesh_artifact_sha256", StrictJson::Type::kString, error);
  const StrictJson *base_sha =
      JsonMember(root, "base_motor_fem_sha256", StrictJson::Type::kString, error);
  const StrictJson *source_sha =
      JsonMember(root, "source_fem_sha256", StrictJson::Type::kString, error);
  const StrictJson *currents =
      JsonMember(root, "circuit_currents_A", StrictJson::Type::kArray, error);
  const StrictJson *selected =
      JsonMember(root, "selected_group_number", StrictJson::Type::kNumber, error);
  const StrictJson *air = JsonMember(root, "air_group_number", StrictJson::Type::kNumber, error);
  const StrictJson *radius = JsonMember(root, "airgap_radius_mm", StrictJson::Type::kNumber, error);
  const StrictJson *angles = JsonMember(root, "airgap_angles_deg", StrictJson::Type::kArray, error);
  const StrictJson *rotor = JsonMember(root, "rotor_angle_deg", StrictJson::Type::kNumber, error);
  const StrictJson *displacement =
      JsonMember(root, "displacement_mm", StrictJson::Type::kArray, error);
  int32_t selected_group = -1, air_group = -1;
  if (protocol == nullptr || path == nullptr || artifact_sha == nullptr || base_sha == nullptr ||
      source_sha == nullptr || currents == nullptr || selected == nullptr || air == nullptr ||
      radius == nullptr || angles == nullptr || rotor == nullptr || displacement == nullptr ||
      (protocol->string != "gpu_femm_motor_sample_v1" &&
       protocol->string != "gpu_femm_motor_sample_v2" &&
       protocol->string != "gpu_femm_motor_sample_v3" &&
       protocol->string != "gpu_femm_motor_sample_v4") ||
      path->string.empty() || !JsonSha256(artifact_sha->string) || !JsonSha256(base_sha->string) ||
      !JsonSha256(source_sha->string) || !JsonInteger(*selected, &selected_group) ||
      !JsonInteger(*air, &air_group) || selected_group < 0 || air_group < 0 ||
      selected_group == air_group || !(radius->number >= 0.0) || displacement->array.size() != 2 ||
      displacement->array[0].type != StrictJson::Type::kNumber ||
      displacement->array[1].type != StrictJson::Type::kNumber ||
      !StrictNumberArray(*currents, &request->circuit_currents_a) ||
      !StrictNumberArray(*angles, &request->airgap_angles_deg) ||
      request->circuit_currents_a.empty() ||
      (!request->airgap_angles_deg.empty() && !(radius->number > 0.0))) {
    if (error->empty())
      *error = "invalid gpu_femm_motor_sample_v1 request";
    return false;
  }
  request->mesh_artifact_path = path->string;
  request->protocol = protocol->string;
  request->mesh_artifact_sha256 = artifact_sha->string;
  request->base_motor_fem_sha256 = base_sha->string;
  request->source_fem_sha256 = source_sha->string;
  request->selected_group_number = selected_group;
  request->air_group_number = air_group;
  request->airgap_radius_mm = radius->number;
  request->rotor_angle_deg = rotor->number;
  request->displacement_mm[0] = displacement->array[0].number;
  request->displacement_mm[1] = displacement->array[1].number;
  request->tangent_requested = false;
  request->tangent_multi_rhs.rhs.clear();
  request->sector_performance = {};
  if (v2 || v3 || v4) {
    const StrictJson &tangent = root.object.at("tangent_multi_rhs");
    // MATLAB homogeneous struct arrays commonly encode an unused nested
    // struct as [] or null.  Both are explicit no-op values.
    const bool empty_tangent = tangent.type == StrictJson::Type::kNull ||
                               (tangent.type == StrictJson::Type::kArray && tangent.array.empty());
    if (empty_tangent && v2)
      return true;
    if (!empty_tangent) {
      if (!ExactObject(tangent, {"schema_version", "rhs"}, error))
        return false;
      const StrictJson *schema =
          JsonMember(tangent, "schema_version", StrictJson::Type::kString, error);
      const StrictJson *rhs = JsonMember(tangent, "rhs", StrictJson::Type::kArray, error);
      if (schema == nullptr || rhs == nullptr ||
          schema->string != "gpu_femm_tangent_multi_rhs_v1" || rhs->array.empty() ||
          rhs->array.size() > 8) {
        if (error->empty())
          *error = "invalid gpu_femm_tangent_multi_rhs_v1 request";
        return false;
      }
      std::map<std::string, bool> names;
      for (const StrictJson &entry : rhs->array) {
        if (!ExactObject(entry, {"name", "circuit_current_derivative_A_per_A"}, error))
          return false;
        const StrictJson *name = JsonMember(entry, "name", StrictJson::Type::kString, error);
        const StrictJson *direction = JsonMember(entry, "circuit_current_derivative_A_per_A",
                                                 StrictJson::Type::kArray, error);
        std::vector<double> values;
        const bool valid_name =
            name != nullptr && !name->string.empty() && name->string.size() <= 128 &&
            std::all_of(name->string.begin(), name->string.end(), [](unsigned char c) {
              return std::isalnum(c) || c == '_' || c == '-' || c == '.';
            });
        if (!valid_name || direction == nullptr || names.count(name->string) != 0 ||
            !StrictNumberArray(*direction, &values) ||
            values.size() != request->circuit_currents_a.size() ||
            !std::all_of(values.begin(), values.end(),
                         [](double value) { return std::isfinite(value); })) {
          if (error->empty())
            *error = "invalid tangent RHS name or circuit derivative vector";
          return false;
        }
        names.emplace(name->string, true);
        request->tangent_multi_rhs.rhs.emplace_back(name->string, std::move(values));
      }
      request->tangent_requested = true;
    }
  }
  if (v3 || v4) {
    const StrictJson &sector_request = root.object.at("sector_performance");
    if (!ExactObject(sector_request, {"schema_version", "model_invariance_certificate_sha256"},
                     error))
      return false;
    const StrictJson *sector_schema =
        JsonMember(sector_request, "schema_version", StrictJson::Type::kString, error);
    const StrictJson *certificate_sha = JsonMember(
        sector_request, "model_invariance_certificate_sha256", StrictJson::Type::kString, error);
    if (sector_schema == nullptr || certificate_sha == nullptr ||
        !JsonLowerSha256(certificate_sha->string) ||
        sector_schema->string != (v4 ? "gpu_femm_motor_sector_performance_v2"
                                     : "gpu_femm_motor_sector_performance_v1")) {
      *error = "invalid gpu_femm_motor_sector_performance_v1 request";
      return false;
    }
    request->sector_performance.enabled = true;
    request->sector_performance.model_invariance_certificate_sha256 = certificate_sha->string;
  }
  // Native AGE cyclic sectors are deliberately nonlinear base-only.  Tangent
  // and suspension calculations retain their validated full-circle routes.
  if (v4 && request->tangent_requested) {
    *error = "gpu_femm_motor_sample_v4 rejects tangent_multi_rhs";
    return false;
  }
  return true;
}

bool ReadMotorSampleRequestJson(const std::string &json, MotorSampleRequest *request,
                                std::string *error)
{
  StrictJson root;
  StrictJsonParser parser(json);
  return request != nullptr && error != nullptr && parser.Parse(&root, error) &&
         ReadMotorSampleRequestValue(root, request, error);
}

bool RequestMatchesArtifact(const MotorSampleRequest &request, const GpuFemmMeshArtifact &artifact)
{
  const bool sector_request = request.protocol == "gpu_femm_motor_sample_v3";
  const bool native_sector_request = request.protocol == "gpu_femm_motor_sample_v4";
  if (sector_request != (artifact.has_motor_sector && !artifact.has_native_age_sector) ||
      native_sector_request != artifact.has_native_age_sector ||
      (sector_request &&
       (artifact.schema_version != "gpu_femm_mesh_v3" || artifact.sector_angle_deg != 180.0 ||
        artifact.full_machine_sector_count != 2 || artifact.electrical_pole_pairs <= 0 ||
        artifact.suspension_spatial_order != artifact.electrical_pole_pairs + 1 ||
        artifact.sector_boundary_sign != ((artifact.electrical_pole_pairs % 2) == 0 ? 1 : -1) ||
        !request.sector_performance.enabled ||
        request.sector_performance.model_invariance_certificate_sha256 !=
            artifact.model_invariance_certificate_sha256)) ||
      (native_sector_request &&
       (artifact.schema_version != "gpu_femm_mesh_v4" || artifact.full_machine_sector_count < 2 ||
        artifact.sector_angle_deg != 360.0 / artifact.full_machine_sector_count ||
        !artifact.has_sliding_band || artifact.model.air_gap_elements.empty() ||
        !request.sector_performance.enabled ||
        request.sector_performance.model_invariance_certificate_sha256 !=
            artifact.model_invariance_certificate_sha256)))
    return false;
  const bool matching_pose =
      artifact.has_sliding_band
          ? request.displacement_mm[0] == 0.0 && request.displacement_mm[1] == 0.0
          : request.rotor_angle_deg == artifact.rotor_angle_deg &&
                request.displacement_mm[0] == artifact.displacement_mm[0] &&
                request.displacement_mm[1] == artifact.displacement_mm[1];
  bool matching_sliding_postprocess = true;
  if (artifact.has_sliding_band) {
    const auto has_group = [&artifact](int32_t group) {
      return std::any_of(
          artifact.model.materials.begin(), artifact.model.materials.end(),
          [group](const NonlinearMaterial &material) { return material.group_number == group; });
    };
    matching_sliding_postprocess =
        has_group(request.selected_group_number) && has_group(request.air_group_number);
    if (matching_sliding_postprocess && !request.airgap_angles_deg.empty()) {
      matching_sliding_postprocess = !artifact.model.air_gap_elements.empty();
      for (const AirGapElement &age : artifact.model.air_gap_elements) {
        const double inner_radius_mm = 1000.0 * age.inner_radius_m;
        const double outer_radius_mm = 1000.0 * age.outer_radius_m;
        const double tolerance_mm = 1e-9 * std::max(1.0, std::abs(outer_radius_mm));
        matching_sliding_postprocess = matching_sliding_postprocess &&
                                       request.airgap_radius_mm >= inner_radius_mm - tolerance_mm &&
                                       request.airgap_radius_mm <= outer_radius_mm + tolerance_mm;
      }
    }
  }
  return request.base_motor_fem_sha256 == artifact.base_motor_fem_sha256 &&
         request.source_fem_sha256 == artifact.pose_fem_sha256 && matching_pose &&
         matching_sliding_postprocess &&
         request.circuit_currents_a.size() == static_cast<size_t>(artifact.model.circuit_count);
}

bool ApplySlidingBandAngle(double requested_angle_deg, const GpuFemmMeshArtifact &artifact,
                           NonlinearModel *model)
{
  if (model == nullptr)
    return false;
  *model = artifact.model;
  if (!artifact.has_sliding_band)
    return true;
  const double delta_deg = requested_angle_deg - artifact.rotor_angle_deg;
  if (!std::isfinite(delta_deg))
    return false;
  for (AirGapElement &age : model->air_gap_elements) {
    const size_t sectors = age.quad_points.size() - 1;
    if (sectors == 0 || !(age.arc_length_deg > 0.0))
      return false;
    // FEMM regenerates AGE point records for every rotor angle.  Its fractional
    // shift is only part of that operation: crossing a ring cell also rotates
    // the inner-side node/weight sequence.  Keep the stator-side records fixed
    // and reproduce that cyclic inner-side remap from the reference artifact.
    const double shifted =
        age.inner_shift + delta_deg * static_cast<double>(sectors) / age.arc_length_deg;
    const int64_t integer_cells = static_cast<int64_t>(std::floor(shifted));
    age.inner_shift = shifted - static_cast<double>(integer_cells);
    std::vector<AirGapQuadPoint> source = age.quad_points;
    const int64_t count = static_cast<int64_t>(sectors);
    for (size_t k = 0; k < sectors; ++k) {
      int64_t source_index = static_cast<int64_t>(k) - integer_cells;
      source_index %= count;
      if (source_index < 0)
        source_index += count;
      for (int local = 0; local < 2; ++local) {
        age.quad_points[k].node[local] = source[static_cast<size_t>(source_index)].node[local];
        age.quad_points[k].weight[local] = source[static_cast<size_t>(source_index)].weight[local];
      }
    }
    // The final point duplicates the first point on the inner ring while its
    // outer-side interpolation stays the original terminal record.
    for (int local = 0; local < 2; ++local) {
      age.quad_points[sectors].node[local] = age.quad_points[0].node[local];
      age.quad_points[sectors].weight[local] = age.quad_points[0].weight[local];
    }
  }
  return true;
}

bool ApplySlidingBandRotorAngle(const MotorSampleRequest &request,
                                const GpuFemmMeshArtifact &artifact, NonlinearModel *model)
{
  return ApplySlidingBandAngle(request.rotor_angle_deg, artifact, model);
}

bool MakeMotorSectorTangentModel(const GpuFemmMeshArtifact &artifact,
                                 const NonlinearModel &posed_base_model,
                                 NonlinearModel *tangent_model)
{
  if (tangent_model == nullptr || !artifact.has_motor_sector ||
      artifact.suspension_spatial_order != artifact.electrical_pole_pairs + 1)
    return false;
  *tangent_model = posed_base_model;
  const int8_t tangent_sign =
      static_cast<int8_t>((artifact.suspension_spatial_order % 2) == 0 ? 1 : -1);
  for (const int32_t apex : artifact.apex_self_pair_nodes) {
    for (size_t index = 0; index < tangent_model->dirichlet_nodes.size();) {
      if (tangent_model->dirichlet_nodes[index] == apex) {
        tangent_model->dirichlet_nodes.erase(tangent_model->dirichlet_nodes.begin() + index);
        tangent_model->dirichlet_a_wb_per_m.erase(tangent_model->dirichlet_a_wb_per_m.begin() +
                                                  index);
      } else {
        ++index;
      }
    }
    if (tangent_sign < 0) {
      tangent_model->dirichlet_nodes.push_back(apex);
      tangent_model->dirichlet_a_wb_per_m.push_back(0.0);
    }
  }
  for (SignedNodeConstraint &constraint : tangent_model->node_constraints)
    constraint.sign = tangent_sign;
  for (AirGapElement &age : tangent_model->air_gap_elements)
    age.antiperiodic = tangent_sign < 0;
  return ValidateNonlinearModel(*tangent_model) == Status::kOk;
}

std::vector<int8_t> MapSectorAirgapAnglesToRepresentative(double sector_start_angle_rad,
                                                          int8_t repeat_sign,
                                                          FrozenPostprocessOptions *options)
{
  std::vector<int8_t> signs;
  if (options == nullptr || !std::isfinite(sector_start_angle_rad))
    return signs;
  signs.reserve(options->airgap_angles_rad.size());
  constexpr double kPi = 3.141592653589793238462643383279502884;
  for (double &angle : options->airgap_angles_rad) {
    const double turns = std::floor((angle - sector_start_angle_rad) / kPi);
    int64_t copy = static_cast<int64_t>(turns);
    angle -= static_cast<double>(copy) * kPi;
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
                             std::max({1.0, std::abs(angle), std::abs(sector_start_angle_rad)});
    if (angle < sector_start_angle_rad - tolerance) {
      angle += kPi;
      --copy;
    } else if (angle >= sector_start_angle_rad + kPi - tolerance) {
      angle -= kPi;
      ++copy;
    }
    if (std::abs(angle - sector_start_angle_rad) <= tolerance)
      angle = sector_start_angle_rad;
    const bool odd_copy = (std::abs(copy) % 2) != 0;
    signs.push_back(odd_copy ? repeat_sign : 1);
  }
  return signs;
}

// Fold global probes into a general k-sector AGE.  FEMM's writepoly expands
// all sector copies into a virtual 360-degree ring, sorts it in global angle,
// then emits q0 as the predecessor that brackets global zero.  Consequently
// the native AGE Fourier basis is global-zero referenced even when a cropped
// physical sector starts elsewhere.  P/AP parity follows those same global
// virtual copies, not the declared crop cut or q0's stored local node angle.
std::vector<int8_t>
MapNativeAgeSectorAirgapAnglesToRepresentative(const GpuFemmMeshArtifact &artifact,
                                               FrozenPostprocessOptions *options)
{
  std::vector<int8_t> signs;
  if (options == nullptr || artifact.full_machine_sector_count < 2)
    return signs;
  constexpr double kPi = 3.141592653589793238462643383279502884;
  const double span = artifact.sector_angle_deg * kPi / 180.0;
  if (!(span > 0.0))
    return signs;
  signs.reserve(options->airgap_angles_rad.size());
  for (double &angle : options->airgap_angles_rad) {
    const double turns = std::floor(angle / span);
    int64_t copy = static_cast<int64_t>(turns);
    double local = angle - static_cast<double>(copy) * span;
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(local));
    if (local < -tolerance) {
      local += span;
      --copy;
    } else if (local >= span - tolerance) {
      local -= span;
      ++copy;
    }
    if (std::abs(local) <= tolerance)
      local = 0.0;
    angle = local;
    const bool odd_copy = (std::abs(copy) % 2) != 0;
    signs.push_back(odd_copy && artifact.sector_boundary_sign < 0 ? -1 : 1);
  }
  return signs;
}

bool ApplySectorAirgapRepeatSigns(const std::vector<int8_t> &signs,
                                  FrozenPostprocessResult *postprocess)
{
  if (postprocess == nullptr || signs.size() != postprocess->airgap_samples.size())
    return false;
  for (size_t sample = 0; sample < signs.size(); ++sample)
    postprocess->airgap_samples[sample].radial_b_t *= signs[sample];
  return true;
}

bool ApplySectorTangentAirgapRepeatSigns(const std::vector<int8_t> &signs,
                                         TangentPostprocessResult *postprocess)
{
  if (postprocess == nullptr ||
      signs.size() != postprocess->airgap_radial_flux_density_t_per_a.size())
    return false;
  for (size_t sample = 0; sample < signs.size(); ++sample)
    postprocess->airgap_radial_flux_density_t_per_a[sample] *= signs[sample];
  return true;
}

} // namespace gpu_femm

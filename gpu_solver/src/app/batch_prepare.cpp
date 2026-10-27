#include "app/motor_batch.h"
#include "app/sample_adapter.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "io/mesh_artifact.h"
#include "io/sample_request.h"
#include "io/strict_json.h"
#include "physics/nonlinear_model.h"

namespace gpu_femm
{

static bool BatchTaskIdValid(const std::string &id)
{
  return !id.empty() && id.size() <= 256 && std::all_of(id.begin(), id.end(), [](unsigned char c) {
    return std::isalnum(c) || c == '_' || c == '-' || c == '.';
  });
}

bool ReadMotorBatchRequestJson(const std::string &json, MotorBatchRequest *request,
                               std::string *error)
{
  StrictJson root;
  StrictJsonParser parser(json);
  if (request == nullptr || error == nullptr || !parser.Parse(&root, error) ||
      !ExactObject(root, {"protocol", "max_items_per_chunk", "items"}, error))
    return false;
  const StrictJson *protocol = JsonMember(root, "protocol", StrictJson::Type::kString, error);
  const StrictJson *chunk =
      JsonMember(root, "max_items_per_chunk", StrictJson::Type::kNumber, error);
  const StrictJson *items = JsonMember(root, "items", StrictJson::Type::kArray, error);
  int32_t max_items = 0;
  if (protocol == nullptr || chunk == nullptr || items == nullptr ||
      (protocol->string != "gpu_femm_motor_batch_v1" &&
       protocol->string != "gpu_femm_motor_batch_v2" &&
       protocol->string != "gpu_femm_motor_batch_v3" &&
       protocol->string != "gpu_femm_motor_batch_v4") ||
      !JsonInteger(*chunk, &max_items) || max_items < 1 || max_items > 4096 ||
      items->array.empty() || items->array.size() > 4096) {
    if (error->empty())
      *error = "invalid gpu_femm_motor_batch_v1 request";
    return false;
  }
  request->protocol = protocol->string;
  request->max_items_per_chunk = max_items;
  request->items.clear();
  std::map<std::string, bool> ids;
  for (const StrictJson &item : items->array) {
    if (!ExactObject(item, {"task_id", "request"}, error))
      return false;
    const StrictJson *id = JsonMember(item, "task_id", StrictJson::Type::kString, error);
    const StrictJson *sample = JsonMember(item, "request", StrictJson::Type::kObject, error);
    MotorBatchItem parsed;
    const std::string expected_sample_protocol =
        request->protocol == "gpu_femm_motor_batch_v4"
            ? "gpu_femm_motor_sample_v4"
            : (request->protocol == "gpu_femm_motor_batch_v3"
                   ? "gpu_femm_motor_sample_v3"
                   : (request->protocol == "gpu_femm_motor_batch_v2" ? "gpu_femm_motor_sample_v2"
                                                                     : "gpu_femm_motor_sample_v1"));
    if (id == nullptr || sample == nullptr || !BatchTaskIdValid(id->string) ||
        ids.count(id->string) != 0 ||
        !ReadMotorSampleRequestValue(*sample, &parsed.request, error)) {
      if (error->empty())
        *error = "invalid or duplicate batch task_id";
      return false;
    }
    if (parsed.request.protocol != expected_sample_protocol) {
      *error = "motor batch and item protocol versions must match";
      return false;
    }
    parsed.task_id = id->string;
    ids.emplace(parsed.task_id, true);
    request->items.push_back(std::move(parsed));
  }
  return true;
}

static std::string NonlinearModelFingerprint(const NonlinearModel &model)
{
  std::ostringstream bytes;
  bytes << std::setprecision(17) << model.depth_m << '|' << model.circuit_count << '|';
  for (const Node &node : model.nodes)
    bytes << node.x_m << ',' << node.y_m << ';';
  for (const NonlinearMaterial &material : model.materials)
    bytes << material.reluctivity_zero_m_per_h << ',' << material.alpha_per_t2 << ','
          << material.source_j_per_a << ',' << material.h_c_a_per_m << ','
          << material.magnetization_deg << ',' << material.bh_curve_index << ','
          << material.circuit_index << ',' << material.group_number << ';';
  for (const NonlinearBhCurve &curve : model.bh_curves) {
    for (double value : curve.h_a_per_m)
      bytes << value << ',';
    bytes << ':';
    for (double value : curve.b_t)
      bytes << value << ',';
    bytes << ';';
  }
  for (const NonlinearTriangle &triangle : model.triangles)
    bytes << triangle.node[0] << ',' << triangle.node[1] << ',' << triangle.node[2] << ','
          << triangle.material << ';';
  for (const AirGapElement &age : model.air_gap_elements) {
    bytes << 'A' << age.antiperiodic << ',' << age.center_x_m << ',' << age.center_y_m << ','
          << age.inner_radius_m << ',' << age.outer_radius_m << ',' << age.arc_length_deg << ','
          << age.inner_shift << ',' << age.outer_shift << ';';
    for (const AirGapQuadPoint &point : age.quad_points)
      for (int local = 0; local < 4; ++local)
        bytes << point.node[local] << ',' << point.weight[local] << ',';
    bytes << ';';
  }
  for (int32_t node : model.dirichlet_nodes)
    bytes << node << ',';
  bytes << ':';
  for (double value : model.dirichlet_a_wb_per_m)
    bytes << value << ',';
  bytes << ':';
  for (const SignedNodeConstraint &constraint : model.node_constraints)
    bytes << constraint.node_a << ',' << constraint.node_b << ','
          << static_cast<int>(constraint.sign) << ';';
  return Sha256Hex(bytes.str());
}

MotorBatchArtifactLoadPlan MakeMotorBatchArtifactLoadPlan(const MotorBatchRequest &request)
{
  MotorBatchArtifactLoadPlan plan;
  plan.item_slots.reserve(request.items.size());
  std::map<std::string, size_t> slot_by_path;
  for (const MotorBatchItem &item : request.items) {
    const std::string canonical_path = CanonicalArtifactPathKey(item.request.mesh_artifact_path);
    const auto found = slot_by_path.find(canonical_path);
    if (found != slot_by_path.end()) {
      plan.item_slots.push_back(found->second);
      ++plan.slot_use_counts[found->second];
      continue;
    }
    const size_t slot = plan.canonical_paths.size();
    slot_by_path.emplace(canonical_path, slot);
    plan.canonical_paths.push_back(canonical_path);
    plan.representative_paths.push_back(item.request.mesh_artifact_path);
    plan.slot_use_counts.push_back(1);
    plan.item_slots.push_back(slot);
  }
  return plan;
}

void LoadMotorBatchArtifact(const std::string &path, MotorBatchArtifactLoad *load)
{
  std::ifstream artifact_file(path, std::ios::binary);
  std::stringstream artifact_bytes;
  artifact_bytes << artifact_file.rdbuf();
  if (!artifact_file) {
    load->status = Status::kInputIo;
    return;
  }
  const std::string bytes = artifact_bytes.str();
  load->artifact_sha256 = Sha256Hex(bytes);
  std::string error;
  if (!ParseGpuFemmMeshArtifactJson(bytes, &load->artifact, &error)) {
    load->status = Status::kInvalidArgument;
    return;
  }
  load->fingerprint = NonlinearModelFingerprint(load->artifact.model);
  load->status = Status::kOk;
}

// SolveBatch can share its CUDA CSR operator only when every item has the
// exact same assembled model.  For v2 this deliberately keys the immutable
// artifact bytes and the requested mechanical pose, rather than treating all
// sliding-band samples as one geometry.  Different current vectors and
// postprocess requests do not change that operator and may therefore batch.
std::string MotorBatchOperatorKey(const MotorSampleRequest &request,
                                  const MotorBatchArtifactLoad &load, const NonlinearModel &model)
{
  if (!load.artifact.has_sliding_band)
    return NonlinearModelFingerprint(model);
  std::ostringstream key;
  key << std::setprecision(17) << "sliding|" << load.artifact_sha256 << '|'
      << request.rotor_angle_deg << '|' << request.displacement_mm[0] << ','
      << request.displacement_mm[1];
  return key.str();
}

// Return stable first-seen groups so the execution order remains predictable;
// response placement always uses request_index and therefore stays in caller
// order even when compatible samples were interleaved in the input.
std::vector<std::vector<size_t>>
GroupMotorBatchPreparedItemsByOperator(const std::vector<MotorBatchPreparedItem> &prepared)
{
  std::map<std::string, size_t> group_by_key;
  std::vector<std::vector<size_t>> groups;
  for (size_t index = 0; index < prepared.size(); ++index) {
    const auto found = group_by_key.find(prepared[index].operator_key);
    if (found != group_by_key.end()) {
      groups[found->second].push_back(index);
      continue;
    }
    group_by_key.emplace(prepared[index].operator_key, groups.size());
    groups.push_back({index});
  }
  return groups;
}

} // namespace gpu_femm

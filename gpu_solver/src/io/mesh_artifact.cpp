#include "io/mesh_artifact.h"
#include "io/strict_json.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <utility>

namespace gpu_femm
{

bool ParseGpuFemmMeshArtifactJson(const std::string &text, GpuFemmMeshArtifact *artifact,
                                  std::string *error)
{
  if (artifact == nullptr || error == nullptr)
    return false;
  StrictJson root;
  StrictJsonParser parser(text);
  if (!parser.Parse(&root, error))
    return false;
  const StrictJson *schema = JsonMember(root, "schema_version", StrictJson::Type::kString, error);
  if (schema == nullptr)
    return false;
  const bool legacy_neutral_schema = schema->string == "gpu_femm_planar_dc_mesh_v1";
  const bool generic_neutral_schema = schema->string == "gpu_femm_magnetostatic_mesh_v1";
  const bool neutral_schema = legacy_neutral_schema || generic_neutral_schema;
  const bool motor_v1 = schema->string == "gpu_femm_mesh_v1";
  const bool motor_v2 = schema->string == "gpu_femm_mesh_v2";
  const bool motor_v3 = schema->string == "gpu_femm_mesh_v3";
  const bool motor_v4 = schema->string == "gpu_femm_mesh_v4";
  if ((!neutral_schema && !motor_v1 && !motor_v2 && !motor_v3 && !motor_v4) ||
      !(neutral_schema ? ExactObject(root,
                                     {"schema_version", "source_fem_sha256",
                                      "canonical_identity_sha256", "resolved"},
                                     error)
                       : ExactObject(root,
                                     {"schema_version", "base_motor_fem_sha256",
                                      "source_fem_sha256", "canonical_identity_sha256", "resolved"},
                                     error)))
    return false;
  const StrictJson *source_sha =
      JsonMember(root, "source_fem_sha256", StrictJson::Type::kString, error);
  const StrictJson *root_base_sha =
      neutral_schema ? source_sha
                     : JsonMember(root, "base_motor_fem_sha256", StrictJson::Type::kString, error);
  const StrictJson *identity =
      JsonMember(root, "canonical_identity_sha256", StrictJson::Type::kString, error);
  const StrictJson *resolved = JsonMember(root, "resolved", StrictJson::Type::kObject, error);
  if (schema == nullptr || root_base_sha == nullptr || source_sha == nullptr ||
      identity == nullptr || resolved == nullptr ||
      !(neutral_schema ? JsonLowerSha256(root_base_sha->string)
                       : JsonSha256(root_base_sha->string)) ||
      !(neutral_schema ? JsonLowerSha256(source_sha->string) : JsonSha256(source_sha->string)) ||
      !(neutral_schema ? JsonLowerSha256(identity->string) : JsonSha256(identity->string)) ||
      !(neutral_schema
            ? (generic_neutral_schema
                   ? ExactObject(*resolved,
                                 {"source_fem_sha256", "model", "nodes_mm", "triangles", "regions",
                                  "materials", "circuits", "outer_dirichlet", "node_constraints"},
                                 error)
                   : ExactObject(*resolved,
                                 {"source_fem_sha256", "model", "nodes_mm", "triangles", "regions",
                                  "materials", "circuits", "outer_dirichlet"},
                                 error))
            : (motor_v1
                   ? ExactObject(*resolved,
                                 {"source_fem_sha256", "base_motor_fem_sha256", "model", "pose",
                                  "nodes_mm", "triangles", "regions", "materials", "circuits",
                                  "outer_dirichlet"},
                                 error)
                   : (motor_v2
                          ? ExactObject(*resolved,
                                        {"source_fem_sha256", "base_motor_fem_sha256", "model",
                                         "pose", "nodes_mm", "triangles", "regions", "materials",
                                         "circuits", "outer_dirichlet", "air_gap_elements"},
                                        error)
                          : ((motor_v3 || motor_v4)
                                 ? ExactObject(*resolved,
                                               {"source_fem_sha256", "base_motor_fem_sha256",
                                                "model", "pose", "nodes_mm", "triangles", "regions",
                                                "materials", "circuits", "outer_dirichlet",
                                                "air_gap_elements", "node_constraints", "sector"},
                                               error)
                                 : false))))) {
    if (error->empty())
      *error = "invalid GPU FEMM mesh artifact header";
    return false;
  }
  if (neutral_schema && CanonicalJsonSha256(*resolved) != identity->string) {
    *error = "canonical artifact identity mismatch";
    return false;
  }
  const StrictJson *resolved_sha =
      JsonMember(*resolved, "source_fem_sha256", StrictJson::Type::kString, error);
  const StrictJson *base_sha = neutral_schema ? resolved_sha
                                              : JsonMember(*resolved, "base_motor_fem_sha256",
                                                           StrictJson::Type::kString, error);
  const StrictJson *model = JsonMember(*resolved, "model", StrictJson::Type::kObject, error);
  const StrictJson *pose =
      neutral_schema ? nullptr : JsonMember(*resolved, "pose", StrictJson::Type::kObject, error);
  const StrictJson *nodes = JsonMember(*resolved, "nodes_mm", StrictJson::Type::kArray, error);
  const StrictJson *triangles =
      JsonMember(*resolved, "triangles", StrictJson::Type::kObject, error);
  const StrictJson *regions = JsonMember(*resolved, "regions", StrictJson::Type::kArray, error);
  const StrictJson *materials = JsonMember(*resolved, "materials", StrictJson::Type::kArray, error);
  const StrictJson *circuits = JsonMember(*resolved, "circuits", StrictJson::Type::kArray, error);
  const StrictJson *boundary =
      JsonMember(*resolved, "outer_dirichlet", StrictJson::Type::kObject, error);
  const StrictJson *node_constraints =
      (generic_neutral_schema || motor_v3 || motor_v4)
          ? JsonMember(*resolved, "node_constraints", StrictJson::Type::kArray, error)
          : nullptr;
  const StrictJson *sector = (motor_v3 || motor_v4)
                                 ? JsonMember(*resolved, "sector", StrictJson::Type::kObject, error)
                                 : nullptr;
  if (resolved_sha == nullptr || base_sha == nullptr || model == nullptr ||
      (!neutral_schema && pose == nullptr) || nodes == nullptr || triangles == nullptr ||
      regions == nullptr || materials == nullptr || circuits == nullptr || boundary == nullptr ||
      ((generic_neutral_schema || motor_v3 || motor_v4) && node_constraints == nullptr) ||
      ((motor_v3 || motor_v4) && sector == nullptr) ||
      !(neutral_schema ? JsonLowerSha256(resolved_sha->string)
                       : JsonSha256(resolved_sha->string)) ||
      !(neutral_schema ? JsonLowerSha256(base_sha->string) : JsonSha256(base_sha->string)) ||
      resolved_sha->string != source_sha->string || base_sha->string != root_base_sha->string) {
    if (error->empty())
      *error = "source FEM SHA mismatch";
    return false;
  }
  if (!ExactObject(*model, {"depth_mm", "problem_type", "frequency_hz"}, error))
    return false;
  const StrictJson *depth = JsonMember(*model, "depth_mm", StrictJson::Type::kNumber, error);
  const StrictJson *type = JsonMember(*model, "problem_type", StrictJson::Type::kString, error);
  const StrictJson *frequency =
      JsonMember(*model, "frequency_hz", StrictJson::Type::kNumber, error);
  const bool planar_problem = type != nullptr && type->string == "planar";
  const bool axisymmetric_problem = type != nullptr && type->string == "axisymmetric";
  if (depth == nullptr || type == nullptr || frequency == nullptr || !(depth->number > 0.0) ||
      frequency->number != 0.0 || (legacy_neutral_schema && !planar_problem) ||
      (!generic_neutral_schema && !planar_problem) ||
      (generic_neutral_schema && !planar_problem && !axisymmetric_problem)) {
    *error = "unsupported model physics";
    return false;
  }
  const StrictJson *rotor_angle = nullptr;
  const StrictJson *displacement = nullptr;
  if (!neutral_schema) {
    if (!ExactObject(*pose, {"rotor_angle_deg", "displacement_mm"}, error))
      return false;
    rotor_angle = JsonMember(*pose, "rotor_angle_deg", StrictJson::Type::kNumber, error);
    displacement = JsonMember(*pose, "displacement_mm", StrictJson::Type::kArray, error);
    if (rotor_angle == nullptr || displacement == nullptr || displacement->array.size() != 2 ||
        displacement->array[0].type != StrictJson::Type::kNumber ||
        displacement->array[1].type != StrictJson::Type::kNumber) {
      *error = "invalid pose";
      return false;
    }
  }
  if (nodes->array.size() < 3 || materials->array.empty() || regions->array.empty()) {
    *error = "empty mesh/material input";
    return false;
  }

  GpuFemmMeshArtifact parsed;
  parsed.schema_version = schema->string;
  bool is_sliding_band_motor = motor_v2 || motor_v4;
  if (motor_v3) {
    const auto sector_ages = resolved->object.find("air_gap_elements");
    if (sector_ages == resolved->object.end() ||
        sector_ages->second.type != StrictJson::Type::kArray ||
        !sector_ages->second.array.empty()) {
      *error = "gpu_femm_mesh_v3 requires a conforming posed sector mesh; native AGE sectors are "
               "not supported";
      return false;
    }
    is_sliding_band_motor = false;
  }
  if (motor_v4) {
    const auto sector_ages = resolved->object.find("air_gap_elements");
    if (sector_ages == resolved->object.end() ||
        (sector_ages->second.type != StrictJson::Type::kArray &&
         sector_ages->second.type != StrictJson::Type::kObject) ||
        (sector_ages->second.type == StrictJson::Type::kArray &&
         sector_ages->second.array.empty())) {
      *error = "gpu_femm_mesh_v4 requires a nonempty native AGE sector";
      return false;
    }
    parsed.has_native_age_sector = true;
  }
  parsed.has_sliding_band = is_sliding_band_motor;
  parsed.pose_fem_sha256 = source_sha->string;
  parsed.base_motor_fem_sha256 = base_sha->string;
  parsed.canonical_identity_sha256 = identity->string;
  parsed.rotor_angle_deg = neutral_schema ? 0.0 : rotor_angle->number;
  parsed.displacement_mm[0] = neutral_schema ? 0.0 : displacement->array[0].number;
  parsed.displacement_mm[1] = neutral_schema ? 0.0 : displacement->array[1].number;
  parsed.model.depth_m = depth->number * 1e-3;
  parsed.model.problem_type =
      axisymmetric_problem ? MagneticProblemType::kAxisymmetric : MagneticProblemType::kPlanar;
  for (const StrictJson &pair : nodes->array) {
    if (pair.type != StrictJson::Type::kArray || pair.array.size() != 2 ||
        pair.array[0].type != StrictJson::Type::kNumber ||
        pair.array[1].type != StrictJson::Type::kNumber) {
      *error = "invalid node coordinate";
      return false;
    }
    parsed.model.nodes.push_back({pair.array[0].number * 1e-3, pair.array[1].number * 1e-3});
  }
  std::map<int32_t, int32_t> material_lookup;
  for (size_t index = 0; index < materials->array.size(); ++index) {
    const StrictJson &entry = materials->array[index];
    if (!ExactObject(entry,
                     {"id", "name", "mu_x", "mu_y", "H_c_A_per_m", "B_T", "H_A_per_m", "lam_type",
                      "lam_fill"},
                     error))
      return false;
    const StrictJson *id = JsonMember(entry, "id", StrictJson::Type::kNumber, error);
    const StrictJson *material_name = JsonMember(entry, "name", StrictJson::Type::kString, error);
    const StrictJson *mux = JsonMember(entry, "mu_x", StrictJson::Type::kNumber, error);
    const StrictJson *muy = JsonMember(entry, "mu_y", StrictJson::Type::kNumber, error);
    const StrictJson *hc = JsonMember(entry, "H_c_A_per_m", StrictJson::Type::kNumber, error);
    const StrictJson *b = JsonMember(entry, "B_T", StrictJson::Type::kArray, error);
    const StrictJson *h = JsonMember(entry, "H_A_per_m", StrictJson::Type::kArray, error);
    const StrictJson *lam_type = JsonMember(entry, "lam_type", StrictJson::Type::kNumber, error);
    const StrictJson *lam_fill = JsonMember(entry, "lam_fill", StrictJson::Type::kNumber, error);
    int32_t material_id = -1;
    if (id == nullptr || material_name == nullptr || mux == nullptr || muy == nullptr ||
        hc == nullptr || b == nullptr || h == nullptr || lam_type == nullptr ||
        lam_fill == nullptr || !JsonInteger(*id, &material_id) || material_id < 0 ||
        material_name->string.empty() || material_lookup.count(material_id) != 0 ||
        !(mux->number > 0.0) || mux->number != muy->number || hc->number < 0.0 ||
        lam_type->number != 0.0 || lam_fill->number != 1.0 || b->array.size() != h->array.size()) {
      *error = "invalid material definition";
      return false;
    }
    NonlinearMaterial material;
    material.reluctivity_zero_m_per_h = 1.0 / (kMu0 * mux->number);
    material.h_c_a_per_m = hc->number;
    if (b->array.size() > 4096) {
      *error = "B-H curve exceeds point cap";
      return false;
    }
    if (!b->array.empty()) {
      if (b->array.size() < 2) {
        *error = "B-H curve needs two points";
        return false;
      }
      NonlinearBhCurve curve;
      for (size_t point = 0; point < b->array.size(); ++point) {
        if (b->array[point].type != StrictJson::Type::kNumber ||
            h->array[point].type != StrictJson::Type::kNumber || b->array[point].number < 0.0 ||
            h->array[point].number < 0.0 ||
            (point > 0 && (b->array[point].number < b->array[point - 1].number ||
                           h->array[point].number < h->array[point - 1].number))) {
          *error = "invalid B-H point";
          return false;
        }
        curve.b_t.push_back(b->array[point].number);
        curve.h_a_per_m.push_back(h->array[point].number);
      }
      if (!BuildNonlinearBhSpline(&curve)) {
        *error = "invalid B-H spline";
        return false;
      }
      material.bh_curve_index = static_cast<int32_t>(parsed.model.bh_curves.size());
      parsed.model.bh_curves.push_back(std::move(curve));
    }
    material_lookup.emplace(material_id, static_cast<int32_t>(parsed.model.materials.size()));
    parsed.model.materials.push_back(material);
  }
  // Regions are expanded into per-label material entries so PM angle, FEMM
  // group, and coil source are never accidentally shared across labels.
  std::map<int32_t, int32_t> circuit_lookup;
  for (size_t index = 0; index < circuits->array.size(); ++index) {
    const StrictJson &entry = circuits->array[index];
    if (!ExactObject(entry, {"index", "name", "type", "current_A"}, error))
      return false;
    const StrictJson *id = JsonMember(entry, "index", StrictJson::Type::kNumber, error);
    const StrictJson *name = JsonMember(entry, "name", StrictJson::Type::kString, error);
    const StrictJson *circuit_type = JsonMember(entry, "type", StrictJson::Type::kString, error);
    const StrictJson *current = JsonMember(entry, "current_A", StrictJson::Type::kNumber, error);
    int32_t circuit = -1;
    if (id == nullptr || name == nullptr || circuit_type == nullptr || current == nullptr ||
        !JsonInteger(*id, &circuit) || circuit != static_cast<int32_t>(index) ||
        name->string.empty() || circuit_type->string != "series") {
      *error = "circuits must be ordered contiguous current-driven series circuits";
      return false;
    }
    circuit_lookup.emplace(circuit, circuit);
    parsed.circuit_currents_a.push_back(current->number);
  }
  parsed.model.circuit_count = static_cast<int32_t>(parsed.circuit_currents_a.size());
  std::map<int32_t, int32_t> region_lookup;
  std::vector<double> region_area_m2;
  struct RegionRaw {
    int32_t material;
    int32_t circuit;
    double turns;
    double angle;
    int32_t group;
  };
  std::vector<RegionRaw> raw_regions;
  for (const StrictJson &entry : regions->array) {
    if (!ExactObject(
            entry,
            {"id", "group_number", "material_id", "circuit_index", "turns", "pm_magnetization_deg"},
            error))
      return false;
    const StrictJson *id = JsonMember(entry, "id", StrictJson::Type::kNumber, error);
    const StrictJson *group = JsonMember(entry, "group_number", StrictJson::Type::kNumber, error);
    const StrictJson *material = JsonMember(entry, "material_id", StrictJson::Type::kNumber, error);
    const StrictJson *circuit =
        JsonMember(entry, "circuit_index", StrictJson::Type::kNumber, error);
    const StrictJson *turns = JsonMember(entry, "turns", StrictJson::Type::kNumber, error);
    const StrictJson *angle =
        JsonMember(entry, "pm_magnetization_deg", StrictJson::Type::kNumber, error);
    int32_t region_id = -1, group_id = -1, material_id = -1, circuit_id = -2;
    if (id == nullptr || group == nullptr || material == nullptr || circuit == nullptr ||
        turns == nullptr || angle == nullptr || !JsonInteger(*id, &region_id) ||
        !JsonInteger(*group, &group_id) || !JsonInteger(*material, &material_id) ||
        !JsonInteger(*circuit, &circuit_id) || region_id < 0 || group_id < 0 ||
        material_lookup.count(material_id) == 0 || region_lookup.count(region_id) != 0 ||
        circuit_id < -1 || (circuit_id == -1 && turns->number != 0.0) ||
        (circuit_id >= 0 && (circuit_lookup.count(circuit_id) == 0 || turns->number == 0.0))) {
      *error = "invalid region definition";
      return false;
    }
    region_lookup.emplace(region_id, static_cast<int32_t>(raw_regions.size()));
    raw_regions.push_back(
        {material_lookup[material_id], circuit_id, turns->number, angle->number, group_id});
    region_area_m2.push_back(0.0);
  }
  if (!ExactObject(*triangles, {"node_indices", "region_ids"}, error))
    return false;
  const StrictJson *faces = JsonMember(*triangles, "node_indices", StrictJson::Type::kArray, error);
  const StrictJson *triangle_regions =
      JsonMember(*triangles, "region_ids", StrictJson::Type::kArray, error);
  if (faces == nullptr || triangle_regions == nullptr || faces->array.empty() ||
      faces->array.size() != triangle_regions->array.size()) {
    *error = "invalid triangle arrays";
    return false;
  }
  for (size_t element = 0; element < faces->array.size(); ++element) {
    if (faces->array[element].type != StrictJson::Type::kArray ||
        faces->array[element].array.size() != 3) {
      *error = "triangle must have three node indices";
      return false;
    }
    int32_t node[3], region_id = -1;
    if (triangle_regions->array[element].type != StrictJson::Type::kNumber ||
        !JsonInteger(triangle_regions->array[element], &region_id) ||
        region_lookup.count(region_id) == 0) {
      *error = "triangle references unknown region";
      return false;
    }
    for (int local = 0; local < 3; ++local) {
      if (!JsonInteger(faces->array[element].array[local], &node[local]) || node[local] < 0 ||
          static_cast<size_t>(node[local]) >= parsed.model.nodes.size()) {
        *error = "triangle node out of range";
        return false;
      }
    }
    Node p[3] = {parsed.model.nodes[node[0]], parsed.model.nodes[node[1]],
                 parsed.model.nodes[node[2]]};
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms)) {
      *error = "triangle is not CCW";
      return false;
    }
    const int32_t mapped = region_lookup[region_id];
    region_area_m2[mapped] += terms.area_m2;
    parsed.model.triangles.push_back({{node[0], node[1], node[2]}, mapped});
  }
  for (size_t region = 0; region < raw_regions.size(); ++region) {
    const RegionRaw &raw = raw_regions[region];
    if (!(region_area_m2[region] > 0.0)) {
      *error = "region has no triangle area";
      return false;
    }
    const NonlinearMaterial base = parsed.model.materials[raw.material];
    NonlinearMaterial label = base;
    label.group_number = raw.group;
    label.circuit_index = raw.circuit;
    label.magnetization_deg = raw.angle;
    label.source_j_per_a = raw.circuit < 0 ? 0.0 : raw.turns / region_area_m2[region];
    parsed.model.materials.push_back(label);
  }
  // Replace temporary region indices by the expanded per-label material IDs.
  const int32_t labels_begin =
      static_cast<int32_t>(parsed.model.materials.size() - raw_regions.size());
  for (NonlinearTriangle &triangle : parsed.model.triangles) {
    triangle.material = labels_begin + triangle.material;
    parsed.triangle_group_numbers.push_back(parsed.model.materials[triangle.material].group_number);
  }
  if (!ExactObject(*boundary, {"node_indices", "A_Wb_per_m"}, error))
    return false;
  const StrictJson *boundary_nodes =
      JsonMember(*boundary, "node_indices", StrictJson::Type::kArray, error);
  const StrictJson *boundary_values =
      JsonMember(*boundary, "A_Wb_per_m", StrictJson::Type::kArray, error);
  if (boundary_nodes == nullptr || boundary_values == nullptr || boundary_nodes->array.empty() ||
      boundary_nodes->array.size() != boundary_values->array.size()) {
    *error = "invalid outer Dirichlet";
    return false;
  }
  for (size_t index = 0; index < boundary_nodes->array.size(); ++index) {
    int32_t node = -1;
    if (!JsonInteger(boundary_nodes->array[index], &node) || node < 0 ||
        static_cast<size_t>(node) >= parsed.model.nodes.size() ||
        boundary_values->array[index].type != StrictJson::Type::kNumber ||
        boundary_values->array[index].number != 0.0) {
      *error = "outer Dirichlet must be zero and in range";
      return false;
    }
    parsed.model.dirichlet_nodes.push_back(node);
    parsed.model.dirichlet_a_wb_per_m.push_back(0.0);
  }
  if (generic_neutral_schema || motor_v3 || motor_v4) {
    for (const StrictJson &entry : node_constraints->array) {
      if (!ExactObject(entry, {"node_a", "node_b", "relation"}, error))
        return false;
      const StrictJson *node_a = JsonMember(entry, "node_a", StrictJson::Type::kNumber, error);
      const StrictJson *node_b = JsonMember(entry, "node_b", StrictJson::Type::kNumber, error);
      const StrictJson *relation = JsonMember(entry, "relation", StrictJson::Type::kString, error);
      SignedNodeConstraint constraint;
      if (node_a == nullptr || node_b == nullptr || relation == nullptr ||
          !JsonInteger(*node_a, &constraint.node_a) || !JsonInteger(*node_b, &constraint.node_b) ||
          constraint.node_a < 0 || constraint.node_b < 0 ||
          static_cast<size_t>(constraint.node_a) >= parsed.model.nodes.size() ||
          static_cast<size_t>(constraint.node_b) >= parsed.model.nodes.size() ||
          constraint.node_a == constraint.node_b ||
          (relation->string != "periodic" && relation->string != "antiperiodic")) {
        *error = "invalid periodic node constraint";
        return false;
      }
      constraint.sign = relation->string == "periodic" ? 1 : -1;
      parsed.model.node_constraints.push_back(constraint);
    }
  }
  if (motor_v3 || motor_v4) {
    if (!ExactObject(*sector,
                     {"schema_version", "sector_start_angle_deg", "angle_deg",
                      "full_machine_sector_count", "electrical_pole_pairs",
                      "suspension_spatial_order", "boundary_relation", "circuit_partner_indices",
                      "circuit_partner_orientations", "apex_self_pair_nodes",
                      "model_invariance_certificate_sha256"},
                     error))
      return false;
    const StrictJson *sector_schema =
        JsonMember(*sector, "schema_version", StrictJson::Type::kString, error);
    const StrictJson *start_angle =
        JsonMember(*sector, "sector_start_angle_deg", StrictJson::Type::kNumber, error);
    const StrictJson *angle = JsonMember(*sector, "angle_deg", StrictJson::Type::kNumber, error);
    const StrictJson *repeats =
        JsonMember(*sector, "full_machine_sector_count", StrictJson::Type::kNumber, error);
    const StrictJson *pole_pairs =
        JsonMember(*sector, "electrical_pole_pairs", StrictJson::Type::kNumber, error);
    const StrictJson *suspension_order =
        JsonMember(*sector, "suspension_spatial_order", StrictJson::Type::kNumber, error);
    const StrictJson *boundary_relation =
        JsonMember(*sector, "boundary_relation", StrictJson::Type::kString, error);
    const StrictJson *circuit_partners =
        JsonMember(*sector, "circuit_partner_indices", StrictJson::Type::kArray, error);
    const StrictJson *circuit_orientations =
        JsonMember(*sector, "circuit_partner_orientations", StrictJson::Type::kArray, error);
    const StrictJson *apex_nodes =
        JsonMember(*sector, "apex_self_pair_nodes", StrictJson::Type::kArray, error);
    const StrictJson *invariance_sha = JsonMember(*sector, "model_invariance_certificate_sha256",
                                                  StrictJson::Type::kString, error);
    int32_t repeat_count = 0, p = 0, suspension_n = 0;
    if (sector_schema == nullptr || start_angle == nullptr || !std::isfinite(start_angle->number) ||
        angle == nullptr || repeats == nullptr || pole_pairs == nullptr ||
        suspension_order == nullptr || boundary_relation == nullptr ||
        circuit_partners == nullptr || circuit_orientations == nullptr || apex_nodes == nullptr ||
        invariance_sha == nullptr || !JsonLowerSha256(invariance_sha->string) ||
        sector_schema->string !=
            (motor_v4 ? "gpu_femm_motor_sector_v2" : "gpu_femm_motor_sector_v1") ||
        !(angle->number > 0.0) || !JsonInteger(*repeats, &repeat_count) || repeat_count < 2 ||
        !JsonInteger(*pole_pairs, &p) || p <= 0 || !JsonInteger(*suspension_order, &suspension_n) ||
        suspension_n != p + 1 ||
        (boundary_relation->string != "periodic" && boundary_relation->string != "antiperiodic") ||
        circuit_partners->array.size() != static_cast<size_t>(parsed.model.circuit_count) ||
        circuit_orientations->array.size() != static_cast<size_t>(parsed.model.circuit_count)) {
      *error = "motor sector requires positive k, p and p+1 harmonic metadata";
      return false;
    }
    if ((!motor_v4 && (angle->number != 180.0 || repeat_count != 2)) ||
        (motor_v4 && angle->number != 360.0 / static_cast<double>(repeat_count))) {
      *error = "motor sector angle/count are inconsistent";
      return false;
    }
    // A p-pole-pair scalar potential can have only a P/AP relation under a
    // sector rotation when its electrical phase advance is an integer pi.
    // Do not silently accept a Bloch phase in this real-valued solver.
    if (motor_v4 && ((2 * p) % repeat_count != 0 || (boundary_relation->string == "antiperiodic") !=
                                                        (((2 * p / repeat_count) % 2) != 0))) {
      *error = "native AGE sector boundary relation does not match electrical phase advance";
      return false;
    }
    parsed.has_motor_sector = true;
    parsed.sector_start_angle_deg = start_angle->number;
    parsed.sector_angle_deg = angle->number;
    parsed.full_machine_sector_count = repeat_count;
    parsed.electrical_pole_pairs = p;
    parsed.suspension_spatial_order = suspension_n;
    parsed.sector_boundary_sign = boundary_relation->string == "periodic" ? 1 : -1;
    parsed.model_invariance_certificate_sha256 = invariance_sha->string;
    std::vector<bool> apex_seen(parsed.model.nodes.size(), false);
    double mesh_radius_m = 1e-3;
    for (const Node &node : parsed.model.nodes)
      mesh_radius_m = std::max(mesh_radius_m, std::hypot(node.x_m, node.y_m));
    const double sector_geometry_tolerance_m = std::max(
        1e-10 * mesh_radius_m, 64.0 * std::numeric_limits<double>::epsilon() * mesh_radius_m);
    for (const SignedNodeConstraint &constraint : parsed.model.node_constraints) {
      const Node &a = parsed.model.nodes[constraint.node_a];
      const Node &b = parsed.model.nodes[constraint.node_b];
      if (!motor_v4) {
        // Preserve v3's original exact C2 predicate; using cos(pi) introduces
        // an avoidable floating residual into its established validation path.
        if (std::hypot(a.x_m + b.x_m, a.y_m + b.y_m) > sector_geometry_tolerance_m) {
          *error = "motor sector non-self constraint nodes must be related by an exact 180-degree "
                   "rotation";
          return false;
        }
      } else {
        // Canonical wire ordering (`node_a < node_b`) is unrelated to the
        // geometric generator direction. Accept either R(a)=b or R(b)=a.
        const double sector_angle_rad =
            angle->number * 3.141592653589793238462643383279502884 / 180.0;
        const double cosine = std::cos(sector_angle_rad);
        const double sine = std::sin(sector_angle_rad);
        const double forward = std::hypot(b.x_m - (cosine * a.x_m - sine * a.y_m),
                                          b.y_m - (sine * a.x_m + cosine * a.y_m));
        const double reverse = std::hypot(a.x_m - (cosine * b.x_m - sine * b.y_m),
                                          a.y_m - (sine * b.x_m + cosine * b.y_m));
        if (std::min(forward, reverse) > sector_geometry_tolerance_m) {
          *error = "motor sector non-self constraint nodes must be related by the sector rotation";
          return false;
        }
      }
    }
    for (const StrictJson &apex : apex_nodes->array) {
      int32_t node = -1;
      if (!JsonInteger(apex, &node) || node < 0 ||
          static_cast<size_t>(node) >= parsed.model.nodes.size() || apex_seen[node] ||
          std::hypot(parsed.model.nodes[node].x_m, parsed.model.nodes[node].y_m) >
              sector_geometry_tolerance_m ||
          std::any_of(parsed.model.node_constraints.begin(), parsed.model.node_constraints.end(),
                      [node](const SignedNodeConstraint &constraint) {
                        return constraint.node_a == node || constraint.node_b == node;
                      }) ||
          std::find(parsed.model.dirichlet_nodes.begin(), parsed.model.dirichlet_nodes.end(),
                    node) != parsed.model.dirichlet_nodes.end()) {
        *error = "motor sector apex self-pair nodes must be unique origin nodes separate from "
                 "physical boundaries and non-self constraints";
        return false;
      }
      apex_seen[node] = true;
      parsed.apex_self_pair_nodes.push_back(node);
    }
    std::vector<bool> partner_seen(parsed.model.circuit_count, false);
    for (size_t circuit = 0; circuit < circuit_partners->array.size(); ++circuit) {
      int32_t partner = -1, orientation = 0;
      if (!JsonInteger(circuit_partners->array[circuit], &partner) ||
          !JsonInteger(circuit_orientations->array[circuit], &orientation) || partner < 0 ||
          partner >= parsed.model.circuit_count || partner_seen[partner] ||
          (orientation != -1 && orientation != 1)) {
        *error = "invalid motor sector circuit partner mapping";
        return false;
      }
      partner_seen[partner] = true;
      parsed.circuit_partner_indices.push_back(partner);
      parsed.circuit_partner_orientations.push_back(static_cast<int8_t>(orientation));
    }
    for (size_t circuit = 0; circuit < parsed.circuit_partner_indices.size(); ++circuit) {
      int32_t current = static_cast<int32_t>(circuit);
      int sign = 1;
      for (int32_t copy = 0; copy < repeat_count; ++copy) {
        sign *= parsed.circuit_partner_orientations[current];
        current = parsed.circuit_partner_indices[current];
      }
      if (current != static_cast<int32_t>(circuit) || sign != 1) {
        *error =
            "motor sector circuit generator must close after k copies with positive orientation";
        return false;
      }
    }
    if (parsed.model.node_constraints.empty() ||
        !std::all_of(
            parsed.model.node_constraints.begin(), parsed.model.node_constraints.end(),
            [&](const SignedNodeConstraint &c) { return c.sign == parsed.sector_boundary_sign; })) {
      *error = "motor sector node constraints do not match boundary_relation";
      return false;
    }
    const int8_t base_sign = (p % 2) == 0 ? 1 : -1;
    if (parsed.sector_boundary_sign != base_sign) {
      *error = "motor sector boundary_relation does not match electrical pole-pair parity";
      return false;
    }
    parsed.model.symmetry_apex_nodes = parsed.apex_self_pair_nodes;
    if (base_sign < 0) {
      for (const int32_t node : parsed.apex_self_pair_nodes) {
        parsed.model.dirichlet_nodes.push_back(node);
        parsed.model.dirichlet_a_wb_per_m.push_back(0.0);
      }
    }
  }
  if (is_sliding_band_motor) {
    const auto ages_it = resolved->object.find("air_gap_elements");
    if (ages_it == resolved->object.end() ||
        (ages_it->second.type != StrictJson::Type::kArray &&
         ages_it->second.type != StrictJson::Type::kObject) ||
        parsed.displacement_mm[0] != 0.0 || parsed.displacement_mm[1] != 0.0 ||
        parsed.rotor_angle_deg != 0.0) {
      if (error->empty())
        *error = "sliding-band artifacts require a centered zero-degree reference mesh";
      return false;
    }
    // MATLAB's jsonencode represents a scalar struct as an object rather
    // than a one-element array.  Accept that canonical single-AGE spelling
    // while retaining the array form for multi-AGE artifacts.
    std::vector<const StrictJson *> age_entries;
    if (ages_it->second.type == StrictJson::Type::kArray) {
      if (ages_it->second.array.empty()) {
        *error = "sliding-band requires an air-gap element";
        return false;
      }
      for (const StrictJson &entry : ages_it->second.array)
        age_entries.push_back(&entry);
    } else {
      age_entries.push_back(&ages_it->second);
    }
    for (const StrictJson *entry_ptr : age_entries) {
      const StrictJson &entry = *entry_ptr;
      if (!ExactObject(entry,
                       {"name", "periodicity", "center_mm", "ri_mm", "ro_mm", "arc_length_deg",
                        "sector_count", "inner_shift", "outer_shift", "quad_points"},
                       error))
        return false;
      const StrictJson *name = JsonMember(entry, "name", StrictJson::Type::kString, error);
      const StrictJson *periodicity =
          JsonMember(entry, "periodicity", StrictJson::Type::kString, error);
      const StrictJson *center = JsonMember(entry, "center_mm", StrictJson::Type::kArray, error);
      const StrictJson *ri = JsonMember(entry, "ri_mm", StrictJson::Type::kNumber, error);
      const StrictJson *ro = JsonMember(entry, "ro_mm", StrictJson::Type::kNumber, error);
      const StrictJson *arc = JsonMember(entry, "arc_length_deg", StrictJson::Type::kNumber, error);
      const StrictJson *sectors =
          JsonMember(entry, "sector_count", StrictJson::Type::kNumber, error);
      const StrictJson *inner_shift =
          JsonMember(entry, "inner_shift", StrictJson::Type::kNumber, error);
      const StrictJson *outer_shift =
          JsonMember(entry, "outer_shift", StrictJson::Type::kNumber, error);
      const StrictJson *points = JsonMember(entry, "quad_points", StrictJson::Type::kArray, error);
      int32_t sector_count = 0;
      if (name == nullptr || name->string.empty() || periodicity == nullptr || center == nullptr ||
          ri == nullptr || ro == nullptr || arc == nullptr || sectors == nullptr ||
          inner_shift == nullptr || outer_shift == nullptr || points == nullptr ||
          center->array.size() != 2 || center->array[0].type != StrictJson::Type::kNumber ||
          center->array[1].type != StrictJson::Type::kNumber ||
          !JsonInteger(*sectors, &sector_count) || sector_count < 2 ||
          points->array.size() != static_cast<size_t>(sector_count + 1) ||
          (periodicity->string != "periodic" && periodicity->string != "antiperiodic")) {
        *error = "invalid sliding-band air-gap element";
        return false;
      }
      AirGapElement age;
      age.antiperiodic = periodicity->string == "antiperiodic";
      age.center_x_m = center->array[0].number * 1e-3;
      age.center_y_m = center->array[1].number * 1e-3;
      age.inner_radius_m = ri->number * 1e-3;
      age.outer_radius_m = ro->number * 1e-3;
      age.arc_length_deg = arc->number;
      age.inner_shift = inner_shift->number;
      age.outer_shift = outer_shift->number;
      for (const StrictJson &point : points->array) {
        if (!ExactObject(point, {"n0", "w0", "n1", "w1", "n2", "w2", "n3", "w3"}, error))
          return false;
        AirGapQuadPoint parsed_point;
        for (int local = 0; local < 4; ++local) {
          const std::string node_name = std::string("n") + char('0' + local);
          const std::string weight_name = std::string("w") + char('0' + local);
          const StrictJson *node =
              JsonMember(point, node_name.c_str(), StrictJson::Type::kNumber, error);
          const StrictJson *weight =
              JsonMember(point, weight_name.c_str(), StrictJson::Type::kNumber, error);
          if (node == nullptr || weight == nullptr ||
              !JsonInteger(*node, &parsed_point.node[local])) {
            *error = "invalid sliding-band quadrature point";
            return false;
          }
          parsed_point.weight[local] = weight->number;
        }
        age.quad_points.push_back(parsed_point);
      }
      parsed.model.air_gap_elements.push_back(std::move(age));
    }
    if (motor_v4 &&
        !std::all_of(parsed.model.air_gap_elements.begin(), parsed.model.air_gap_elements.end(),
                     [&](const AirGapElement &age) {
                       return age.antiperiodic == (parsed.sector_boundary_sign < 0) &&
                              age.arc_length_deg == parsed.sector_angle_deg;
                     })) {
      *error = "native AGE sector periodicity or arc length disagrees with sector metadata";
      return false;
    }
  }
  if (ValidateNonlinearModel(parsed.model) != Status::kOk) {
    *error = "mapped mesh model validation failed";
    return false;
  }
  *artifact = std::move(parsed);
  return true;
}

bool ReadGpuFemmMeshArtifact(const std::string &path, GpuFemmMeshArtifact *artifact,
                             std::string *error)
{
  std::ifstream input(path);
  std::stringstream contents;
  contents << input.rdbuf();
  if (!input) {
    *error = "cannot open mesh artifact";
    return false;
  }
  return ParseGpuFemmMeshArtifactJson(contents.str(), artifact, error);
}

} // namespace gpu_femm

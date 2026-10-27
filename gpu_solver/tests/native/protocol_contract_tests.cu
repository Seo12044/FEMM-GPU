#include "comparison.h"
#include "fixtures.h"
#include "suites.h"
#include "physics/nonlinear_model.h"
#include "postprocess/postprocess.h"
#include "solvers/nonlinear_solver.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "app/sample_adapter.h"
#include "app/motor_batch.h"
#include "io/mesh_artifact.h"
#include "io/sample_request.h"
#include "io/strict_json.h"
#include "physics/nonlinear_assembly.cuh"
#include "axisymmetric_p1.h"
#include "native_airgap_matrix.h"
#include "signed_node_constraints.h"

#include <fstream>
#include <sstream>
#include <utility>

namespace gpu_femm
{

int ProtocolContractTests()
{
  int failures = 0;
  auto expect = [&failures](bool condition, const std::string &message) {
    if (!condition) {
      std::cerr << "FAIL: " << message << '\n';
      ++failures;
    }
  };

  const std::string mesh_artifact = R"json({
    "schema_version":"gpu_femm_mesh_v1",
    "base_motor_fem_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "source_fem_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "canonical_identity_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "resolved":{
      "source_fem_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "base_motor_fem_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "model":{"depth_mm":20,"problem_type":"planar","frequency_hz":0},
      "pose":{"rotor_angle_deg":3,"displacement_mm":[0.1,0]},
      "nodes_mm":[[0,0],[1,0],[1,1],[0,1],[0.5,0.5]],
      "triangles":{"node_indices":[[0,1,4],[1,2,4],[2,3,4],[3,0,4]],"region_ids":[7,8,9,9]},
      "regions":[
        {"id":7,"group_number":20,"material_id":4,"circuit_index":-1,"turns":0,"pm_magnetization_deg":30},
        {"id":8,"group_number":50,"material_id":9,"circuit_index":0,"turns":50,"pm_magnetization_deg":0},
        {"id":9,"group_number":50,"material_id":9,"circuit_index":1,"turns":-50,"pm_magnetization_deg":0}],
      "materials":[
        {"id":4,"name":"PM","mu_x":1.05,"mu_y":1.05,"H_c_A_per_m":900000,"B_T":[],"H_A_per_m":[],"lam_type":0,"lam_fill":1},
        {"id":9,"name":"Air","mu_x":1,"mu_y":1,"H_c_A_per_m":0,"B_T":[],"H_A_per_m":[],"lam_type":0,"lam_fill":1}],
      "circuits":[{"index":0,"name":"coil_00","type":"series","current_A":2},{"index":1,"name":"coil_01","type":"series","current_A":-2}],
      "outer_dirichlet":{"node_indices":[0,1,2,3],"A_Wb_per_m":[0,0,0,0]}
    }
  })json";
  GpuFemmMeshArtifact parsed_artifact;
  std::string parser_error;
  expect(ParseGpuFemmMeshArtifactJson(mesh_artifact, &parsed_artifact, &parser_error),
         std::string("gpu_femm_mesh_v1 parser: ") + parser_error);
  expect(parsed_artifact.model.circuit_count == 2 &&
             parsed_artifact.circuit_currents_a.size() == 2 &&
             parsed_artifact.triangle_group_numbers.size() == 4 &&
             parsed_artifact.base_motor_fem_sha256.size() == 64 &&
             parsed_artifact.pose_fem_sha256.size() == 64,
         "gpu_femm_mesh_v1 maps multi-circuit labels and identities");
  const auto replace_once = [](std::string *value, const std::string &before,
                               const std::string &after) {
    const size_t at = value->find(before);
    if (at == std::string::npos)
      return false;
    value->replace(at, before.size(), after);
    return true;
  };
  std::string sector_mesh_artifact = mesh_artifact;
  const std::string sector_boundary_suffix =
      "\"outer_dirichlet\":{\"node_indices\":[1,2,3],\"A_Wb_per_m\":[0,0,0]},"
      "\"air_gap_elements\":[],"
      "\"node_constraints\":[{\"node_a\":1,\"node_b\":2,\"relation\":\"antiperiodic\"}],"
      "\"sector\":{\"schema_version\":\"gpu_femm_motor_sector_v1\","
      "\"sector_start_angle_deg\":15,\"angle_deg\":180,"
      "\"full_machine_sector_count\":2,"
      "\"electrical_pole_pairs\":3,\"suspension_spatial_order\":4,"
      "\"boundary_relation\":\"antiperiodic\","
      "\"circuit_partner_indices\":[1,0],"
      "\"circuit_partner_orientations\":[1,1],"
      "\"apex_self_pair_nodes\":[0],"
      "\"model_invariance_certificate_sha256\":"
      "\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\"}";
  const bool sector_document_built =
      replace_once(&sector_mesh_artifact, "gpu_femm_mesh_v1", "gpu_femm_mesh_v3") &&
      replace_once(&sector_mesh_artifact, "\"nodes_mm\":[[0,0],[1,0],[1,1],[0,1],[0.5,0.5]]",
                   "\"nodes_mm\":[[0,0],[1,0],[-1,0],[0,1],[0,0.5]]") &&
      replace_once(&sector_mesh_artifact, "\"node_indices\":[[0,1,4],[1,2,4],[2,3,4],[3,0,4]]",
                   "\"node_indices\":[[0,1,4],[1,3,4],[3,2,4],[2,0,4]]") &&
      replace_once(&sector_mesh_artifact,
                   "\"outer_dirichlet\":{\"node_indices\":[0,1,2,3],\"A_Wb_per_m\":[0,0,0,0]}",
                   sector_boundary_suffix);
  GpuFemmMeshArtifact sector_artifact;
  parser_error.clear();
  expect(sector_document_built &&
             ParseGpuFemmMeshArtifactJson(sector_mesh_artifact, &sector_artifact, &parser_error) &&
             sector_artifact.has_motor_sector && !sector_artifact.has_sliding_band &&
             sector_artifact.sector_start_angle_deg == 15.0 &&
             sector_artifact.electrical_pole_pairs == 3 &&
             sector_artifact.suspension_spatial_order == 4 &&
             sector_artifact.sector_boundary_sign == -1,
         std::string("strict generic 180-degree motor sector artifact parser: ") + parser_error);
  std::string rejected_missing_sector_start = sector_mesh_artifact;
  replace_once(&rejected_missing_sector_start, "\"sector_start_angle_deg\":15,", "");
  GpuFemmMeshArtifact invalid_missing_sector_start;
  parser_error.clear();
  expect(!ParseGpuFemmMeshArtifactJson(rejected_missing_sector_start, &invalid_missing_sector_start,
                                       &parser_error),
         "v3 strictly rejects a sector artifact without its SHA-bound start angle");
  NonlinearModel sector_tangent_model;
  expect(
      MakeMotorSectorTangentModel(sector_artifact, sector_artifact.model, &sector_tangent_model) &&
          !sector_tangent_model.node_constraints.empty() &&
          sector_tangent_model.node_constraints.front().sign == 1 &&
          std::find(sector_artifact.model.dirichlet_nodes.begin(),
                    sector_artifact.model.dirichlet_nodes.end(),
                    0) != sector_artifact.model.dirichlet_nodes.end() &&
          std::find(sector_tangent_model.dirichlet_nodes.begin(),
                    sector_tangent_model.dirichlet_nodes.end(),
                    0) == sector_tangent_model.dirichlet_nodes.end(),
      "p=3 AP base derives a separate p+1=4 periodic tangent reduction");
  std::string even_p_sector_artifact = sector_mesh_artifact;
  replace_once(&even_p_sector_artifact,
               "\"electrical_pole_pairs\":3,\"suspension_spatial_order\":4",
               "\"electrical_pole_pairs\":2,\"suspension_spatial_order\":3");
  for (size_t at = even_p_sector_artifact.find("antiperiodic"); at != std::string::npos;
       at = even_p_sector_artifact.find("antiperiodic", at + 8))
    even_p_sector_artifact.replace(at, 12, "periodic");
  GpuFemmMeshArtifact even_p_sector;
  parser_error.clear();
  NonlinearModel even_p_tangent_model;
  expect(
      ParseGpuFemmMeshArtifactJson(even_p_sector_artifact, &even_p_sector, &parser_error) &&
          even_p_sector.electrical_pole_pairs == 2 && even_p_sector.sector_boundary_sign == 1 &&
          MakeMotorSectorTangentModel(even_p_sector, even_p_sector.model, &even_p_tangent_model) &&
          even_p_tangent_model.node_constraints.front().sign == -1 &&
          std::find(even_p_sector.model.dirichlet_nodes.begin(),
                    even_p_sector.model.dirichlet_nodes.end(),
                    0) == even_p_sector.model.dirichlet_nodes.end() &&
          std::find(even_p_tangent_model.dirichlet_nodes.begin(),
                    even_p_tangent_model.dirichlet_nodes.end(),
                    0) != even_p_tangent_model.dirichlet_nodes.end(),
      "generic p=2 periodic base derives p+1=3 antiperiodic tangent reduction");
  std::string wrong_p3_base_sign = sector_mesh_artifact;
  for (size_t at = wrong_p3_base_sign.find("antiperiodic"); at != std::string::npos;
       at = wrong_p3_base_sign.find("antiperiodic", at + 8))
    wrong_p3_base_sign.replace(at, 12, "periodic");
  GpuFemmMeshArtifact rejected_wrong_p3;
  parser_error.clear();
  expect(!ParseGpuFemmMeshArtifactJson(wrong_p3_base_sign, &rejected_wrong_p3, &parser_error),
         "p=3 sector rejects periodic base relation even though p+1 tangent is periodic");
  std::string wrong_p4_base_sign = sector_mesh_artifact;
  replace_once(&wrong_p4_base_sign, "\"electrical_pole_pairs\":3,\"suspension_spatial_order\":4",
               "\"electrical_pole_pairs\":4,\"suspension_spatial_order\":5");
  GpuFemmMeshArtifact rejected_wrong_p4;
  parser_error.clear();
  expect(!ParseGpuFemmMeshArtifactJson(wrong_p4_base_sign, &rejected_wrong_p4, &parser_error),
         "p=4 sector rejects antiperiodic base relation even though p+1 tangent is antiperiodic");
  std::string rejected_sector_age = sector_mesh_artifact;
  replace_once(&rejected_sector_age, "\"air_gap_elements\":[]", "\"air_gap_elements\":{}");
  GpuFemmMeshArtifact invalid_sector_age;
  parser_error.clear();
  expect(!ParseGpuFemmMeshArtifactJson(rejected_sector_age, &invalid_sector_age, &parser_error),
         "v3 rejects unvalidated native AGE sectors and requires posed conforming mesh");
  std::string rejected_nonorigin_apex = sector_mesh_artifact;
  replace_once(&rejected_nonorigin_apex, "\"apex_self_pair_nodes\":[0]",
               "\"apex_self_pair_nodes\":[4]");
  GpuFemmMeshArtifact invalid_nonorigin_apex;
  parser_error.clear();
  expect(!ParseGpuFemmMeshArtifactJson(rejected_nonorigin_apex, &invalid_nonorigin_apex,
                                       &parser_error),
         "v3 rejects arbitrary non-origin nodes declared as rotational apex self-pairs");
  std::string rejected_wrong_radius_pair = sector_mesh_artifact;
  replace_once(&rejected_wrong_radius_pair, "\"node_a\":1,\"node_b\":2",
               "\"node_a\":1,\"node_b\":4");
  GpuFemmMeshArtifact invalid_wrong_radius_pair;
  parser_error.clear();
  expect(!ParseGpuFemmMeshArtifactJson(rejected_wrong_radius_pair, &invalid_wrong_radius_pair,
                                       &parser_error),
         "v3 rejects non-self PBC nodes that are not exact 180-degree geometric partners");
  // Nonlinear central-difference oracle for the production tangent path.
  // The h/2 estimate must approach the analytic consistent-Jacobian result
  // faster than h, while the reported PCG residual is J*dA-S itself.
  NonlinearModel tangent_oracle_model = parsed_artifact.model;
  for (NonlinearMaterial &material : tangent_oracle_model.materials)
    material.alpha_per_t2 = 0.3;
  NonlinearP1Solver tangent_oracle_solver;
  const NonlinearOptions tangent_oracle_options = ProductionSampleOptions();
  const Status tangent_oracle_initialize = tangent_oracle_solver.Initialize(tangent_oracle_model);
  const std::vector<double> tangent_base_currents = {2.0, -2.0};
  const NonlinearSolveResult tangent_base =
      tangent_oracle_initialize == Status::kOk
          ? tangent_oracle_solver.Solve(tangent_base_currents, tangent_oracle_options)
          : NonlinearSolveResult{};
  const std::vector<TangentSolveResult> tangent_oracle =
      tangent_base.info.status == Status::kOk
          ? tangent_oracle_solver.SolveTangentMultiRhs(
                tangent_base, {{"d", {1.0, 0.0}}, {"q", {0.0, 1.0}}}, tangent_oracle_options)
          : std::vector<TangentSolveResult>{};
  const auto centered_derivative = [&](double step) {
    std::vector<double> plus_currents = tangent_base_currents;
    std::vector<double> minus_currents = tangent_base_currents;
    plus_currents[0] += step;
    minus_currents[0] -= step;
    const NonlinearSolveResult plus = tangent_oracle_solver.Solve(
        plus_currents, tangent_oracle_options, &tangent_base.a_wb_per_m);
    const NonlinearSolveResult minus = tangent_oracle_solver.Solve(
        minus_currents, tangent_oracle_options, &tangent_base.a_wb_per_m);
    std::vector<double> derivative;
    if (plus.info.status == Status::kOk && minus.info.status == Status::kOk) {
      derivative.resize(plus.a_wb_per_m.size());
      for (size_t node = 0; node < derivative.size(); ++node)
        derivative[node] = (plus.a_wb_per_m[node] - minus.a_wb_per_m[node]) / (2.0 * step);
    }
    return derivative;
  };
  const std::vector<double> tangent_cd_h = centered_derivative(0.2);
  const std::vector<double> tangent_cd_h2 = centered_derivative(0.1);
  const auto derivative_error = [](const std::vector<double> &actual,
                                   const std::vector<double> &expected) {
    double difference_sq = 0.0, expected_sq = 0.0;
    if (actual.size() != expected.size() || actual.empty())
      return std::numeric_limits<double>::infinity();
    for (size_t index = 0; index < actual.size(); ++index) {
      const double difference = actual[index] - expected[index];
      difference_sq += difference * difference;
      expected_sq += expected[index] * expected[index];
    }
    return std::sqrt(difference_sq) /
           std::max(std::sqrt(expected_sq), std::numeric_limits<double>::min());
  };
  const double tangent_error_h =
      tangent_oracle.empty()
          ? std::numeric_limits<double>::infinity()
          : derivative_error(tangent_cd_h, tangent_oracle[0].field_derivative.a_wb_per_m);
  const double tangent_error_h2 =
      tangent_oracle.empty()
          ? std::numeric_limits<double>::infinity()
          : derivative_error(tangent_cd_h2, tangent_oracle[0].field_derivative.a_wb_per_m);
  expect(tangent_oracle.size() == 2 && tangent_oracle[0].info.status == Status::kOk &&
             tangent_oracle[1].info.status == Status::kOk &&
             tangent_oracle[0].info.residual_l2 < 1e-10 && tangent_error_h2 < tangent_error_h &&
             tangent_error_h2 < 5e-4,
         std::string("consistent tangent multi-RHS matches nonlinear h/h2 oracle (h=") +
             std::to_string(tangent_error_h) + ", h2=" + std::to_string(tangent_error_h2) +
             ", residual=" +
             (tangent_oracle.empty() ? "missing"
                                     : std::to_string(tangent_oracle[0].info.residual_l2)) +
             ")");
  FrozenPostprocessOptions tangent_force_options;
  tangent_force_options.selected_group_number = 20;
  tangent_force_options.air_group_number = 50;
  NonlinearModel tangent_force_model = parsed_artifact.model;
  tangent_force_model.dirichlet_nodes = {3};
  tangent_force_model.dirichlet_a_wb_per_m = {0.0};
  NonlinearP1Solver tangent_force_solver;
  const Status tangent_force_initialize = tangent_force_solver.Initialize(tangent_force_model);
  const NonlinearSolveResult tangent_force_base =
      tangent_force_initialize == Status::kOk
          ? tangent_force_solver.Solve(tangent_base_currents, tangent_oracle_options)
          : NonlinearSolveResult{};
  const std::vector<TangentSolveResult> tangent_force_solutions =
      tangent_force_base.info.status == Status::kOk
          ? tangent_force_solver.SolveTangentMultiRhs(tangent_force_base, {{"d", {1.0, 0.0}}},
                                                      tangent_oracle_options)
          : std::vector<TangentSolveResult>{};
  std::vector<double> tangent_force_mask;
  const Status tangent_force_mask_status =
      BuildWeightedStressMask(tangent_force_model, tangent_force_options, &tangent_force_mask);
  TangentPostprocessResult tangent_force_direct;
  FrozenPostprocessResult tangent_force_plus, tangent_force_minus;
  if (tangent_force_mask_status == Status::kOk && !tangent_force_solutions.empty()) {
    tangent_force_direct = ComputeFrozenTangentPostprocess(
        tangent_force_model, tangent_force_base, tangent_force_solutions[0].field_derivative,
        tangent_force_options, tangent_force_mask);
    NonlinearSolveResult plus = tangent_force_base, minus = tangent_force_base;
    for (size_t node = 0; node < plus.a_wb_per_m.size(); ++node) {
      plus.a_wb_per_m[node] += tangent_force_solutions[0].field_derivative.a_wb_per_m[node];
      minus.a_wb_per_m[node] -= tangent_force_solutions[0].field_derivative.a_wb_per_m[node];
    }
    for (size_t element = 0; element < plus.bx_t.size(); ++element) {
      plus.bx_t[element] += tangent_force_solutions[0].field_derivative.bx_t[element];
      plus.by_t[element] += tangent_force_solutions[0].field_derivative.by_t[element];
      minus.bx_t[element] -= tangent_force_solutions[0].field_derivative.bx_t[element];
      minus.by_t[element] -= tangent_force_solutions[0].field_derivative.by_t[element];
    }
    tangent_force_plus = ComputeFrozenPostprocess(tangent_force_model, plus, tangent_force_options);
    tangent_force_minus =
        ComputeFrozenPostprocess(tangent_force_model, minus, tangent_force_options);
  }
  expect(
      tangent_force_initialize == Status::kOk && tangent_force_base.info.status == Status::kOk &&
          tangent_force_solutions.size() == 1 &&
          tangent_force_solutions[0].info.status == Status::kOk &&
          tangent_force_mask_status == Status::kOk && tangent_force_direct.status == Status::kOk &&
          tangent_force_plus.status == Status::kOk && tangent_force_minus.status == Status::kOk &&
          NearMixed(tangent_force_direct.force_x_n_per_a,
                    0.5 * (tangent_force_plus.force_x_n - tangent_force_minus.force_x_n), 1e-12,
                    1e-12) &&
          NearMixed(tangent_force_direct.force_y_n_per_a,
                    0.5 * (tangent_force_plus.force_y_n - tangent_force_minus.force_y_n), 1e-12,
                    1e-12) &&
          NearMixed(tangent_force_direct.torque_nm_per_a,
                    0.5 * (tangent_force_plus.torque_nm - tangent_force_minus.torque_nm), 1e-12,
                    1e-12),
      std::string(
          "direct bilinear weighted-stress derivative matches polarization oracle (direct=") +
          std::to_string(tangent_force_direct.force_x_n_per_a) + "," +
          std::to_string(tangent_force_direct.force_y_n_per_a) + "," +
          std::to_string(tangent_force_direct.torque_nm_per_a) + "; polarized=" +
          std::to_string(0.5 * (tangent_force_plus.force_x_n - tangent_force_minus.force_x_n)) +
          "," +
          std::to_string(0.5 * (tangent_force_plus.force_y_n - tangent_force_minus.force_y_n)) +
          "," +
          std::to_string(0.5 * (tangent_force_plus.torque_nm - tangent_force_minus.torque_nm)) +
          ")");
  std::string neutral_mesh_artifact = mesh_artifact;
  neutral_mesh_artifact.replace(neutral_mesh_artifact.find("gpu_femm_mesh_v1"), 16,
                                "gpu_femm_planar_dc_mesh_v1");
  const std::string motor_base_field =
      "\"base_motor_fem_sha256\":"
      "\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\",";
  for (size_t found = neutral_mesh_artifact.find(motor_base_field); found != std::string::npos;
       found = neutral_mesh_artifact.find(motor_base_field))
    neutral_mesh_artifact.erase(found, motor_base_field.size());
  const std::string motor_pose_field =
      "\"pose\":{\"rotor_angle_deg\":3,\"displacement_mm\":[0.1,0]},";
  const size_t motor_pose_at = neutral_mesh_artifact.find(motor_pose_field);
  if (motor_pose_at != std::string::npos)
    neutral_mesh_artifact.erase(motor_pose_at, motor_pose_field.size());
  const auto rebind_neutral_identity = [](std::string *document) {
    StrictJson root;
    std::string error;
    StrictJsonParser parser(*document);
    if (!parser.Parse(&root, &error))
      return false;
    const StrictJson *identity =
        JsonMember(root, "canonical_identity_sha256", StrictJson::Type::kString, &error);
    const StrictJson *resolved = JsonMember(root, "resolved", StrictJson::Type::kObject, &error);
    if (identity == nullptr || resolved == nullptr)
      return false;
    const std::string replacement = CanonicalJsonSha256(*resolved);
    const size_t position = document->find(identity->string);
    if (position == std::string::npos)
      return false;
    document->replace(position, identity->string.size(), replacement);
    return true;
  };
  expect(rebind_neutral_identity(&neutral_mesh_artifact),
         "self-test neutral artifact identity is rebound");
  GpuFemmMeshArtifact neutral_artifact;
  expect(ParseGpuFemmMeshArtifactJson(neutral_mesh_artifact, &neutral_artifact, &parser_error) &&
             neutral_artifact.model.circuit_count == 2 &&
             neutral_artifact.base_motor_fem_sha256 == neutral_artifact.pose_fem_sha256,
         std::string("project-neutral mesh artifact parser: ") + parser_error);
  std::string pm_only_artifact = neutral_mesh_artifact;
  const auto replace_all = [](std::string *value, const std::string &before,
                              const std::string &after) {
    for (size_t found = value->find(before); found != std::string::npos;
         found = value->find(before, found + after.size()))
      value->replace(found, before.size(), after);
  };
  replace_all(&pm_only_artifact, "\"circuit_index\":0,\"turns\":50",
              "\"circuit_index\":-1,\"turns\":0");
  replace_all(&pm_only_artifact, "\"circuit_index\":1,\"turns\":-50",
              "\"circuit_index\":-1,\"turns\":0");
  const std::string two_circuits =
      "\"circuits\":[{\"index\":0,\"name\":\"coil_00\",\"type\":\"series\",\"current_A\":2},{"
      "\"index\":1,\"name\":\"coil_01\",\"type\":\"series\",\"current_A\":-2}]";
  replace_all(&pm_only_artifact, two_circuits, "\"circuits\":[]");
  expect(rebind_neutral_identity(&pm_only_artifact),
         "self-test PM-only artifact identity is rebound");
  GpuFemmMeshArtifact pm_only_parsed;
  expect(ParseGpuFemmMeshArtifactJson(pm_only_artifact, &pm_only_parsed, &parser_error) &&
             pm_only_parsed.model.circuit_count == 0,
         "project-neutral artifact permits PM-only zero-circuit models");
  NonlinearP1Solver pm_only_solver;
  const Status pm_only_initialize = pm_only_solver.Initialize(pm_only_parsed.model);
  const NonlinearSolveResult pm_only_solution =
      pm_only_initialize == Status::kOk
          ? pm_only_solver.Solve(std::vector<double>{}, ProductionSampleOptions())
          : NonlinearSolveResult{};
  expect(pm_only_initialize == Status::kOk && pm_only_solution.info.status == Status::kOk &&
             pm_only_solution.circuit_currents_a.empty() &&
             pm_only_solution.circuit_flux_linkage_wb.empty(),
         std::string(
             "project-neutral PM-only model solves with an empty current vector (initialize=") +
             StatusName(pm_only_initialize) +
             ", solve=" + StatusName(pm_only_solution.info.status) + ")");
  double native_age_matrix[10][10] = {};
  expect(BuildNativeFemmAirGapMatrix(0.75, 1.0 / 0.75, 0.25, 0.75, native_age_matrix) &&
             native_age_matrix[0][0] > 0.0 && native_age_matrix[7][7] > 0.0 &&
             Near(native_age_matrix[1][8], native_age_matrix[8][1], 1e-14),
         "native FEMM AGE matrix is finite, positive-diagonal, and symmetric");
  expect(Near(kNativeFemmAgeToSiReluctivity * kMu0, 1.0, 1e-15),
         "native FEMM AGE stiffness is converted to SI free-space reluctivity");
  std::string sliding_mesh_artifact = mesh_artifact;
  sliding_mesh_artifact.replace(sliding_mesh_artifact.find("gpu_femm_mesh_v1"), 16,
                                "gpu_femm_mesh_v2");
  const std::string v1_pose = "\"pose\":{\"rotor_angle_deg\":3,\"displacement_mm\":[0.1,0]}";
  sliding_mesh_artifact.replace(sliding_mesh_artifact.find(v1_pose), v1_pose.size(),
                                "\"pose\":{\"rotor_angle_deg\":0,\"displacement_mm\":[0,0]}");
  const std::string v1_boundary =
      "\"outer_dirichlet\":{\"node_indices\":[0,1,2,3],\"A_Wb_per_m\":[0,0,0,0]}";
  const std::string v2_boundary =
      v1_boundary +
      ",\"air_gap_elements\":[{\"name\":\"gap\",\"periodicity\":\"periodic\",\"center_mm\":[0,0],"
      "\"ri_mm\":0.4,\"ro_mm\":0.6,\"arc_length_deg\":360,\"sector_count\":2,\"inner_shift\":0,"
      "\"outer_shift\":0,\"quad_points\":[{\"n0\":0,\"w0\":1,\"n1\":1,\"w1\":1,\"n2\":2,\"w2\":1,"
      "\"n3\":3,\"w3\":1},{\"n0\":1,\"w0\":1,\"n1\":2,\"w1\":1,\"n2\":3,\"w2\":1,\"n3\":0,\"w3\":1}"
      ",{\"n0\":2,\"w0\":1,\"n1\":3,\"w1\":1,\"n2\":0,\"w2\":1,\"n3\":1,\"w3\":1}]}]";
  sliding_mesh_artifact.replace(sliding_mesh_artifact.find(v1_boundary), v1_boundary.size(),
                                v2_boundary);
  GpuFemmMeshArtifact sliding_artifact;
  expect(ParseGpuFemmMeshArtifactJson(sliding_mesh_artifact, &sliding_artifact, &parser_error) &&
             sliding_artifact.has_sliding_band &&
             sliding_artifact.model.air_gap_elements.size() == 1,
         std::string("gpu_femm_mesh_v2 sliding-band parser: ") + parser_error);
  // Parser-only v4 fixture: it proves the new cyclic AGE contract accepts a
  // 90-degree P cut and rejects neither AGE data nor k>2 generator metadata.
  std::string native_age_sector_mesh = sector_mesh_artifact;
  replace_once(&native_age_sector_mesh, "gpu_femm_mesh_v3", "gpu_femm_mesh_v4");
  replace_once(&native_age_sector_mesh,
               "\"pose\":{\"rotor_angle_deg\":3,\"displacement_mm\":[0.1,0]}",
               "\"pose\":{\"rotor_angle_deg\":0,\"displacement_mm\":[0,0]}");
  replace_once(&native_age_sector_mesh, "gpu_femm_motor_sector_v1", "gpu_femm_motor_sector_v2");
  replace_once(&native_age_sector_mesh,
               "\"angle_deg\":180,\"full_machine_sector_count\":2,"
               "\"electrical_pole_pairs\":3,\"suspension_spatial_order\":4,\"boundary_relation\":"
               "\"antiperiodic\"",
               "\"angle_deg\":90,\"full_machine_sector_count\":4,"
               "\"electrical_pole_pairs\":4,\"suspension_spatial_order\":5,\"boundary_relation\":"
               "\"periodic\"");
  replace_once(&native_age_sector_mesh,
               "\"node_constraints\":[{\"node_a\":1,\"node_b\":2,\"relation\":\"antiperiodic\"}]",
               "\"node_constraints\":[{\"node_a\":1,\"node_b\":3,\"relation\":\"periodic\"}]");
  const std::string age_only = v2_boundary.substr(v1_boundary.size() + 1);
  replace_once(&native_age_sector_mesh, "\"air_gap_elements\":[]", age_only);
  replace_once(&native_age_sector_mesh, "\"arc_length_deg\":360", "\"arc_length_deg\":90");
  GpuFemmMeshArtifact native_age_sector_artifact;
  parser_error.clear();
  expect(ParseGpuFemmMeshArtifactJson(native_age_sector_mesh, &native_age_sector_artifact,
                                      &parser_error) &&
             native_age_sector_artifact.has_native_age_sector &&
             native_age_sector_artifact.has_sliding_band &&
             native_age_sector_artifact.full_machine_sector_count == 4 &&
             native_age_sector_artifact.sector_angle_deg == 90.0,
         std::string("gpu_femm_mesh_v4 accepts 90-degree native AGE sector: ") + parser_error);
  std::string scalar_native_age_sector_mesh = native_age_sector_mesh;
  replace_once(&scalar_native_age_sector_mesh, "\"air_gap_elements\":[{", "\"air_gap_elements\":{");
  replace_once(&scalar_native_age_sector_mesh, "}],\"node_constraints\"", "},\"node_constraints\"");
  GpuFemmMeshArtifact scalar_native_age_sector_artifact;
  parser_error.clear();
  expect(ParseGpuFemmMeshArtifactJson(scalar_native_age_sector_mesh,
                                      &scalar_native_age_sector_artifact, &parser_error) &&
             scalar_native_age_sector_artifact.has_native_age_sector &&
             scalar_native_age_sector_artifact.model.air_gap_elements.size() == 1,
         std::string("gpu_femm_mesh_v4 accepts MATLAB scalar native AGE spelling: ") +
             parser_error);
  std::string invalid_native_age_rotation = native_age_sector_mesh;
  replace_once(&invalid_native_age_rotation, "\"node_b\":3", "\"node_b\":2");
  GpuFemmMeshArtifact rejected_native_age_rotation;
  parser_error.clear();
  expect(!ParseGpuFemmMeshArtifactJson(invalid_native_age_rotation, &rejected_native_age_rotation,
                                       &parser_error),
         "gpu_femm_mesh_v4 rejects a non-rotational sector cut pair");
  std::string reverse_canonical_native_age_rotation = native_age_sector_mesh;
  replace_once(&reverse_canonical_native_age_rotation, "\"node_a\":1,\"node_b\":3",
               "\"node_a\":3,\"node_b\":1");
  GpuFemmMeshArtifact reverse_canonical_native_age_artifact;
  parser_error.clear();
  expect(ParseGpuFemmMeshArtifactJson(reverse_canonical_native_age_rotation,
                                      &reverse_canonical_native_age_artifact, &parser_error),
         std::string("gpu_femm_mesh_v4 accepts reverse-directed canonical 90-degree pair: ") +
             parser_error);
  std::string scalar_sliding_mesh_artifact = sliding_mesh_artifact;
  const std::string age_array_open = "\"air_gap_elements\":[{";
  const size_t age_array_open_at = scalar_sliding_mesh_artifact.find(age_array_open);
  if (age_array_open_at != std::string::npos)
    scalar_sliding_mesh_artifact.replace(age_array_open_at, age_array_open.size(),
                                         "\"air_gap_elements\":{");
  const size_t age_array_close_at = scalar_sliding_mesh_artifact.rfind("}]\n    }");
  if (age_array_close_at != std::string::npos)
    scalar_sliding_mesh_artifact.erase(age_array_close_at + 1, 1);
  GpuFemmMeshArtifact scalar_sliding_artifact;
  expect(ParseGpuFemmMeshArtifactJson(scalar_sliding_mesh_artifact, &scalar_sliding_artifact,
                                      &parser_error) &&
             scalar_sliding_artifact.has_sliding_band &&
             scalar_sliding_artifact.model.air_gap_elements.size() == 1,
         "gpu_femm_mesh_v2 parser accepts MATLAB scalar AGE object spelling");
  NonlinearSolveResult zero_age_solution;
  zero_age_solution.a_wb_per_m.assign(sliding_artifact.model.nodes.size(), 0.0);
  FrozenPostprocessOptions zero_age_options;
  zero_age_options.airgap_angles_rad = {0.0};
  const FrozenPostprocessResult zero_age_postprocess =
      ComputeAirGapElementPostprocess(sliding_artifact.model, zero_age_solution, zero_age_options);
  expect(zero_age_postprocess.status == Status::kOk && zero_age_postprocess.force_x_n == 0.0 &&
             zero_age_postprocess.force_y_n == 0.0 && zero_age_postprocess.torque_nm == 0.0 &&
             zero_age_postprocess.airgap_samples.size() == 1 &&
             zero_age_postprocess.airgap_samples[0].radial_b_t == 0.0,
         "sliding AGE postprocess initializes finite zero-field outputs");
  expect(Sha256Hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "dependency-free SHA-256 matches known vector");
  const std::string motor_request_json = R"json({
    "protocol":"gpu_femm_motor_sample_v1",
    "mesh_artifact_path":"fixture.json",
    "mesh_artifact_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    "base_motor_fem_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "source_fem_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "circuit_currents_A":[2,-2],"selected_group_number":20,"air_group_number":50,
    "airgap_radius_mm":0,"airgap_angles_deg":[],"rotor_angle_deg":3,"displacement_mm":[0.1,0]
  })json";
  MotorSampleRequest parsed_request;
  expect(ReadMotorSampleRequestJson(motor_request_json, &parsed_request, &parser_error) &&
             RequestMatchesArtifact(parsed_request, parsed_artifact),
         "motor request binds artifact identities and pose");
  const std::string tangent_request_json = R"json({
    "protocol":"gpu_femm_motor_sample_v2",
    "mesh_artifact_path":"fixture.json",
    "mesh_artifact_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    "base_motor_fem_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "source_fem_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "circuit_currents_A":[2,-2],"selected_group_number":20,"air_group_number":50,
    "airgap_radius_mm":0,"airgap_angles_deg":[],"rotor_angle_deg":3,
    "displacement_mm":[0.1,0],"tangent_multi_rhs":{
      "schema_version":"gpu_femm_tangent_multi_rhs_v1","rhs":[
        {"name":"d","circuit_current_derivative_A_per_A":[1,0]},
        {"name":"q","circuit_current_derivative_A_per_A":[0,1]}]}
  })json";
  MotorSampleRequest parsed_tangent_request;
  expect(ReadMotorSampleRequestJson(tangent_request_json, &parsed_tangent_request, &parser_error) &&
             parsed_tangent_request.protocol == "gpu_femm_motor_sample_v2" &&
             parsed_tangent_request.tangent_requested &&
             parsed_tangent_request.tangent_multi_rhs.rhs.size() == 2,
         "motor sample v2 parses strict named tangent multi-RHS vectors");
  const std::string sector_request_json = R"json({
    "protocol":"gpu_femm_motor_sample_v3",
    "mesh_artifact_path":"fixture.json",
    "mesh_artifact_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    "base_motor_fem_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "source_fem_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "circuit_currents_A":[2,-2],"selected_group_number":20,"air_group_number":50,
    "airgap_radius_mm":0,"airgap_angles_deg":[],"rotor_angle_deg":3,
    "displacement_mm":[0.1,0],"tangent_multi_rhs":{
      "schema_version":"gpu_femm_tangent_multi_rhs_v1","rhs":[
        {"name":"d","circuit_current_derivative_A_per_A":[1,0]},
        {"name":"q","circuit_current_derivative_A_per_A":[0,1]}]},
    "sector_performance":{"schema_version":"gpu_femm_motor_sector_performance_v1",
      "model_invariance_certificate_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}
  })json";
  MotorSampleRequest parsed_sector_request;
  parser_error.clear();
  expect(ReadMotorSampleRequestJson(sector_request_json, &parsed_sector_request, &parser_error) &&
             parsed_sector_request.protocol == "gpu_femm_motor_sample_v3" &&
             parsed_sector_request.sector_performance.enabled &&
             RequestMatchesArtifact(parsed_sector_request, sector_artifact),
         std::string("strict motor sample v3 binds cropped sector metadata: ") + parser_error);
  std::string sector_base_only_request_json = sector_request_json;
  const size_t tangent_member_at = sector_base_only_request_json.find("\"tangent_multi_rhs\":{");
  const size_t sector_member_at =
      sector_base_only_request_json.find(",\n    \"sector_performance\"", tangent_member_at);
  if (tangent_member_at != std::string::npos && sector_member_at != std::string::npos)
    sector_base_only_request_json.replace(tangent_member_at, sector_member_at - tangent_member_at,
                                          "\"tangent_multi_rhs\":[]");
  MotorSampleRequest parsed_sector_base_only;
  parser_error.clear();
  expect(ReadMotorSampleRequestJson(sector_base_only_request_json, &parsed_sector_base_only,
                                    &parser_error) &&
             parsed_sector_base_only.protocol == "gpu_femm_motor_sample_v3" &&
             parsed_sector_base_only.sector_performance.enabled &&
             !parsed_sector_base_only.tangent_requested &&
             RequestMatchesArtifact(parsed_sector_base_only, sector_artifact),
         "motor sample v3 accepts explicit base-only zero/reference item");
  std::string native_age_sector_request_json = sector_base_only_request_json;
  replace_once(&native_age_sector_request_json, "gpu_femm_motor_sample_v3",
               "gpu_femm_motor_sample_v4");
  replace_once(&native_age_sector_request_json, "gpu_femm_motor_sector_performance_v1",
               "gpu_femm_motor_sector_performance_v2");
  replace_once(&native_age_sector_request_json, "\"rotor_angle_deg\":3", "\"rotor_angle_deg\":0");
  replace_once(&native_age_sector_request_json, "\"displacement_mm\":[0.1,0]",
               "\"displacement_mm\":[0,0]");
  MotorSampleRequest parsed_native_age_sector_request;
  parser_error.clear();
  expect(ReadMotorSampleRequestJson(native_age_sector_request_json,
                                    &parsed_native_age_sector_request, &parser_error) &&
             parsed_native_age_sector_request.protocol == "gpu_femm_motor_sample_v4" &&
             !parsed_native_age_sector_request.tangent_requested,
         std::string("motor sample v4 accepts base-only native-AGE sector item: ") + parser_error);
  MotorSampleRequest displaced_native_age_request = parsed_native_age_sector_request;
  displaced_native_age_request.displacement_mm[0] = 0.01;
  expect(RequestMatchesArtifact(parsed_native_age_sector_request, native_age_sector_artifact) &&
             !RequestMatchesArtifact(parsed_native_age_sector_request, sector_artifact) &&
             !RequestMatchesArtifact(displaced_native_age_request, native_age_sector_artifact),
         "v4 strictly binds its native-AGE mesh schema and rejects displacement");
  std::string native_age_tangent_request_json = sector_request_json;
  replace_once(&native_age_tangent_request_json, "gpu_femm_motor_sample_v3",
               "gpu_femm_motor_sample_v4");
  replace_once(&native_age_tangent_request_json, "gpu_femm_motor_sector_performance_v1",
               "gpu_femm_motor_sector_performance_v2");
  MotorSampleRequest rejected_native_age_tangent;
  parser_error.clear();
  expect(!ReadMotorSampleRequestJson(native_age_tangent_request_json, &rejected_native_age_tangent,
                                     &parser_error),
         "motor sample v4 rejects tangent/suspension work");
  GpuFemmMeshArtifact cyclic_flux_artifact = sector_artifact;
  cyclic_flux_artifact.has_native_age_sector = true;
  cyclic_flux_artifact.full_machine_sector_count = 4;
  cyclic_flux_artifact.sector_boundary_sign = -1;
  cyclic_flux_artifact.circuit_partner_indices = {1, 2, 3, 0};
  cyclic_flux_artifact.circuit_partner_orientations = {1, 1, 1, 1};
  const std::vector<double> cyclic_flux =
      ReconstructCyclicSectorCircuitFluxLinkage({1.0, 2.0, 3.0, 4.0}, cyclic_flux_artifact);
  expect(cyclic_flux == std::vector<double>({-2.0, 2.0, -2.0, 2.0}),
         "anti-periodic C4 circuit scatter-sum includes field sign on every copy");
  FrozenPostprocessOptions cyclic_angle_options;
  constexpr double kCyclicSelfTestPi = 3.141592653589793238462643383279502884;
  cyclic_angle_options.airgap_angles_rad = {0.0, 0.5 * kCyclicSelfTestPi, kCyclicSelfTestPi,
                                            1.5 * kCyclicSelfTestPi};
  cyclic_flux_artifact.sector_start_angle_deg = 0.0;
  cyclic_flux_artifact.sector_angle_deg = 90.0;
  cyclic_flux_artifact.sector_boundary_sign = -1;
  // Deliberately place q0 away from both global zero and the declared crop
  // cut.  writepoly stores this local node as the predecessor that brackets
  // global zero in its expanded/sorted virtual ring, so it must not shift the
  // native AGE Fourier probe phase.
  cyclic_flux_artifact.model.nodes = {
      {std::cos(30.0 * kCyclicSelfTestPi / 180.0), std::sin(30.0 * kCyclicSelfTestPi / 180.0)}};
  AirGapElement cyclic_anchor_age;
  cyclic_anchor_age.center_x_m = 0.0;
  cyclic_anchor_age.center_y_m = 0.0;
  AirGapQuadPoint cyclic_anchor_point;
  cyclic_anchor_point.node[0] = 0;
  cyclic_anchor_age.quad_points.push_back(cyclic_anchor_point);
  cyclic_flux_artifact.model.air_gap_elements = {cyclic_anchor_age};
  const std::vector<int8_t> cyclic_angle_signs =
      MapNativeAgeSectorAirgapAnglesToRepresentative(cyclic_flux_artifact, &cyclic_angle_options);
  expect(cyclic_angle_signs == std::vector<int8_t>({1, -1, 1, -1}) &&
             Near(cyclic_angle_options.airgap_angles_rad[0], 0.0, 1e-14) &&
             Near(cyclic_angle_options.airgap_angles_rad[1], 0.0, 1e-14) &&
             Near(cyclic_angle_options.airgap_angles_rad[2], 0.0, 1e-14) &&
             Near(cyclic_angle_options.airgap_angles_rad[3], 0.0, 1e-14),
         "native AGE sector preserves global-zero phase and AP copy parity");
  cyclic_flux_artifact.sector_start_angle_deg = 15.0;
  cyclic_angle_options.airgap_angles_rad = {0.0,
                                            15.0 * kCyclicSelfTestPi / 180.0,
                                            90.0 * kCyclicSelfTestPi / 180.0,
                                            105.0 * kCyclicSelfTestPi / 180.0,
                                            -90.0 * kCyclicSelfTestPi / 180.0,
                                            180.0 * kCyclicSelfTestPi / 180.0};
  const std::vector<int8_t> shifted_cyclic_angle_signs =
      MapNativeAgeSectorAirgapAnglesToRepresentative(cyclic_flux_artifact, &cyclic_angle_options);
  expect(shifted_cyclic_angle_signs == std::vector<int8_t>({1, 1, -1, -1, -1, 1}) &&
             Near(cyclic_angle_options.airgap_angles_rad[0], 0.0, 1e-14) &&
             Near(cyclic_angle_options.airgap_angles_rad[1], 15.0 * kCyclicSelfTestPi / 180.0,
                  1e-14) &&
             Near(cyclic_angle_options.airgap_angles_rad[2], 0.0, 1e-14) &&
             Near(cyclic_angle_options.airgap_angles_rad[3], 15.0 * kCyclicSelfTestPi / 180.0,
                  1e-14) &&
             Near(cyclic_angle_options.airgap_angles_rad[4], 0.0, 1e-14) &&
             Near(cyclic_angle_options.airgap_angles_rad[5], 0.0, 1e-14),
         "native AGE Br ignores crop/q0 offsets and uses global AP parity");
  const std::string mixed_sector_batch =
      std::string("{\"protocol\":\"gpu_femm_motor_batch_v3\","
                  "\"max_items_per_chunk\":2,\"items\":["
                  "{\"task_id\":\"zero\",\"request\":") +
      sector_base_only_request_json +
      "},{\"task_id\":\"drive\",\"request\":" + sector_request_json + "}]}";
  MotorBatchRequest parsed_mixed_sector_batch;
  parser_error.clear();
  expect(ReadMotorBatchRequestJson(mixed_sector_batch, &parsed_mixed_sector_batch, &parser_error) &&
             parsed_mixed_sector_batch.items.size() == 2 &&
             !parsed_mixed_sector_batch.items[0].request.tangent_requested &&
             parsed_mixed_sector_batch.items[1].request.tangent_requested,
         "motor batch v3 strictly accepts mixed base-only zero and tangent drive items");
  std::string v2_with_sector = sector_request_json;
  replace_once(&v2_with_sector, "gpu_femm_motor_sample_v3", "gpu_femm_motor_sample_v2");
  MotorSampleRequest rejected_v2_sector;
  parser_error.clear();
  expect(!ReadMotorSampleRequestJson(v2_with_sector, &rejected_v2_sector, &parser_error),
         "motor sample v2 remains strict and rejects sector extension fields");
  expect(ReconstructSectorCircuitFluxLinkage({2.0, 3.0}, sector_artifact, -1) ==
                 std::vector<double>({-1.0, 1.0}) &&
             ReconstructSectorCircuitFluxLinkage({2.0, 3.0}, sector_artifact, 1) ==
                 std::vector<double>({5.0, 5.0}),
         "sector circuit flux reconstruction applies partner permutation, orientation, and field "
         "sign");
  FrozenPostprocessOptions sector_angle_options;
  constexpr double kSelfTestPi = 3.141592653589793238462643383279502884;
  sector_angle_options.airgap_angles_rad = {
      0.0, 0.5 * kSelfTestPi, kSelfTestPi, 1.5 * kSelfTestPi, 2.0 * kSelfTestPi, -kSelfTestPi};
  const std::vector<int8_t> sector_angle_signs =
      MapSectorAirgapAnglesToRepresentative(0.0, -1, &sector_angle_options);
  expect(sector_angle_signs == std::vector<int8_t>({1, 1, -1, -1, 1, -1}) &&
             Near(sector_angle_options.airgap_angles_rad[0], 0.0) &&
             Near(sector_angle_options.airgap_angles_rad[1], 0.5 * kSelfTestPi) &&
             Near(sector_angle_options.airgap_angles_rad[2], 0.0) &&
             Near(sector_angle_options.airgap_angles_rad[3], 0.5 * kSelfTestPi) &&
             Near(sector_angle_options.airgap_angles_rad[4], 0.0) &&
             Near(sector_angle_options.airgap_angles_rad[5], 0.0),
         "full-circle Br requests fold into 180-degree sector with exact AP copy signs");
  FrozenPostprocessOptions shifted_sector_angle_options;
  const double shifted_sector_start = 15.0 * kSelfTestPi / 180.0;
  shifted_sector_angle_options.airgap_angles_rad = {
      0.0, 10.0 * kSelfTestPi / 180.0, 15.0 * kSelfTestPi / 180.0, 190.0 * kSelfTestPi / 180.0,
      195.0 * kSelfTestPi / 180.0};
  const std::vector<int8_t> shifted_sector_signs = MapSectorAirgapAnglesToRepresentative(
      shifted_sector_start, -1, &shifted_sector_angle_options);
  expect(shifted_sector_signs == std::vector<int8_t>({-1, -1, 1, 1, -1}) &&
             Near(shifted_sector_angle_options.airgap_angles_rad[0], kSelfTestPi) &&
             Near(shifted_sector_angle_options.airgap_angles_rad[1], 190.0 * kSelfTestPi / 180.0) &&
             Near(shifted_sector_angle_options.airgap_angles_rad[2], shifted_sector_start) &&
             Near(shifted_sector_angle_options.airgap_angles_rad[4], shifted_sector_start),
         "nonzero-start sector folds below-start angles into its actual posed interval");
  FrozenPostprocessOptions full_br_options;
  constexpr size_t kBrOracleSamples = 72;
  for (size_t sample = 0; sample < kBrOracleSamples; ++sample)
    full_br_options.airgap_angles_rad.push_back(2.0 * kSelfTestPi * static_cast<double>(sample) /
                                                static_cast<double>(kBrOracleSamples));
  const std::vector<double> global_br_angles = full_br_options.airgap_angles_rad;
  const std::vector<int8_t> full_br_signs =
      MapSectorAirgapAnglesToRepresentative(shifted_sector_start, -1, &full_br_options);
  std::vector<double> reconstructed_br, direct_full_br;
  for (size_t sample = 0; sample < kBrOracleSamples; ++sample) {
    const double sector_value = std::cos(3.0 * full_br_options.airgap_angles_rad[sample]) +
                                0.1 * std::cos(9.0 * full_br_options.airgap_angles_rad[sample]);
    reconstructed_br.push_back(full_br_signs[sample] * sector_value);
    direct_full_br.push_back(std::cos(3.0 * global_br_angles[sample]) +
                             0.1 * std::cos(9.0 * global_br_angles[sample]));
  }
  const auto sampled_thd = [&](const std::vector<double> &values, size_t fundamental) {
    std::vector<double> amplitudes(values.size() / 2, 0.0);
    for (size_t order = 1; order < amplitudes.size(); ++order) {
      double cosine = 0.0, sine = 0.0;
      for (size_t sample = 0; sample < values.size(); ++sample) {
        const double phase = 2.0 * kSelfTestPi * static_cast<double>(order * sample) /
                             static_cast<double>(values.size());
        cosine += values[sample] * std::cos(phase);
        sine += values[sample] * std::sin(phase);
      }
      amplitudes[order] = 2.0 * std::hypot(cosine, sine) / static_cast<double>(values.size());
    }
    double harmonic_sq = 0.0;
    for (size_t order = 1; order < amplitudes.size(); ++order)
      if (order != fundamental)
        harmonic_sq += amplitudes[order] * amplitudes[order];
    return std::sqrt(harmonic_sq) / amplitudes[fundamental];
  };
  bool full_br_matches = reconstructed_br.size() == direct_full_br.size();
  for (size_t sample = 0; full_br_matches && sample < reconstructed_br.size(); ++sample)
    full_br_matches = Near(reconstructed_br[sample], direct_full_br[sample], 1e-13);
  expect(full_br_matches &&
             Near(sampled_thd(reconstructed_br, 3), sampled_thd(direct_full_br, 3), 1e-13) &&
             Near(sampled_thd(reconstructed_br, 3), 0.1, 1e-13),
         "nonzero-start 180-degree AP sector reconstructs full-circle Br and spatial THD oracle "
         "exactly");
  MotorBatchResponseDto sector_reconstruction_oracle;
  sector_reconstruction_oracle.status = Status::kOk;
  sector_reconstruction_oracle.force_x_n = 7.0;
  sector_reconstruction_oracle.force_y_n = -4.0;
  sector_reconstruction_oracle.torque_nm = 3.0;
  sector_reconstruction_oracle.circuit_flux_linkage_wb = {2.0, 3.0};
  ApplyBaseSectorReconstruction(sector_artifact, parsed_sector_request,
                                &sector_reconstruction_oracle);
  TangentRhsResponseDto tangent_reconstruction_oracle;
  tangent_reconstruction_oracle.force_x_n_per_a = 5.0;
  tangent_reconstruction_oracle.force_y_n_per_a = -2.0;
  tangent_reconstruction_oracle.torque_nm_per_a = 11.0;
  tangent_reconstruction_oracle.circuit_flux_linkage_derivative_wb_per_a = {2.0, 3.0};
  const bool tangent_reconstruction_ok = ApplyTangentSectorReconstruction(
      sector_artifact, sector_reconstruction_oracle, &tangent_reconstruction_oracle);
  expect(
      sector_reconstruction_oracle.sector_start_angle_deg == 15.0 &&
          sector_reconstruction_oracle.force_x_n == 0.0 &&
          sector_reconstruction_oracle.force_y_n == 0.0 &&
          sector_reconstruction_oracle.torque_nm == 6.0 &&
          sector_reconstruction_oracle.circuit_flux_linkage_wb ==
              std::vector<double>({-1.0, 1.0}) &&
          tangent_reconstruction_ok && tangent_reconstruction_oracle.force_x_n_per_a == 10.0 &&
          tangent_reconstruction_oracle.force_y_n_per_a == -4.0 &&
          tangent_reconstruction_oracle.torque_nm_per_a == 0.0 &&
          tangent_reconstruction_oracle.circuit_flux_linkage_derivative_wb_per_a ==
              std::vector<double>({5.0, 5.0}),
      "C2 reconstruction cancels base force/tangent torque and doubles base torque/tangent force");
  std::string v1_with_tangent = motor_request_json;
  const size_t v1_close = v1_with_tangent.rfind('}');
  if (v1_close != std::string::npos)
    v1_with_tangent.insert(v1_close, ",\"tangent_multi_rhs\":null");
  MotorSampleRequest rejected_v1_extension;
  expect(!ReadMotorSampleRequestJson(v1_with_tangent, &rejected_v1_extension, &parser_error),
         "motor sample v1 remains strict and rejects tangent extension fields");
  const std::string planar_request_json = R"json({
    "protocol":"gpu_femm_planar_dc_sample_v1",
    "mesh_artifact_path":"fixture.json",
    "mesh_artifact_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    "circuit_currents_A":[2,-2],"force_group_number":-1,
    "stress_air_group_number":-1,"airgap_radius_mm":0,
    "airgap_angles_deg":[],"sliding_band_angle_deg":0,
    "compute_force_torque":false,"include_field_solution":true
  })json";
  PlanarDcSampleRequest planar_request;
  expect(ReadPlanarDcSampleRequestJson(planar_request_json, &planar_request, &parser_error) &&
             PlanarDcRequestMatchesArtifact(planar_request, parsed_artifact) &&
             planar_request.include_field_solution,
         "project-neutral planar-DC request validates without motor identity fields");
  SignedDofMap signed_map;
  expect(BuildSignedDofMap(5, {{0, 1, 1}, {1, 2, -1}}, {3}, {0.0}, &signed_map) ==
                 SignedDofStatus::kOk &&
             signed_map.node_root[0] == signed_map.node_root[2] && signed_map.node_sign[0] == 1 &&
             signed_map.node_sign[1] == 1 && signed_map.node_sign[2] == -1 &&
             signed_map.root_fixed[signed_map.node_root[3]],
         "signed DOF map composes periodic and anti-periodic relations");
  expect(BuildSignedDofMap(3, {{0, 1, 1}, {1, 2, 1}, {0, 2, -1}}, {}, {}, &signed_map) ==
             SignedDofStatus::kContradictoryCycle,
         "signed DOF map rejects contradictory periodic cycles");
  NonlinearModel signed_reduction_model;
  signed_reduction_model.nodes.resize(3);
  signed_reduction_model.node_constraints = {{1, 2, -1}};
  signed_reduction_model.dirichlet_nodes = {0};
  signed_reduction_model.dirichlet_a_wb_per_m = {0.0};
  SparseRows signed_full_matrix(3);
  signed_full_matrix[0][0] = 1.0;
  signed_full_matrix[1][1] = 2.0;
  signed_full_matrix[1][2] = 1.0;
  signed_full_matrix[2][1] = 1.0;
  signed_full_matrix[2][2] = 2.0;
  Assembly signed_reduction;
  const Status signed_reduction_status = ReduceSignedSystem(
      signed_reduction_model, signed_full_matrix, {0.0, 3.0, -3.0}, &signed_reduction);
  GpuCsrSolver signed_reduction_solver;
  std::vector<double> signed_free_solution;
  SolveInfo signed_reduction_info;
  if (signed_reduction_status == Status::kOk &&
      signed_reduction_solver.Initialize(signed_reduction) == Status::kOk)
    signed_reduction_info = signed_reduction_solver.Solve(
        {signed_reduction.rhs_per_amp[0] + signed_reduction.rhs_offset[0]}, 1e-14, 16,
        &signed_free_solution);
  const std::vector<double> signed_expanded =
      ExpandAssemblySolution(signed_reduction, signed_free_solution);
  expect(signed_reduction_status == Status::kOk && signed_reduction_info.status == Status::kOk &&
             signed_expanded.size() == 3 && Near(signed_expanded[1], 3.0, 1e-13) &&
             Near(signed_expanded[2], -3.0, 1e-13),
         "anti-periodic reduction solves T^T K T and reconstructs signed nodes");
  NonlinearModel cancellation_model;
  cancellation_model.nodes.resize(4);
  cancellation_model.node_constraints = {{1, 2, -1}};
  cancellation_model.dirichlet_nodes = {0};
  cancellation_model.dirichlet_a_wb_per_m = {0.0};
  SparseRows cancellation_matrix(4);
  cancellation_matrix[0][0] = 1.0;
  cancellation_matrix[1][1] = 2.0;
  cancellation_matrix[2][2] = 2.0;
  cancellation_matrix[3][3] = 3.0;
  cancellation_matrix[1][3] = 1.0;
  cancellation_matrix[2][3] = 1.0;
  cancellation_matrix[3][1] = 1.0;
  cancellation_matrix[3][2] = 1.0;
  Assembly cancellation_reduction;
  const Status cancellation_status = ReduceSignedSystem(
      cancellation_model, cancellation_matrix, {0.0, 0.0, 0.0, 0.0}, &cancellation_reduction);
  expect(cancellation_status == Status::kOk &&
             cancellation_reduction.row_offsets == std::vector<int32_t>({0, 2, 4}) &&
             cancellation_reduction.column_indices == std::vector<int32_t>({0, 1, 0, 1}) &&
             cancellation_reduction.values.size() == 4 && cancellation_reduction.values[1] == 0.0 &&
             cancellation_reduction.values[2] == 0.0,
         "signed reduction preserves exact-cancellation CSR slots for tangent symbolic reuse");
  const double manufactured_radius[3] = {1.0, 2.0, 1.0};
  const double manufactured_axial[3] = {0.0, 0.0, 1.0};
  const double manufactured_a[3] = {0.25, 0.5, 0.25};
  double manufactured_br = 0.0, manufactured_bz = 0.0;
  expect(EvaluateAxisymmetricP1Centroid(manufactured_radius, manufactured_axial, manufactured_a,
                                        &manufactured_br, &manufactured_bz) &&
             Near(manufactured_br, 0.0, 1e-14) && Near(manufactured_bz, 0.5, 1e-14),
         "axisymmetric P1 curl reproduces A_phi=c*r exactly");
  std::string generic_mesh_artifact = neutral_mesh_artifact;
  replace_all(&generic_mesh_artifact, "gpu_femm_planar_dc_mesh_v1",
              "gpu_femm_magnetostatic_mesh_v1");
  replace_all(&generic_mesh_artifact, v1_boundary,
              v1_boundary +
                  ",\"node_constraints\":[{\"node_a\":0,\"node_b\":1,\"relation\":\"periodic\"}]");
  expect(rebind_neutral_identity(&generic_mesh_artifact),
         "self-test generic artifact identity is rebound");
  StrictJson generic_document;
  StrictJsonParser generic_document_parser(generic_mesh_artifact);
  std::string generic_document_error;
  expect(generic_document_parser.Parse(&generic_document, &generic_document_error),
         "self-test rebound generic document parses");
  const StrictJson *generic_identity_value =
      JsonMember(generic_document, "canonical_identity_sha256", StrictJson::Type::kString,
                 &generic_document_error);
  const std::string generic_identity =
      generic_identity_value == nullptr ? std::string() : generic_identity_value->string;
  GpuFemmMeshArtifact generic_artifact;
  expect(ParseGpuFemmMeshArtifactJson(generic_mesh_artifact, &generic_artifact, &parser_error) &&
             generic_artifact.model.node_constraints.size() == 1 &&
             generic_artifact.model.node_constraints[0].sign == 1,
         std::string("generic periodic artifact parser: ") + parser_error);
  std::string corrupted_generic_artifact = generic_mesh_artifact;
  if (!generic_identity.empty()) {
    const size_t identity_at = corrupted_generic_artifact.find(generic_identity);
    if (identity_at != std::string::npos)
      corrupted_generic_artifact[identity_at] = generic_identity[0] == '0' ? '1' : '0';
  }
  GpuFemmMeshArtifact rejected_generic_artifact;
  expect(!ParseGpuFemmMeshArtifactJson(corrupted_generic_artifact, &rejected_generic_artifact,
                                       &parser_error),
         "generic artifact rejects a corrupted canonical identity");
  std::string generic_request_json = planar_request_json;
  replace_all(&generic_request_json, "gpu_femm_planar_dc_sample_v1",
              "gpu_femm_magnetostatic_sample_v1");
  PlanarDcSampleRequest generic_request;
  expect(ReadPlanarDcSampleRequestJson(generic_request_json, &generic_request, &parser_error) &&
             PlanarDcRequestMatchesArtifact(generic_request, generic_artifact),
         "generic request binds the periodic artifact schema");
  generic_request.compute_force_torque = true;
  expect(!PlanarDcRequestMatchesArtifact(generic_request, generic_artifact),
         "generic request rejects legacy force/air-gap postprocessing");
  generic_request.compute_force_torque = false;
  generic_request.force_group_number = 20;
  expect(!PlanarDcRequestMatchesArtifact(generic_request, generic_artifact),
         "generic request rejects dormant legacy group controls");
  generic_request.force_group_number = -1;
  generic_request.airgap_radius_mm = 1.0;
  expect(!PlanarDcRequestMatchesArtifact(generic_request, generic_artifact),
         "generic request rejects a dormant legacy air-gap radius");
  NonlinearModel invalid_axis_boundary = neutral_artifact.model;
  invalid_axis_boundary.problem_type = MagneticProblemType::kAxisymmetric;
  invalid_axis_boundary.dirichlet_a_wb_per_m[0] = 1.0;
  expect(ValidateNonlinearModel(invalid_axis_boundary) == Status::kBoundaryInvalid,
         "axisymmetric models require a zero potential on the symmetry axis");
  MotorSampleRequest sliding_request = parsed_request;
  sliding_request.rotor_angle_deg = 3.0;
  sliding_request.displacement_mm[0] = 0.0;
  expect(RequestMatchesArtifact(sliding_request, sliding_artifact),
         "v2 request permits a centered non-reference rotor angle");
  MotorSampleRequest invalid_sliding_postprocess = sliding_request;
  invalid_sliding_postprocess.airgap_angles_deg = {0.0};
  invalid_sliding_postprocess.airgap_radius_mm = 0.7;
  expect(!RequestMatchesArtifact(invalid_sliding_postprocess, sliding_artifact),
         "v2 request rejects sampling outside the native AGE annulus");
  invalid_sliding_postprocess = sliding_request;
  invalid_sliding_postprocess.selected_group_number = 999;
  expect(!RequestMatchesArtifact(invalid_sliding_postprocess, sliding_artifact),
         "v2 request rejects postprocess groups absent from the artifact");
  NonlinearModel shifted_sliding_model;
  expect(ApplySlidingBandRotorAngle(sliding_request, sliding_artifact, &shifted_sliding_model) &&
             Near(shifted_sliding_model.air_gap_elements[0].inner_shift, 3.0 / 180.0, 1e-14) &&
             shifted_sliding_model.air_gap_elements[0].quad_points[0].node[0] ==
                 sliding_artifact.model.air_gap_elements[0].quad_points[0].node[0],
         "v2 request applies FEMM's positive AGE shift and inner-ring remap for rotor angle");
  const auto device_assembly_matches_host = [](const NonlinearModel &model,
                                               const std::vector<double> &state,
                                               const std::vector<double> &currents,
                                               bool use_newton) {
    Assembly host;
    if (AssembleNonlinearNewton(model, state, currents, use_newton, &host) != Status::kOk)
      return false;
    DeviceNonlinearAssemblyPlan plan;
    GpuCsrSolver device_solver;
    if (plan.Initialize(model) != Status::kOk ||
        device_solver.Initialize(plan.symbolic_assembly()) != Status::kOk ||
        plan.Assemble({state}, {currents}, use_newton, &device_solver, nullptr) != Status::kOk)
      return false;
    const Assembly &symbolic = plan.symbolic_assembly();
    if (host.row_offsets != symbolic.row_offsets || host.column_indices != symbolic.column_indices)
      return false;
    std::vector<double> values, diagonal, rhs;
    if (device_solver.CopyPreparedDeviceAssembly(0, &values, &diagonal, &rhs) != Status::kOk ||
        values.size() != host.values.size() || diagonal.size() != host.diagonal.size() ||
        rhs.size() != host.rhs_per_amp.size())
      return false;
    for (size_t index = 0; index < values.size(); ++index)
      if (!NearMixed(values[index], host.values[index], 1e-12, 5e-11))
        return false;
    for (size_t row = 0; row < diagonal.size(); ++row) {
      if (!NearMixed(diagonal[row], host.diagonal[row], 1e-12, 5e-11) ||
          !NearMixed(rhs[row], host.rhs_per_amp[row] + host.rhs_offset[row], 1e-12, 5e-11))
        return false;
    }
    for (int repeat = 0; repeat < 10; ++repeat) {
      if (plan.Assemble({state}, {currents}, use_newton, &device_solver, nullptr) != Status::kOk)
        return false;
      std::vector<double> repeated_values, repeated_diagonal, repeated_rhs;
      if (device_solver.CopyPreparedDeviceAssembly(0, &repeated_values, &repeated_diagonal,
                                                   &repeated_rhs) != Status::kOk ||
          repeated_values.size() != values.size() || repeated_diagonal.size() != diagonal.size() ||
          repeated_rhs.size() != rhs.size() ||
          std::memcmp(repeated_values.data(), values.data(), values.size() * sizeof(double)) != 0 ||
          std::memcmp(repeated_diagonal.data(), diagonal.data(),
                      diagonal.size() * sizeof(double)) != 0 ||
          std::memcmp(repeated_rhs.data(), rhs.data(), rhs.size() * sizeof(double)) != 0)
        return false;
    }
    return true;
  };
  NonlinearModel bh_assembly_model = NonlinearThreeRegionFixture();
  std::vector<double> cold_state(bh_assembly_model.nodes.size(), 0.0);
  std::vector<double> newton_state = cold_state;
  newton_state.back() = 0.25;
  expect(device_assembly_matches_host(bh_assembly_model, cold_state, {2.0}, false) &&
             device_assembly_matches_host(bh_assembly_model, newton_state, {2.0}, true),
         "device nonlinear assembly matches host CSR, diagonal, and RHS for cold and BH Newton "
         "states");
  NonlinearModel signed_assembly_model = NonlinearThreeRegionFixture();
  signed_assembly_model.dirichlet_nodes = {0};
  signed_assembly_model.dirichlet_a_wb_per_m = {0.1};
  signed_assembly_model.node_constraints = {{0, 2, -1}, {1, 3, -1}};
  std::vector<double> signed_state(signed_assembly_model.nodes.size(), 0.0);
  signed_state[0] = 0.1;
  signed_state[2] = -0.1;
  signed_state[1] = 0.2;
  signed_state[3] = -0.2;
  signed_state[4] = 0.15;
  expect(device_assembly_matches_host(signed_assembly_model, signed_state, {2.0}, true),
         "device assembly applies signed periodic reduction and fixed-root expansion");
  NonlinearModel axisymmetric_assembly_model = NonlinearThreeRegionFixture();
  axisymmetric_assembly_model.problem_type = MagneticProblemType::kAxisymmetric;
  std::vector<double> axisymmetric_state(axisymmetric_assembly_model.nodes.size(), 0.0);
  axisymmetric_state.back() = 0.25;
  expect(device_assembly_matches_host(axisymmetric_assembly_model, axisymmetric_state, {2.0}, true),
         "device assembly matches host axisymmetric seven-point P1 Newton operator");
  NonlinearModel age_assembly_model = shifted_sliding_model;
  age_assembly_model.dirichlet_nodes = {0};
  age_assembly_model.dirichlet_a_wb_per_m = {0.25};
  std::vector<double> age_state(age_assembly_model.nodes.size(), 0.05);
  age_state[0] = 0.25;
  expect(device_assembly_matches_host(age_assembly_model, age_state, {2.0, -2.0}, true),
         "device nonlinear assembly matches host AGE and nonzero-Dirichlet elimination");
  MotorSampleRequest cell_shift_request = sliding_request;
  cell_shift_request.rotor_angle_deg = 180.0;
  NonlinearModel cell_shift_model;
  expect(ApplySlidingBandRotorAngle(cell_shift_request, sliding_artifact, &cell_shift_model) &&
             Near(cell_shift_model.air_gap_elements[0].inner_shift, 0.0, 1e-14) &&
             cell_shift_model.air_gap_elements[0].quad_points[0].node[0] ==
                 sliding_artifact.model.air_gap_elements[0].quad_points[1].node[0],
         "v2 request cyclically remaps inner AGE records after a whole-cell shift");
  // The batch key must batch same-pose current samples, but split a changed
  // sliding operator even if that request is interleaved in caller order.
  MotorBatchArtifactLoad sliding_load;
  sliding_load.status = Status::kOk;
  sliding_load.artifact_sha256 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
  sliding_load.artifact = sliding_artifact;
  MotorSampleRequest same_pose_request = sliding_request;
  same_pose_request.circuit_currents_a = {3.0, -3.0};
  MotorSampleRequest next_pose_request = sliding_request;
  next_pose_request.rotor_angle_deg = 4.0;
  NonlinearModel same_pose_model, next_pose_model;
  const bool batch_key_models_valid =
      ApplySlidingBandRotorAngle(same_pose_request, sliding_artifact, &same_pose_model) &&
      ApplySlidingBandRotorAngle(next_pose_request, sliding_artifact, &next_pose_model);
  const std::string same_pose_key =
      batch_key_models_valid
          ? MotorBatchOperatorKey(same_pose_request, sliding_load, same_pose_model)
          : "";
  const std::string next_pose_key =
      batch_key_models_valid
          ? MotorBatchOperatorKey(next_pose_request, sliding_load, next_pose_model)
          : "";
  std::vector<MotorBatchPreparedItem> sliding_prepared;
  if (batch_key_models_valid) {
    sliding_prepared.push_back({0,
                                &sliding_artifact,
                                shifted_sliding_model,
                                {},
                                sliding_request.circuit_currents_a,
                                same_pose_key});
    sliding_prepared.push_back({1,
                                &sliding_artifact,
                                std::move(next_pose_model),
                                {},
                                next_pose_request.circuit_currents_a,
                                next_pose_key});
    sliding_prepared.push_back({2,
                                &sliding_artifact,
                                std::move(same_pose_model),
                                {},
                                same_pose_request.circuit_currents_a,
                                same_pose_key});
  }
  const std::vector<std::vector<size_t>> sliding_groups =
      GroupMotorBatchPreparedItemsByOperator(sliding_prepared);
  expect(batch_key_models_valid && same_pose_key != next_pose_key && sliding_groups.size() == 2 &&
             sliding_groups[0] == std::vector<size_t>({0, 2}) &&
             sliding_groups[1] == std::vector<size_t>({1}),
         "v2 batch groups same artifact-and-pose samples while preserving response slots");
  parsed_request.circuit_currents_a = {3.0, -3.0};
  expect(RequestMatchesArtifact(parsed_request, parsed_artifact),
         "motor request accepts a new circuit vector for one immutable geometry artifact");
  parsed_request.circuit_currents_a = {3.0};
  expect(!RequestMatchesArtifact(parsed_request, parsed_artifact),
         "motor request rejects a circuit-count mismatch for its geometry artifact");
  parsed_request.circuit_currents_a = {3.0, -3.0};
  parsed_request.base_motor_fem_sha256[0] = '0';
  expect(!RequestMatchesArtifact(parsed_request, parsed_artifact),
         "motor request rejects artifact identity hash mismatch");
  const std::string batch_request_json =
      std::string("{\"protocol\":\"gpu_femm_motor_batch_v1\",") +
      "\"max_items_per_chunk\":2,\"items\":[{\"task_id\":\"case_0001.op\",\"request\":" +
      motor_request_json + "}]}";
  MotorBatchRequest parsed_batch;
  expect(ReadMotorBatchRequestJson(batch_request_json, &parsed_batch, &parser_error) &&
             parsed_batch.items.size() == 1 && parsed_batch.items[0].task_id == "case_0001.op" &&
             RequestMatchesArtifact(parsed_batch.items[0].request, parsed_artifact),
         "motor batch preserves ordered single-sample identity");
  const std::string duplicate_batch_json =
      std::string("{\"protocol\":\"gpu_femm_motor_batch_v1\",") +
      "\"max_items_per_chunk\":1,\"items\":[{\"task_id\":\"duplicate\",\"request\":" +
      motor_request_json + "},{\"task_id\":\"duplicate\",\"request\":" + motor_request_json + "}]}";
  expect(!ReadMotorBatchRequestJson(duplicate_batch_json, &parsed_batch, &parser_error),
         "motor batch rejects duplicate task identities");
  // A tiny real adapter run guards the batch envelope contract, including the
  // top-level PASS spelling consumed by MATLAB and DTO response ordering.
  const std::string batch_artifact_path = "gpu_femm_batch_selftest_artifact.json";
  const std::string batch_single_request_path = "gpu_femm_batch_selftest_single_request.json";
  const std::string batch_single_response_path = "gpu_femm_batch_selftest_single_response.json";
  const std::string batch_request_path = "gpu_femm_batch_selftest_request.json";
  const std::string batch_response_path = "gpu_femm_batch_selftest_response.json";
  const std::string batch_default_response_path = "gpu_femm_batch_selftest_default_response.json";
  const std::string mixed_request_path = "gpu_femm_batch_selftest_mixed_request.json";
  const std::string mixed_response_path = "gpu_femm_batch_selftest_mixed_response.json";
  const std::string sliding_batch_artifact_path = "gpu_femm_sliding_batch_selftest_artifact.json";
  const std::string sliding_batch_request_path = "gpu_femm_sliding_batch_selftest_request.json";
  const std::string sliding_batch_response_path = "gpu_femm_sliding_batch_selftest_response.json";
  const std::string sliding_interleaved_request_path =
      "gpu_femm_batch_selftest_sliding_interleaved_request.json";
  const std::string sliding_interleaved_response_path =
      "gpu_femm_batch_selftest_sliding_interleaved_response.json";
  std::string batch_mesh_artifact = mesh_artifact;
  const std::string selftest_boundary = "\"node_indices\":[0,1,2,3],\"A_Wb_per_m\":[0,0,0,0]";
  const size_t selftest_boundary_at = batch_mesh_artifact.find(selftest_boundary);
  if (selftest_boundary_at != std::string::npos)
    batch_mesh_artifact.replace(selftest_boundary_at, selftest_boundary.size(),
                                "\"node_indices\":[3],\"A_Wb_per_m\":[0]");
  const std::string artifact_sha = Sha256Hex(batch_mesh_artifact);
  // Make the v2 batch fixture's inner AGE records touch the free center node.
  // A rotor-angle remap now changes the assembled operator, so the
  // interleaved-angle test detects accidental reuse of the prior angle.
  std::string batch_sliding_mesh_artifact = sliding_mesh_artifact;
  const std::string original_batch_quad_points =
      "\"quad_points\":[{\"n0\":0,\"w0\":1,\"n1\":1,\"w1\":1,\"n2\":2,\"w2\":1,\"n3\":3,\"w3\":1},{"
      "\"n0\":1,\"w0\":1,\"n1\":2,\"w1\":1,\"n2\":3,\"w2\":1,\"n3\":0,\"w3\":1},{\"n0\":2,\"w0\":1,"
      "\"n1\":3,\"w1\":1,\"n2\":0,\"w2\":1,\"n3\":1,\"w3\":1}]";
  const std::string angle_sensitive_batch_quad_points =
      "\"quad_points\":[{\"n0\":4,\"w0\":1,\"n1\":0,\"w1\":1,\"n2\":2,\"w2\":1,\"n3\":3,\"w3\":1},{"
      "\"n0\":0,\"w0\":1,\"n1\":4,\"w1\":1,\"n2\":3,\"w2\":1,\"n3\":0,\"w3\":1},{\"n0\":4,\"w0\":1,"
      "\"n1\":0,\"w1\":1,\"n2\":0,\"w2\":1,\"n3\":1,\"w3\":1}]";
  const size_t batch_quad_points_at = batch_sliding_mesh_artifact.find(original_batch_quad_points);
  if (batch_quad_points_at != std::string::npos)
    batch_sliding_mesh_artifact.replace(batch_quad_points_at, original_batch_quad_points.size(),
                                        angle_sensitive_batch_quad_points);
  GpuFemmMeshArtifact batch_sliding_artifact;
  expect(ParseGpuFemmMeshArtifactJson(batch_sliding_mesh_artifact, &batch_sliding_artifact,
                                      &parser_error),
         "angle-sensitive v2 batch fixture parses");
  const std::string sliding_artifact_sha = Sha256Hex(batch_sliding_mesh_artifact);
  std::string selftest_motor_request = motor_request_json;
  const size_t artifact_sha_at = selftest_motor_request.find(
      "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
  if (artifact_sha_at != std::string::npos)
    selftest_motor_request.replace(artifact_sha_at, 64, artifact_sha);
  const size_t artifact_path_at = selftest_motor_request.find("fixture.json");
  if (artifact_path_at != std::string::npos)
    selftest_motor_request.replace(artifact_path_at, 12, batch_artifact_path);
  std::string selftest_sliding_request = motor_request_json;
  const size_t sliding_sha_at = selftest_sliding_request.find(
      "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
  if (sliding_sha_at != std::string::npos)
    selftest_sliding_request.replace(sliding_sha_at, 64, sliding_artifact_sha);
  const size_t sliding_path_at = selftest_sliding_request.find("fixture.json");
  if (sliding_path_at != std::string::npos)
    selftest_sliding_request.replace(sliding_path_at, 12, sliding_batch_artifact_path);
  const std::string posed_request = "\"rotor_angle_deg\":3,\"displacement_mm\":[0.1,0]";
  const size_t posed_request_at = selftest_sliding_request.find(posed_request);
  if (posed_request_at != std::string::npos)
    selftest_sliding_request.replace(posed_request_at, posed_request.size(),
                                     "\"rotor_angle_deg\":3,\"displacement_mm\":[0,0]");
  const std::string empty_airgap_request = "\"airgap_radius_mm\":0,\"airgap_angles_deg\":[]";
  const size_t empty_airgap_at = selftest_sliding_request.find(empty_airgap_request);
  if (empty_airgap_at != std::string::npos)
    selftest_sliding_request.replace(empty_airgap_at, empty_airgap_request.size(),
                                     "\"airgap_radius_mm\":0.5,\"airgap_angles_deg\":[0,90]");
  std::string selftest_sliding_angle4_request = selftest_sliding_request;
  const size_t angle4_at = selftest_sliding_angle4_request.find("\"rotor_angle_deg\":3");
  if (angle4_at != std::string::npos)
    selftest_sliding_angle4_request.replace(angle4_at, std::string("\"rotor_angle_deg\":3").size(),
                                            "\"rotor_angle_deg\":4");
  std::string selftest_sliding_angle3_high_current_request = selftest_sliding_request;
  const size_t angle3_current_at =
      selftest_sliding_angle3_high_current_request.find("\"circuit_currents_A\":[2,-2]");
  if (angle3_current_at != std::string::npos)
    selftest_sliding_angle3_high_current_request.replace(
        angle3_current_at, std::string("\"circuit_currents_A\":[2,-2]").size(),
        "\"circuit_currents_A\":[3,-3]");
  std::string selftest_sliding_angle4_low_current_request = selftest_sliding_angle4_request;
  const size_t angle4_current_at =
      selftest_sliding_angle4_low_current_request.find("\"circuit_currents_A\":[2,-2]");
  if (angle4_current_at != std::string::npos)
    selftest_sliding_angle4_low_current_request.replace(
        angle4_current_at, std::string("\"circuit_currents_A\":[2,-2]").size(),
        "\"circuit_currents_A\":[1,-1]");
  {
    std::ofstream artifact_output(batch_artifact_path, std::ios::binary | std::ios::trunc);
    artifact_output << batch_mesh_artifact;
    std::ofstream single_request_output(batch_single_request_path, std::ios::trunc);
    single_request_output << selftest_motor_request;
    std::ofstream batch_request_output(batch_request_path, std::ios::trunc);
    batch_request_output
        << "{\"protocol\":\"gpu_femm_motor_batch_v1\",\"max_items_per_chunk\":4,\"items\":["
        << "{\"task_id\":\"first\",\"request\":" << selftest_motor_request << "},"
        << "{\"task_id\":\"second\",\"request\":" << selftest_motor_request << "},"
        << "{\"task_id\":\"third\",\"request\":" << selftest_motor_request << "},"
        << "{\"task_id\":\"fourth\",\"request\":" << selftest_motor_request << "}]}";
    std::string bad_hash_request = selftest_motor_request;
    const size_t hash_at = bad_hash_request.find(artifact_sha);
    if (hash_at != std::string::npos)
      bad_hash_request.replace(hash_at, artifact_sha.size(), std::string(64, '0'));
    const size_t bad_path_at = bad_hash_request.find(batch_artifact_path);
    if (bad_path_at != std::string::npos)
      bad_hash_request.replace(bad_path_at, batch_artifact_path.size(), "./" + batch_artifact_path);
    std::ofstream mixed_request_output(mixed_request_path, std::ios::trunc);
    mixed_request_output
        << "{\"protocol\":\"gpu_femm_motor_batch_v1\",\"max_items_per_chunk\":4,\"items\":["
        << "{\"task_id\":\"good_one\",\"request\":" << selftest_motor_request << "},"
        << "{\"task_id\":\"bad\",\"request\":" << bad_hash_request << "},"
        << "{\"task_id\":\"good_two\",\"request\":" << selftest_motor_request << "},"
        << "{\"task_id\":\"good_three\",\"request\":" << selftest_motor_request << "}]}";
    std::ofstream sliding_artifact_output(sliding_batch_artifact_path,
                                          std::ios::binary | std::ios::trunc);
    sliding_artifact_output << batch_sliding_mesh_artifact;
    std::ofstream sliding_request_output(sliding_batch_request_path, std::ios::trunc);
    sliding_request_output
        << "{\"protocol\":\"gpu_femm_motor_batch_v1\",\"max_items_per_chunk\":4,\"items\":["
        << "{\"task_id\":\"sliding_first\",\"request\":" << selftest_sliding_request << "},"
        << "{\"task_id\":\"sliding_second\",\"request\":" << selftest_sliding_request << "},"
        << "{\"task_id\":\"sliding_third\",\"request\":" << selftest_sliding_request << "},"
        << "{\"task_id\":\"sliding_fourth\",\"request\":" << selftest_sliding_request << "}]}";
    std::ofstream sliding_interleaved_request_output(sliding_interleaved_request_path,
                                                     std::ios::trunc);
    sliding_interleaved_request_output
        << "{\"protocol\":\"gpu_femm_motor_batch_v1\",\"max_items_per_chunk\":4,\"items\":["
        << "{\"task_id\":\"angle3_base\",\"request\":" << selftest_sliding_request << "},"
        << "{\"task_id\":\"angle4_base\",\"request\":" << selftest_sliding_angle4_request << "},"
        << "{\"task_id\":\"angle3_high\",\"request\":"
        << selftest_sliding_angle3_high_current_request << "},"
        << "{\"task_id\":\"angle4_low\",\"request\":" << selftest_sliding_angle4_low_current_request
        << "}]}";
  }
  MotorSampleRequest selftest_batch_request;
  MotorBatchRequest alias_batch_request;
  const bool selftest_batch_request_valid =
      ReadMotorSampleRequestJson(selftest_motor_request, &selftest_batch_request, &parser_error);
  if (selftest_batch_request_valid) {
    MotorSampleRequest dot_alias_request = selftest_batch_request;
    dot_alias_request.mesh_artifact_path = "./" + batch_artifact_path;
    alias_batch_request.items.push_back({"canonical_first", selftest_batch_request});
    alias_batch_request.items.push_back({"canonical_alias", dot_alias_request});
  }
  const MotorBatchArtifactLoadPlan alias_load_plan =
      MakeMotorBatchArtifactLoadPlan(alias_batch_request);
  expect(selftest_batch_request_valid && alias_load_plan.canonical_paths.size() == 1 &&
             alias_load_plan.item_slots.size() == 2 &&
             alias_load_plan.item_slots[0] == alias_load_plan.item_slots[1] &&
             alias_load_plan.slot_use_counts[0] == 2,
         "motor batch deduplicates canonical artifact-path aliases before SHA and parse");
  const int single_adapter_status =
      MotorSingleSampleAdapter(batch_single_request_path, batch_single_response_path);
  const int batch_adapter_status = MotorBatchAdapter(batch_request_path, batch_response_path, true);
  const int batch_default_adapter_status =
      MotorBatchAdapter(batch_request_path, batch_default_response_path);
  const int mixed_adapter_status = MotorBatchAdapter(mixed_request_path, mixed_response_path);
  const int sliding_batch_adapter_status =
      MotorBatchAdapter(sliding_batch_request_path, sliding_batch_response_path);
  const int sliding_interleaved_adapter_status =
      MotorBatchAdapter(sliding_interleaved_request_path, sliding_interleaved_response_path);
  std::ifstream single_response_input(batch_single_response_path);
  std::stringstream single_response_bytes;
  single_response_bytes << single_response_input.rdbuf();
  std::ifstream batch_response_input(batch_response_path);
  std::stringstream batch_response_bytes;
  batch_response_bytes << batch_response_input.rdbuf();
  std::ifstream batch_default_response_input(batch_default_response_path);
  std::stringstream batch_default_response_bytes;
  batch_default_response_bytes << batch_default_response_input.rdbuf();
  std::ifstream mixed_response_input(mixed_response_path);
  std::stringstream mixed_response_bytes;
  mixed_response_bytes << mixed_response_input.rdbuf();
  std::ifstream sliding_batch_response_input(sliding_batch_response_path);
  std::stringstream sliding_batch_response_bytes;
  sliding_batch_response_bytes << sliding_batch_response_input.rdbuf();
  std::ifstream sliding_interleaved_response_input(sliding_interleaved_response_path);
  std::stringstream sliding_interleaved_response_bytes;
  sliding_interleaved_response_bytes << sliding_interleaved_response_input.rdbuf();
  StrictJson single_response_json, batch_response_json;
  StrictJson mixed_response_json;
  StrictJson sliding_batch_response_json;
  StrictJson sliding_interleaved_response_json;
  const bool single_response_valid =
      single_adapter_status == 0 &&
      StrictJsonParser(single_response_bytes.str()).Parse(&single_response_json, &parser_error);
  const bool batch_response_valid =
      batch_adapter_status == 0 &&
      StrictJsonParser(batch_response_bytes.str()).Parse(&batch_response_json, &parser_error);
  const bool batch_default_response_valid =
      batch_default_adapter_status == 0 &&
      batch_default_response_bytes.str().find("\"timing\"") == std::string::npos;
  const bool mixed_response_valid =
      mixed_adapter_status == 1 &&
      StrictJsonParser(mixed_response_bytes.str()).Parse(&mixed_response_json, &parser_error);
  const bool sliding_batch_response_valid =
      sliding_batch_adapter_status == 0 && StrictJsonParser(sliding_batch_response_bytes.str())
                                               .Parse(&sliding_batch_response_json, &parser_error);
  const bool sliding_interleaved_response_valid =
      sliding_interleaved_adapter_status == 0 &&
      StrictJsonParser(sliding_interleaved_response_bytes.str())
          .Parse(&sliding_interleaved_response_json, &parser_error);
  const StrictJson *batch_status =
      batch_response_valid
          ? JsonMember(batch_response_json, "status", StrictJson::Type::kString, &parser_error)
          : nullptr;
  const StrictJson *batch_solve_status = batch_response_valid
                                             ? JsonMember(batch_response_json, "solve_status",
                                                          StrictJson::Type::kString, &parser_error)
                                             : nullptr;
  const StrictJson *batch_items =
      batch_response_valid
          ? JsonMember(batch_response_json, "items", StrictJson::Type::kArray, &parser_error)
          : nullptr;
  expect(
      single_response_valid && batch_response_valid && batch_default_response_valid &&
          batch_status != nullptr && batch_solve_status != nullptr &&
          batch_status->string == "PASS" && batch_solve_status->string == "PASS" &&
          batch_items != nullptr && batch_items->array.size() == 4 &&
          batch_items->array[0].object.at("task_id").string == "first" &&
          batch_items->array[1].object.at("task_id").string == "second" &&
          batch_items->array[2].object.at("task_id").string == "third" &&
          batch_items->array[3].object.at("task_id").string == "fourth" &&
          batch_response_bytes.str().find("\"mesh_cache_hits\": 3") != std::string::npos &&
          batch_response_bytes.str().find("\"actual_parallel_width\": 4") != std::string::npos &&
          batch_response_bytes.str().find("\"batched_pcg_launches\": ") != std::string::npos &&
          batch_response_bytes.str().find("\"csr_symbolic_cache_hits\": ") != std::string::npos &&
          batch_response_bytes.str().find(
              "\"schema_version\": \"gpu_femm_motor_batch_timing_v1\"") != std::string::npos &&
          batch_response_bytes.str().find("\"host_assembly_seconds\": ") != std::string::npos &&
          batch_response_bytes.str().find("\"device_assembly_seconds\": ") != std::string::npos &&
          batch_response_bytes.str().find("\"postprocess_seconds\": ") != std::string::npos,
      "four-item motor batch adapter preserves PASS, order, and cache evidence");
  const StrictJson *mixed_status =
      mixed_response_valid
          ? JsonMember(mixed_response_json, "status", StrictJson::Type::kString, &parser_error)
          : nullptr;
  const StrictJson *mixed_items =
      mixed_response_valid
          ? JsonMember(mixed_response_json, "items", StrictJson::Type::kArray, &parser_error)
          : nullptr;
  expect(mixed_status != nullptr && mixed_status->string == "FAIL" && mixed_items != nullptr &&
             mixed_items->array.size() == 4 &&
             mixed_response_bytes.str().find("\"timing\"") == std::string::npos &&
             mixed_items->array[0].object.at("task_id").string == "good_one" &&
             mixed_items->array[0].object.at("response").object.at("status").string == "PASS" &&
             mixed_items->array[1].object.at("task_id").string == "bad" &&
             mixed_items->array[1].object.at("response").object.at("status").string == "FAIL" &&
             mixed_items->array[3].object.at("task_id").string == "good_three" &&
             mixed_items->array[3].object.at("response").object.at("status").string == "PASS" &&
             mixed_response_bytes.str().find("\"actual_parallel_width\": 3") != std::string::npos,
         "failed item leaves compacted active batch ordered and independently valid");
  const StrictJson *sliding_batch_items = sliding_batch_response_valid
                                              ? JsonMember(sliding_batch_response_json, "items",
                                                           StrictJson::Type::kArray, &parser_error)
                                              : nullptr;
  expect(sliding_batch_response_valid && sliding_batch_items != nullptr &&
             sliding_batch_items->array.size() == 4 &&
             sliding_batch_items->array[0].object.at("task_id").string == "sliding_first" &&
             sliding_batch_items->array[3].object.at("task_id").string == "sliding_fourth" &&
             sliding_batch_response_bytes.str().find("\"effective_chunk_size\": 4") !=
                 std::string::npos &&
             sliding_batch_response_bytes.str().find("\"actual_parallel_width\": 4") !=
                 std::string::npos,
         "v2 same-pose batch uses requested chunk width and preserves response order");
  MotorSampleRequest interleaved_requests[4];
  const bool interleaved_requests_valid =
      ReadMotorSampleRequestJson(selftest_sliding_request, &interleaved_requests[0],
                                 &parser_error) &&
      ReadMotorSampleRequestJson(selftest_sliding_angle4_request, &interleaved_requests[1],
                                 &parser_error) &&
      ReadMotorSampleRequestJson(selftest_sliding_angle3_high_current_request,
                                 &interleaved_requests[2], &parser_error) &&
      ReadMotorSampleRequestJson(selftest_sliding_angle4_low_current_request,
                                 &interleaved_requests[3], &parser_error);
  const auto response_matches_single_v2 =
      [&parser_error, &batch_sliding_artifact](const StrictJson &response,
                                               const MotorSampleRequest &sample_request) {
        NonlinearSolveResult expected_solution;
        FrozenPostprocessResult expected_postprocess;
        if (SolveMeshArtifactSingleSample(sample_request, batch_sliding_artifact,
                                          &expected_solution, &expected_postprocess) != Status::kOk)
          return false;
        const StrictJson *status =
            JsonMember(response, "status", StrictJson::Type::kString, &parser_error);
        const auto same_number = [&response, &parser_error](const char *field, double expected) {
          const StrictJson *actual =
              JsonMember(response, field, StrictJson::Type::kNumber, &parser_error);
          return actual != nullptr && NearMixed(actual->number, expected, 1e-12, 1e-9);
        };
        const auto same_array = [&response, &parser_error](const char *field,
                                                           const std::vector<double> &expected) {
          const StrictJson *actual =
              JsonMember(response, field, StrictJson::Type::kArray, &parser_error);
          if (actual == nullptr || actual->array.size() != expected.size())
            return false;
          for (size_t index = 0; index < expected.size(); ++index)
            if (actual->array[index].type != StrictJson::Type::kNumber ||
                !NearMixed(actual->array[index].number, expected[index], 1e-12, 1e-9))
              return false;
          return true;
        };
        std::vector<double> expected_radial_flux_density;
        for (const AirgapSample &sample : expected_postprocess.airgap_samples)
          expected_radial_flux_density.push_back(sample.radial_b_t);
        return status != nullptr && status->string == "PASS" &&
               same_number("Fx_N", expected_postprocess.force_x_n) &&
               same_number("Fy_N", expected_postprocess.force_y_n) &&
               same_number("torque_Nm", expected_postprocess.torque_nm) &&
               same_array("actual_circuit_currents_A", expected_solution.circuit_currents_a) &&
               same_array("circuit_flux_linkage_Wb", expected_solution.circuit_flux_linkage_wb) &&
               same_array("airgap_sample_angles_deg", sample_request.airgap_angles_deg) &&
               same_array("airgap_radial_flux_density_T", expected_radial_flux_density);
      };
  const StrictJson *sliding_interleaved_items =
      sliding_interleaved_response_valid ? JsonMember(sliding_interleaved_response_json, "items",
                                                      StrictJson::Type::kArray, &parser_error)
                                         : nullptr;
  const bool interleaved_matches_singles =
      interleaved_requests_valid && sliding_interleaved_items != nullptr &&
      sliding_interleaved_items->array.size() == 4 &&
      response_matches_single_v2(sliding_interleaved_items->array[0].object.at("response"),
                                 interleaved_requests[0]) &&
      response_matches_single_v2(sliding_interleaved_items->array[1].object.at("response"),
                                 interleaved_requests[1]) &&
      response_matches_single_v2(sliding_interleaved_items->array[2].object.at("response"),
                                 interleaved_requests[2]) &&
      response_matches_single_v2(sliding_interleaved_items->array[3].object.at("response"),
                                 interleaved_requests[3]);
  expect(sliding_interleaved_response_valid && sliding_interleaved_items != nullptr &&
             sliding_interleaved_items->array.size() == 4 &&
             sliding_interleaved_items->array[0].object.at("task_id").string == "angle3_base" &&
             sliding_interleaved_items->array[1].object.at("task_id").string == "angle4_base" &&
             sliding_interleaved_items->array[2].object.at("task_id").string == "angle3_high" &&
             sliding_interleaved_items->array[3].object.at("task_id").string == "angle4_low" &&
             sliding_interleaved_response_bytes.str().find("\"effective_chunk_size\": 4") !=
                 std::string::npos &&
             sliding_interleaved_response_bytes.str().find("\"actual_parallel_width\": 2") !=
                 std::string::npos &&
             sliding_interleaved_response_bytes.str().find("\"csr_symbolic_cache_hits\": 6") !=
                 std::string::npos &&
             interleaved_matches_singles,
         "interleaved v2 angles split operators, preserve order, and match single postprocess "
         "outputs");
  if (single_response_valid && batch_response_valid && batch_items != nullptr &&
      batch_items->array.size() == 4) {
    const StrictJson &first_response = batch_items->array[0].object.at("response");
    const auto same_number = [&parser_error](const StrictJson &left, const StrictJson &right,
                                             const char *field) {
      const StrictJson *lhs = JsonMember(left, field, StrictJson::Type::kNumber, &parser_error);
      const StrictJson *rhs = JsonMember(right, field, StrictJson::Type::kNumber, &parser_error);
      return lhs != nullptr && rhs != nullptr && NearMixed(lhs->number, rhs->number, 1e-12, 1e-9);
    };
    const auto same_array = [&parser_error](const StrictJson &left, const StrictJson &right,
                                            const char *field) {
      const StrictJson *lhs = JsonMember(left, field, StrictJson::Type::kArray, &parser_error);
      const StrictJson *rhs = JsonMember(right, field, StrictJson::Type::kArray, &parser_error);
      if (lhs == nullptr || rhs == nullptr || lhs->array.size() != rhs->array.size())
        return false;
      for (size_t index = 0; index < lhs->array.size(); ++index)
        if (lhs->array[index].type != StrictJson::Type::kNumber ||
            rhs->array[index].type != StrictJson::Type::kNumber ||
            !NearMixed(lhs->array[index].number, rhs->array[index].number, 1e-12, 1e-9))
          return false;
      return true;
    };
    expect(same_number(single_response_json, first_response, "Fx_N") &&
               same_number(single_response_json, first_response, "Fy_N") &&
               same_number(single_response_json, first_response, "torque_Nm") &&
               same_array(single_response_json, first_response, "actual_circuit_currents_A") &&
               same_array(single_response_json, first_response, "circuit_flux_linkage_Wb") &&
               same_array(single_response_json, first_response, "airgap_sample_angles_deg") &&
               same_array(single_response_json, first_response, "airgap_radial_flux_density_T"),
           "batched B=4 item matches every single-sample output within parity tolerance");
  }
  const NonlinearTriangle &selected_triangle = parsed_artifact.model.triangles[0];
  const NonlinearTriangle &air_triangle = parsed_artifact.model.triangles[1];
  FrozenPostprocessOptions group_options;
  group_options.selected_group_number = 20;
  group_options.air_group_number = 50;
  expect(IsSelectedPostprocessMaterial(parsed_artifact.model.materials[selected_triangle.material],
                                       group_options, selected_triangle.material) &&
             IsAirPostprocessMaterial(parsed_artifact.model.materials[air_triangle.material],
                                      group_options, air_triangle.material),
         "group-based selected and air regions are distinct");
  StrictJson duplicate_json;
  StrictJsonParser duplicate_parser("{\"x\":1,\"x\":2}");
  expect(!duplicate_parser.Parse(&duplicate_json, &parser_error),
         "gpu_femm_mesh_v1 parser rejects duplicate keys");
  return failures;
}

} // namespace gpu_femm

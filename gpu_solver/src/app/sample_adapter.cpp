#include "app/sample_adapter.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>

#include "core/profile.h"
#include "io/json_output.h"
#include "io/mesh_artifact.h"
#include "io/sample_request.h"
#include "io/strict_json.h"
#include "postprocess/postprocess.h"
#include "solvers/nonlinear_solver.cuh"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace gpu_femm
{

Status SolveMeshArtifactSingleSample(const MotorSampleRequest &request,
                                     const GpuFemmMeshArtifact &artifact,
                                     NonlinearSolveResult *solution,
                                     FrozenPostprocessResult *postprocess)
{
  if (solution == nullptr || postprocess == nullptr || !RequestMatchesArtifact(request, artifact))
    return Status::kInvalidArgument;
  NonlinearModel model;
  if (!ApplySlidingBandRotorAngle(request, artifact, &model))
    return Status::kInvalidArgument;
  NonlinearP1Solver solver;
  Status status = solver.Initialize(model);
  if (status != Status::kOk)
    return status;
  const NonlinearOptions nonlinear_options = ProductionSampleOptions();
  *solution = solver.Solve(request.circuit_currents_a, nonlinear_options);
  if (solution->info.status != Status::kOk)
    return solution->info.status;
  FrozenPostprocessOptions postprocess_options;
  postprocess_options.selected_group_number = request.selected_group_number;
  postprocess_options.air_group_number = request.air_group_number;
  postprocess_options.max_mask_iterations = 4096;
  postprocess_options.airgap_radius_m = request.airgap_radius_mm * 1e-3;
  for (double angle : request.airgap_angles_deg)
    postprocess_options.airgap_angles_rad.push_back(angle * 3.141592653589793238462643383279502884 /
                                                    180.0);
  *postprocess = artifact.has_sliding_band
                     ? ComputeAirGapElementPostprocess(model, *solution, postprocess_options)
                     : ComputeFrozenPostprocess(model, *solution, postprocess_options);
  return postprocess->status;
}

static void WriteMotorSampleResponseJson(std::ostream &output, Status status,
                                         const MotorSampleRequest *request,
                                         const NonlinearSolveResult *solution,
                                         const FrozenPostprocessResult *postprocess)
{
  const auto field = [request](const std::string MotorSampleRequest::*member) {
    return request == nullptr ? "" : request->*member;
  };
  output << std::setprecision(17) << "{\n  \"protocol\": \""
         << (request == nullptr ? "gpu_femm_motor_sample_v1" : request->protocol) << "\",\n"
         << "  \"mesh_artifact_sha256\": \"" << field(&MotorSampleRequest::mesh_artifact_sha256)
         << "\",\n"
         << "  \"base_motor_fem_sha256\": \"" << field(&MotorSampleRequest::base_motor_fem_sha256)
         << "\",\n"
         << "  \"source_fem_sha256\": \"" << field(&MotorSampleRequest::source_fem_sha256)
         << "\",\n"
         << "  \"status\": \"" << (status == Status::kOk ? "PASS" : "FAIL") << "\",\n"
         << "  \"solve_status\": \"" << StatusName(status) << "\",\n"
         << "  \"error_identifier\": \""
         << (status == Status::kOk ? "" : std::string("GPU_FEMM_") + StatusName(status)) << "\",\n"
         << "  \"error_message\": \"" << (status == Status::kOk ? "" : StatusName(status))
         << "\",\n";
  if (status == Status::kOk && request != nullptr && solution != nullptr &&
      postprocess != nullptr) {
    output << "  \"Fx_N\": " << postprocess->force_x_n
           << ",\n  \"Fy_N\": " << postprocess->force_y_n
           << ",\n  \"torque_Nm\": " << postprocess->torque_nm
           << ",\n  \"actual_circuit_currents_A\": [";
    for (size_t i = 0; i < solution->circuit_currents_a.size(); ++i)
      output << (i ? ", " : "") << solution->circuit_currents_a[i];
    output << "],\n  \"circuit_flux_linkage_Wb\": [";
    for (size_t i = 0; i < solution->circuit_flux_linkage_wb.size(); ++i)
      output << (i ? ", " : "") << solution->circuit_flux_linkage_wb[i];
    output << "],\n  \"airgap_sample_angles_deg\": [";
    for (size_t i = 0; i < request->airgap_angles_deg.size(); ++i)
      output << (i ? ", " : "") << request->airgap_angles_deg[i];
    output << "],\n  \"airgap_radial_flux_density_T\": [";
    for (size_t i = 0; i < postprocess->airgap_samples.size(); ++i)
      output << (i ? ", " : "") << postprocess->airgap_samples[i].radial_b_t;
    output << "],\n  \"mesh_element_count\": " << solution->bx_t.size()
           << ",\n  \"convergence\": {\"iterations\": " << solution->info.iterations
           << ", \"residual_l2\": " << solution->info.residual_l2 << "}\n";
  } else {
    output << "  \"Fx_N\": null,\n  \"Fy_N\": null,\n  \"torque_Nm\": null,\n"
           << "  \"actual_circuit_currents_A\": [],\n  \"circuit_flux_linkage_Wb\": [],\n"
           << "  \"airgap_sample_angles_deg\": [],\n  \"airgap_radial_flux_density_T\": [],\n"
           << "  \"mesh_element_count\": 0,\n  \"convergence\": {\"iterations\": "
           << (solution == nullptr ? 0 : solution->info.iterations) << ", \"residual_l2\": ";
    if (solution != nullptr && std::isfinite(solution->info.residual_l2))
      output << solution->info.residual_l2;
    else
      output << "null";
    output << "}\n";
  }
  output << "}";
}

static bool WriteMotorSampleResponse(const std::string &path, Status status,
                                     const MotorSampleRequest *request,
                                     const NonlinearSolveResult *solution,
                                     const FrozenPostprocessResult *postprocess)
{
  std::ofstream output(path, std::ios::trunc);
  if (!output)
    return false;
  WriteMotorSampleResponseJson(output, status, request, solution, postprocess);
  output << '\n';
  return static_cast<bool>(output);
}

int MotorSingleSampleAdapter(const std::string &request_path, const std::string &response_path)
{
  std::ifstream request_file(request_path);
  std::stringstream bytes;
  bytes << request_file.rdbuf();
  MotorSampleRequest request;
  std::string error;
  if (!request_file || !ReadMotorSampleRequestJson(bytes.str(), &request, &error)) {
    WriteMotorSampleResponse(response_path, Status::kInvalidArgument, nullptr, nullptr, nullptr);
    std::cerr << "FAIL motor single-sample request: " << error << '\n';
    return 1;
  }
  std::ifstream artifact_file(request.mesh_artifact_path, std::ios::binary);
  std::stringstream artifact_bytes;
  artifact_bytes << artifact_file.rdbuf();
  GpuFemmMeshArtifact artifact;
  Status status = Status::kOk;
  if (!artifact_file) {
    status = Status::kInputIo;
    error = "cannot open mesh artifact";
  } else if (Sha256Hex(artifact_bytes.str()) != request.mesh_artifact_sha256) {
    status = Status::kInvalidArgument;
    error = "mesh artifact SHA mismatch";
  } else if (!ParseGpuFemmMeshArtifactJson(artifact_bytes.str(), &artifact, &error))
    status = Status::kInvalidArgument;
  else if (!RequestMatchesArtifact(request, artifact)) {
    status = Status::kInvalidArgument;
    error = "artifact identity, pose, or circuit-count mismatch";
  }
  NonlinearSolveResult solution;
  FrozenPostprocessResult postprocess;
  const bool solve_attempted = status == Status::kOk;
  if (solve_attempted)
    status = SolveMeshArtifactSingleSample(request, artifact, &solution, &postprocess);
  if (!WriteMotorSampleResponse(response_path, status, &request,
                                solve_attempted ? &solution : nullptr,
                                status == Status::kOk ? &postprocess : nullptr))
    return 1;
  if (status != Status::kOk) {
    std::cerr << "FAIL motor single-sample: " << error << (error.empty() ? StatusName(status) : "")
              << '\n';
    return 1;
  }
  return 0;
}

bool PlanarDcRequestMatchesArtifact(const PlanarDcSampleRequest &request,
                                    const GpuFemmMeshArtifact &artifact)
{
  const bool generic_request = request.protocol == "gpu_femm_magnetostatic_sample_v1";
  const bool generic_artifact = artifact.schema_version == "gpu_femm_magnetostatic_mesh_v1";
  if (generic_request != generic_artifact)
    return false;
  if (request.circuit_currents_a.size() != static_cast<size_t>(artifact.model.circuit_count))
    return false;
  const auto has_group = [&artifact](int32_t group) {
    return std::any_of(
        artifact.model.materials.begin(), artifact.model.materials.end(),
        [group](const NonlinearMaterial &material) { return material.group_number == group; });
  };
  const bool groups_enabled = request.force_group_number >= 0;
  if (groups_enabled &&
      (!has_group(request.force_group_number) || !has_group(request.stress_air_group_number)))
    return false;
  const bool needs_postprocess = request.compute_force_torque || !request.airgap_angles_deg.empty();
  // The generic protocol deliberately exposes only the field solve.  Its
  // periodic reduction and cylindrical operators are not yet implemented in
  // the legacy weighted-stress/air-gap postprocessors.
  if (generic_request)
    return !needs_postprocess && request.force_group_number == -1 &&
           request.stress_air_group_number == -1 && request.airgap_radius_mm == 0.0 &&
           request.airgap_angles_deg.empty() && request.sliding_band_angle_deg == 0.0;
  if (!artifact.has_sliding_band) {
    // A conforming artifact is already posed.  Rotation is only meaningful
    // for a native sliding-band artifact, and weighted-stress postprocessing
    // needs an explicit selected/air group pair.
    return request.sliding_band_angle_deg == 0.0 && (!needs_postprocess || groups_enabled);
  }
  if (!request.airgap_angles_deg.empty()) {
    if (artifact.model.air_gap_elements.empty())
      return false;
    for (const AirGapElement &age : artifact.model.air_gap_elements) {
      const double inner_mm = 1000.0 * age.inner_radius_m;
      const double outer_mm = 1000.0 * age.outer_radius_m;
      const double tolerance_mm = 1e-9 * std::max(1.0, std::abs(outer_mm));
      if (request.airgap_radius_mm < inner_mm - tolerance_mm ||
          request.airgap_radius_mm > outer_mm + tolerance_mm)
        return false;
    }
  }
  return true;
}

bool WritePlanarDcSampleResponse(const std::string &path, Status status, Status solve_status,
                                 bool postprocess_requested, Status postprocess_status,
                                 const PlanarDcSampleRequest *request,
                                 const NonlinearSolveResult *solution,
                                 const FrozenPostprocessResult *postprocess,
                                 const std::string &diagnostic, const NonlinearModel *model)
{
  std::ofstream output(path, std::ios::trunc);
  if (!output)
    return false;
  const bool solved = solve_status == Status::kOk && request != nullptr && solution != nullptr;
  const bool postprocessed = solved && postprocess_requested && postprocess_status == Status::kOk &&
                             postprocess != nullptr;
  const std::string message =
      status == Status::kOk ? "" : (diagnostic.empty() ? StatusName(status) : diagnostic);
  const std::string protocol =
      request == nullptr ? "gpu_femm_planar_dc_sample_v1" : request->protocol;
  const bool generic = protocol == "gpu_femm_magnetostatic_sample_v1";
  const MagneticProblemType problem_type =
      model == nullptr ? MagneticProblemType::kPlanar : model->problem_type;
  const char *node_field_name = generic ? "node_potential" : "node_A_Wb_per_m";
  const char *first_field_name = generic ? "element_B_component_1_T" : "element_Bx_T";
  const char *second_field_name = generic ? "element_B_component_2_T" : "element_By_T";
  output << std::setprecision(17) << "{\n  \"protocol\": \"" << protocol << "\",\n"
         << "  \"mesh_artifact_sha256\": \""
         << (request == nullptr ? "" : request->mesh_artifact_sha256) << "\",\n";
  if (generic)
    output << "  \"problem_type\": \""
           << (problem_type == MagneticProblemType::kAxisymmetric ? "axisymmetric" : "planar")
           << "\",\n  \"field_components\": [\""
           << (problem_type == MagneticProblemType::kAxisymmetric ? "Br" : "Bx") << "\", \""
           << (problem_type == MagneticProblemType::kAxisymmetric ? "Bz" : "By")
           << "\"],\n  \"node_potential_quantity\": \""
           << (problem_type == MagneticProblemType::kAxisymmetric ? "poloidal_flux_function"
                                                                  : "magnetic_vector_potential")
           << "\",\n  \"node_potential_unit\": \""
           << (problem_type == MagneticProblemType::kAxisymmetric ? "Wb" : "Wb/m") << "\",\n";
  output << "  \"status\": \"" << (status == Status::kOk ? "PASS" : "FAIL") << "\",\n"
         << "  \"solve_status\": \"" << StatusName(solve_status) << "\",\n"
         << "  \"postprocess_status\": \""
         << (!postprocess_requested
                 ? "NOT_REQUESTED"
                 : (solve_status == Status::kOk ? StatusName(postprocess_status) : "NOT_RUN"))
         << "\",\n"
         << "  \"error_identifier\": \""
         << (status == Status::kOk ? "" : std::string("GPU_FEMM_") + StatusName(status)) << "\",\n"
         << "  \"error_message\": \"" << EscapeJsonString(message) << "\",\n";
  if (postprocessed && request->compute_force_torque) {
    output << "  \"Fx_N\": " << postprocess->force_x_n
           << ",\n  \"Fy_N\": " << postprocess->force_y_n
           << ",\n  \"torque_Nm\": " << postprocess->torque_nm << ",\n";
  } else {
    output << "  \"Fx_N\": null,\n  \"Fy_N\": null,\n  \"torque_Nm\": null,\n";
  }
  output << "  \"actual_circuit_currents_A\": [";
  if (solved)
    for (size_t i = 0; i < solution->circuit_currents_a.size(); ++i)
      output << (i ? ", " : "") << solution->circuit_currents_a[i];
  output << "],\n  \"circuit_flux_linkage_Wb\": [";
  if (solved)
    for (size_t i = 0; i < solution->circuit_flux_linkage_wb.size(); ++i)
      output << (i ? ", " : "") << solution->circuit_flux_linkage_wb[i];
  output << "],\n  \"airgap_sample_angles_deg\": [";
  if (postprocessed)
    for (size_t i = 0; i < request->airgap_angles_deg.size(); ++i)
      output << (i ? ", " : "") << request->airgap_angles_deg[i];
  output << "],\n  \"airgap_radial_flux_density_T\": [";
  if (postprocessed)
    for (size_t i = 0; i < postprocess->airgap_samples.size(); ++i)
      output << (i ? ", " : "") << postprocess->airgap_samples[i].radial_b_t;
  output << "],\n  \"mesh_node_count\": " << (solved ? solution->a_wb_per_m.size() : 0)
         << ",\n  \"mesh_element_count\": " << (solved ? solution->bx_t.size() : 0)
         << ",\n  \"field_solution_included\": "
         << (solved && request->include_field_solution ? "true" : "false") << ",\n  \""
         << node_field_name << "\": [";
  if (solved && request->include_field_solution)
    for (size_t i = 0; i < solution->a_wb_per_m.size(); ++i) {
      double potential = solution->a_wb_per_m[i];
      if (generic && problem_type == MagneticProblemType::kAxisymmetric && model != nullptr &&
          i < model->nodes.size())
        potential *= 6.283185307179586476925286766559005768 * model->nodes[i].x_m;
      output << (i ? ", " : "") << potential;
    }
  output << "],\n  \"" << first_field_name << "\": [";
  if (solved && request->include_field_solution)
    for (size_t i = 0; i < solution->bx_t.size(); ++i)
      output << (i ? ", " : "") << solution->bx_t[i];
  output << "],\n  \"" << second_field_name << "\": [";
  if (solved && request->include_field_solution)
    for (size_t i = 0; i < solution->by_t.size(); ++i)
      output << (i ? ", " : "") << solution->by_t[i];
  output << "],\n  \"convergence\": {\"iterations\": "
         << (solution == nullptr ? 0 : solution->info.iterations) << ", \"residual_l2\": ";
  if (solution != nullptr && std::isfinite(solution->info.residual_l2))
    output << solution->info.residual_l2;
  else
    output << "null";
  output << "}\n}\n";
  return static_cast<bool>(output);
}

std::string CanonicalArtifactPathKey(const std::string &path)
{
  std::error_code error;
  const std::filesystem::path resolved =
      std::filesystem::weakly_canonical(std::filesystem::path(path), error);
  std::string key = error ? path : resolved.generic_string();
#ifdef _WIN32
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
#endif
  return key;
}

bool SameFilesystemObject(const std::string &first, const std::string &second)
{
  std::error_code error;
  if (std::filesystem::exists(first, error) && !error && std::filesystem::exists(second, error) &&
      !error && std::filesystem::equivalent(first, second, error) && !error)
    return true;
  return CanonicalArtifactPathKey(first) == CanonicalArtifactPathKey(second);
}

static bool AtomicReplaceFile(const std::filesystem::path &source,
                              const std::filesystem::path &destination)
{
#ifdef _WIN32
  return MoveFileExW(source.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  std::error_code error;
  std::filesystem::rename(source, destination, error);
  return !error;
#endif
}

static int PlanarDcSingleSampleAdapterDirect(const std::string &request_path,
                                             const std::string &response_path,
                                             NonlinearAssemblyMode assembly_mode)
{
  std::ifstream request_file(request_path);
  std::stringstream request_bytes;
  request_bytes << request_file.rdbuf();
  PlanarDcSampleRequest request;
  std::string error;
  if (!request_file || !ReadPlanarDcSampleRequestJson(request_bytes.str(), &request, &error)) {
    WritePlanarDcSampleResponse(response_path, Status::kInvalidArgument, Status::kInvalidArgument,
                                false, Status::kOk, nullptr, nullptr, nullptr, error);
    std::cerr << "FAIL planar-DC request: " << error << '\n';
    return 1;
  }
  std::ifstream artifact_file(request.mesh_artifact_path, std::ios::binary);
  std::stringstream artifact_bytes;
  artifact_bytes << artifact_file.rdbuf();
  GpuFemmMeshArtifact artifact;
  Status status = Status::kOk;
  Status solve_status = Status::kOk;
  Status postprocess_status = Status::kOk;
  if (!artifact_file) {
    status = Status::kInputIo;
    error = "cannot open mesh artifact";
  } else if (Sha256Hex(artifact_bytes.str()) != request.mesh_artifact_sha256) {
    status = Status::kInvalidArgument;
    error = "mesh artifact SHA mismatch";
  } else if (!ParseGpuFemmMeshArtifactJson(artifact_bytes.str(), &artifact, &error)) {
    status = Status::kInvalidArgument;
  } else if (!PlanarDcRequestMatchesArtifact(request, artifact)) {
    status = Status::kInvalidArgument;
    error = "artifact physics, circuit count, group, or sliding-band request mismatch";
  }

  NonlinearModel model;
  NonlinearSolveResult solution;
  FrozenPostprocessResult postprocess;
  bool solve_attempted = false;
  bool postprocess_attempted = false;
  if (status == Status::kOk) {
    if (!ApplySlidingBandAngle(request.sliding_band_angle_deg, artifact, &model)) {
      status = Status::kInvalidArgument;
      error = "could not apply the requested sliding-band angle";
    } else {
      NonlinearP1Solver solver;
      status = solver.Initialize(model);
      if (status == Status::kOk) {
        const NonlinearOptions options = ProductionSampleOptions();
        solve_attempted = true;
        solution =
            assembly_mode == NonlinearAssemblyMode::kHost
                ? solver.Solve(request.circuit_currents_a, options)
                : solver.SolveAccelerated(request.circuit_currents_a, options, assembly_mode);
        status = solution.info.status;
      }
    }
  }
  solve_status = status;
  const bool postprocess_requested =
      request.compute_force_torque || !request.airgap_angles_deg.empty();
  if (solve_status == Status::kOk && postprocess_requested) {
    FrozenPostprocessOptions options;
    options.selected_group_number = request.force_group_number;
    options.air_group_number = request.stress_air_group_number;
    options.max_mask_iterations = 4096;
    options.airgap_radius_m = request.airgap_radius_mm * 1e-3;
    for (double angle : request.airgap_angles_deg)
      options.airgap_angles_rad.push_back(angle * 3.141592653589793238462643383279502884 / 180.0);
    postprocess_attempted = true;
    postprocess = artifact.has_sliding_band
                      ? ComputeAirGapElementPostprocess(model, solution, options)
                      : ComputeFrozenPostprocess(model, solution, options);
    postprocess_status = postprocess.status;
    status = postprocess_status;
  }
  if (!WritePlanarDcSampleResponse(
          response_path, status, solve_status, postprocess_requested, postprocess_status, &request,
          solve_attempted ? &solution : nullptr,
          postprocess_attempted && postprocess_status == Status::kOk ? &postprocess : nullptr,
          error, &artifact.model))
    return 1;
  if (status != Status::kOk) {
    std::cerr << "FAIL magnetostatic solve: " << (error.empty() ? StatusName(status) : error)
              << '\n';
    return 1;
  }
  return 0;
}

int PlanarDcSingleSampleAdapter(const std::string &request_path, const std::string &response_path,
                                NonlinearAssemblyMode assembly_mode)
{
  if (SameFilesystemObject(request_path, response_path)) {
    std::cerr << "FAIL magnetostatic solve: response must not overwrite request\n";
    return 1;
  }

  std::ifstream request_file(request_path);
  std::stringstream request_bytes;
  request_bytes << request_file.rdbuf();
  PlanarDcSampleRequest request;
  std::string parse_error;
  if (request_file && ReadPlanarDcSampleRequestJson(request_bytes.str(), &request, &parse_error) &&
      SameFilesystemObject(request.mesh_artifact_path, response_path)) {
    std::cerr << "FAIL magnetostatic solve: response must not overwrite mesh artifact\n";
    return 1;
  }

  static std::atomic<uint64_t> sequence{0};
  std::filesystem::path temporary(response_path);
  temporary += ".tmp." + std::to_string(ProfileClock::now().time_since_epoch().count()) + "." +
               std::to_string(sequence.fetch_add(1));
  const int result =
      PlanarDcSingleSampleAdapterDirect(request_path, temporary.string(), assembly_mode);
  std::error_code error;
  if (!std::filesystem::is_regular_file(temporary, error) ||
      !AtomicReplaceFile(temporary, std::filesystem::path(response_path))) {
    std::filesystem::remove(temporary, error);
    std::cerr << "FAIL magnetostatic solve: could not atomically publish response\n";
    return 1;
  }
  return result;
}

} // namespace gpu_femm

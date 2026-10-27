#include "commands.h"
#include "reference.h"
#include "app/sample_adapter.h"
#include "io/sample_request.h"
#include "io/strict_json.h"
#include "solvers/nonlinear_solver.cuh"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace gpu_femm
{

bool WriteSingleSampleResponse(const std::string &path, Status status,
                               const SingleSampleRequest *request,
                               const NonlinearSolveResult *solution,
                               const FrozenPostprocessResult *postprocess)
{
  std::ofstream output(path, std::ios::trunc);
  if (!output)
    return false;
  output << std::setprecision(17) << "{\n"
         << "  \"protocol\": \"gpu_femm_single_sample_v1\",\n"
         << "  \"source_motor_fem_sha256\": \""
         << (request == nullptr ? "" : request->source_motor_fem_sha256) << "\",\n"
         << "  \"status\": \"" << (status == Status::kOk ? "PASS" : "FAIL") << "\",\n"
         << "  \"solve_status\": \"" << (status == Status::kOk ? "PASS" : "FAIL") << "\",\n"
         << "  \"error_identifier\": \""
         << (status == Status::kOk ? "" : std::string("GPU_FEMM_") + StatusName(status)) << "\",\n"
         << "  \"error_message\": \"" << (status == Status::kOk ? "" : StatusName(status))
         << "\",\n";
  if (status == Status::kOk && request != nullptr && solution != nullptr &&
      postprocess != nullptr) {
    output << "  \"Fx_N\": " << postprocess->force_x_n << ",\n"
           << "  \"Fy_N\": " << postprocess->force_y_n << ",\n"
           << "  \"torque_Nm\": " << postprocess->torque_nm << ",\n"
           << "  \"actual_circuit_currents_A\": [" << request->current_a << "],\n"
           << "  \"circuit_flux_linkage_Wb\": [" << solution->flux_linkage_wb << "],\n"
           << "  \"airgap_sample_angles_deg\": [";
    for (size_t index = 0; index < request->airgap_angles_deg.size(); ++index)
      output << (index == 0 ? "" : ", ") << request->airgap_angles_deg[index];
    output << "],\n  \"airgap_radial_flux_density_T\": [";
    for (size_t index = 0; index < postprocess->airgap_samples.size(); ++index)
      output << (index == 0 ? "" : ", ") << postprocess->airgap_samples[index].radial_b_t;
    output << "],\n  \"mesh_element_count\": " << solution->bx_t.size() << ",\n"
           << "  \"convergence\": {\"iterations\": " << solution->info.iterations
           << ", \"residual_l2\": " << solution->info.residual_l2 << "}\n";
  } else {
    output << "  \"Fx_N\": null,\n  \"Fy_N\": null,\n  \"torque_Nm\": null,\n"
           << "  \"actual_circuit_currents_A\": [],\n"
           << "  \"circuit_flux_linkage_Wb\": [],\n"
           << "  \"airgap_sample_angles_deg\": [],\n"
           << "  \"airgap_radial_flux_density_T\": [],\n"
           << "  \"mesh_element_count\": 0,\n"
           << "  \"convergence\": {\"iterations\": 0, \"residual_l2\": null}\n";
  }
  output << "}\n";
  return static_cast<bool>(output);
}

int SingleSampleAdapter(const std::string &request_path, const std::string &response_path)
{
  std::ifstream protocol_file(request_path);
  std::stringstream protocol_bytes;
  protocol_bytes << protocol_file.rdbuf();
  StrictJson protocol_root;
  std::string protocol_error;
  if (protocol_file &&
      StrictJsonParser(protocol_bytes.str()).Parse(&protocol_root, &protocol_error) &&
      protocol_root.type == StrictJson::Type::kObject) {
    const auto protocol = protocol_root.object.find("protocol");
    if (protocol != protocol_root.object.end() &&
        protocol->second.type == StrictJson::Type::kString &&
        protocol->second.string == "gpu_femm_motor_sample_v1")
      return MotorSingleSampleAdapter(request_path, response_path);
  }
  SingleSampleRequest request;
  std::string error;
  if (!ReadSingleSampleRequest(request_path, &request, &error)) {
    WriteSingleSampleResponse(response_path, Status::kInvalidArgument, nullptr, nullptr, nullptr);
    std::cerr << "FAIL single-sample request: " << error << '\n';
    return 1;
  }
  NonlinearFemmReference reference;
  if (!ReadNonlinearFemmReference(request.stem, request.curve_directory, &reference, &error)) {
    WriteSingleSampleResponse(response_path, Status::kInputIo, &request, nullptr, nullptr);
    std::cerr << "FAIL single-sample reference: " << error << '\n';
    return 1;
  }
  NonlinearP1Solver solver;
  Status status = solver.Initialize(reference.model);
  NonlinearSolveResult solution;
  if (status == Status::kOk) {
    NonlinearOptions options;
    options.relative_tolerance = 1e-8;
    options.max_newton_iterations = 128;
    solution = solver.Solve(request.current_a, options);
    status = solution.info.status;
  }
  FrozenPostprocessResult postprocess;
  if (status == Status::kOk) {
    FrozenPostprocessOptions options;
    options.selected_material_label = request.selected_material_label;
    options.airgap_radius_m = request.airgap_radius_mm * 1e-3;
    for (const double angle_deg : request.airgap_angles_deg)
      options.airgap_angles_rad.push_back(angle_deg * 3.141592653589793238462643383279502884 /
                                          180.0);
    postprocess = ComputeFrozenPostprocess(reference.model, solution, options);
    status = postprocess.status;
  }
  if (!WriteSingleSampleResponse(response_path, status, &request,
                                 status == Status::kOk ? &solution : nullptr,
                                 status == Status::kOk ? &postprocess : nullptr)) {
    std::cerr << "FAIL single-sample response: cannot write " << response_path << '\n';
    return 1;
  }
  if (status != Status::kOk) {
    std::cerr << "FAIL single-sample: " << StatusName(status) << '\n';
    return 1;
  }
  return 0;
}

} // namespace gpu_femm

#include "app/cli.h"
#include "app/sample_adapter.h"
#include "app/motor_batch.h"
#include "app/motor_worker.h"
#include "commands.h"

#include <iostream>
#include <string>
#include <vector>

int gpu_femm::RunCli(int argc, char **argv)
{
  // Keep argv[0] intact: output-file protection compares it with the response.
  std::vector<char *> arguments;
  int next = 1;
  bool have_preconditioner = false, have_solver = false;
  while (next < argc && (std::string(argv[next]) == "--preconditioner" ||
                         std::string(argv[next]) == "--linear-solver")) {
    const bool solver = std::string(argv[next]) == "--linear-solver";
    if (next + 2 >= argc || (solver ? have_solver : have_preconditioner)) {
      std::cerr << "Solver options require a single value and a command\n";
      return 2;
    }
    const std::string value = argv[next + 1];
    if (solver) {
      if (value != "pcg" && value != "cudss") {
        std::cerr << "--linear-solver requires pcg or cudss\n";
        return 2;
      }
#ifndef GPU_FEMM_ENABLE_CUDSS
      if (value == "cudss") {
        std::cerr << "cuDSS was not enabled in this build\n";
        return 2;
      }
#endif
      gpu_femm::selected_linear_solver = value == "cudss" ? gpu_femm::LinearSolverBackend::kCudss
                                                          : gpu_femm::LinearSolverBackend::kPcg;
      have_solver = true;
    } else {
      if (value != "jacobi" && value != "block-jacobi") {
        std::cerr << "--preconditioner requires jacobi or block-jacobi\n";
        return 2;
      }
      gpu_femm::selected_preconditioner = value == "block-jacobi"
                                              ? gpu_femm::PcgPreconditioner::kBlockJacobi
                                              : gpu_femm::PcgPreconditioner::kJacobi;
      have_preconditioner = true;
    }
    next += 2;
  }
  if (gpu_femm::selected_linear_solver == gpu_femm::LinearSolverBackend::kCudss &&
      have_preconditioner) {
    std::cerr << "--preconditioner only applies to pcg\n";
    return 2;
  }
  if (next != 1) {
    arguments.push_back(argv[0]);
    for (int i = next; i < argc; ++i)
      arguments.push_back(argv[i]);
    arguments.push_back(nullptr);
    argc = static_cast<int>(arguments.size()) - 1;
    argv = arguments.data();
  }
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    if (gpu_femm::selected_linear_solver == gpu_femm::LinearSolverBackend::kCudss)
      return gpu_femm::CudssRegressionTests() == 0 ? 0 : 1;
    return gpu_femm::SelfTest();
  }
  if (argc == 2 && std::string(argv[1]) == "--internal-preconditioner-test") {
    return gpu_femm::BlockPreconditionerTests(false) == 0 ? 0 : 1;
  }
  if (argc == 2 && std::string(argv[1]) == "--internal-pcg-storage-test") {
    return gpu_femm::PcgSpmvLayoutTests() == 0 ? 0 : 1;
  }
  if (argc == 2 && std::string(argv[1]) == "--internal-cudss-test") {
    return gpu_femm::CudssRegressionTests() == 0 ? 0 : 1;
  }
  if (argc == 3 && std::string(argv[1]) == "--femm-reference") {
    return gpu_femm::FemmReferenceTest(argv[2]);
  }
  if (argc == 4 && std::string(argv[1]) == "--nonlinear-reference") {
    return gpu_femm::NonlinearFemmReferenceTest(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--postprocess-reference") {
    return gpu_femm::FrozenPostprocessReferenceTest(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--single-sample") {
    return gpu_femm::SingleSampleAdapter(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--solve") {
    if (gpu_femm::SameFilesystemObject(argv[0], argv[3])) {
      std::cerr << "FAIL magnetostatic solve: response must not overwrite solver executable\n";
      return 1;
    }
    return gpu_femm::PlanarDcSingleSampleAdapter(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--internal-solve-host") {
    return gpu_femm::PlanarDcSingleSampleAdapter(argv[2], argv[3],
                                                 gpu_femm::NonlinearAssemblyMode::kHost);
  }
  if (argc == 4 && std::string(argv[1]) == "--internal-solve-device") {
    return gpu_femm::PlanarDcSingleSampleAdapter(argv[2], argv[3],
                                                 gpu_femm::NonlinearAssemblyMode::kDevice);
  }
  if (argc == 4 && std::string(argv[1]) == "--motor-single-sample") {
    return gpu_femm::MotorSingleSampleAdapter(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--motor-batch") {
    return gpu_femm::MotorBatchAdapter(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--motor-batch-profile") {
    return gpu_femm::MotorBatchAdapter(argv[2], argv[3], true);
  }
  if (argc == 2 && std::string(argv[1]) == "--motor-worker") {
    return gpu_femm::MotorBatchWorker(argv[0]);
  }
  if (argc == 4 && std::string(argv[1]) == "--internal-motor-batch-profile-host") {
    return gpu_femm::MotorBatchAdapter(argv[2], argv[3], true,
                                       gpu_femm::NonlinearAssemblyMode::kHost);
  }
  if (argc == 4 && std::string(argv[1]) == "--internal-motor-batch-profile-device") {
    return gpu_femm::MotorBatchAdapter(argv[2], argv[3], true,
                                       gpu_femm::NonlinearAssemblyMode::kDevice);
  }
  if (argc == 3 && std::string(argv[1]) == "--mesh-artifact") {
    return gpu_femm::GpuFemmMeshArtifactTest(argv[2]);
  }
  if (argc == 2 && std::string(argv[1]) == "--capabilities") {
    std::cout << "{\n"
              << "  \"schema_version\": \"gpu_femm_capabilities_v1\",\n"
              << "  \"sample_protocol\": \"gpu_femm_planar_dc_sample_v1\",\n"
              << "  \"sample_protocols\": [\"gpu_femm_planar_dc_sample_v1\", "
                 "\"gpu_femm_magnetostatic_sample_v1\"],\n"
              << "  \"motor_sample_protocols\": [\"gpu_femm_motor_sample_v1\", "
                 "\"gpu_femm_motor_sample_v2\", \"gpu_femm_motor_sample_v3\", "
                 "\"gpu_femm_motor_sample_v4\"],\n"
              << "  \"motor_batch_protocols\": [\"gpu_femm_motor_batch_v1\", "
                 "\"gpu_femm_motor_batch_v2\", \"gpu_femm_motor_batch_v3\", "
                 "\"gpu_femm_motor_batch_v4\"],\n"
              << "  \"motor_batch_stream_v1\": true,\n"
              << "  \"problem_types\": [\"planar\", \"axisymmetric\"],\n"
              << "  \"frequency_hz\": [0],\n"
              << "  \"precision\": \"fp64\",\n"
#ifdef GPU_FEMM_ENABLE_CUDSS
              << "  \"linear_solvers\": [\"pcg\", \"cudss\"],\n"
#else
              << "  \"linear_solvers\": [\"pcg\"],\n"
#endif
              << "  \"default_linear_solver\": \"pcg\",\n"
              << "  \"preconditioners\": [\"jacobi\", \"block-jacobi\"],\n"
              << "  \"default_preconditioner\": \"jacobi\",\n"
              << "  \"nonlinear_bh\": true,\n"
              << "  \"permanent_magnets\": true,\n"
              << "  \"native_sliding_band\": true,\n"
              << "  \"general_periodic_node_constraints\": true,\n"
              << "  \"motor_tangent_multi_rhs_v1\": true,\n"
              << "  \"motor_sector_performance_180_v1\": true,\n"
              << "  \"motor_sector_posed_periodic_wst_v1\": true,\n"
              << "  \"motor_sector_native_age_v1\": false,\n"
              << "  \"motor_sector_native_age_cyclic_v1\": true,\n"
              << "  \"full_field_output\": true,\n"
              << "  \"ac_complex\": false,\n"
              << "  \"axisymmetric\": true,\n"
              << "  \"axisymmetric_assembly\": \"device\",\n"
              << "  \"generic_force_airgap_postprocessing\": false\n"
              << "}\n";
    return 0;
  }
  std::cerr << "Usage: gpu_linear_p1_poc [--linear-solver pcg|cudss] [--preconditioner "
               "jacobi|block-jacobi] --self-test | --femm-reference <stem>"
            << " | --nonlinear-reference <stem> <curve_dir>"
            << " | --postprocess-reference <stem> <curve_dir>"
            << " | --single-sample <request.json> <response.json>"
            << " | --solve <request.json> <response.json>"
            << " | --motor-single-sample <request.json> <response.json>"
            << " | --motor-batch <request.json> <response.json>"
            << " | --motor-batch-profile <request.json> <response.json>"
            << " | --motor-worker"
            << " | --mesh-artifact <gpu_femm_mesh_v1.json>"
            << " | --capabilities\n";
  return 2;
}

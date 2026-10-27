#pragma once

#include "core/types.h"
#include "io/mesh_artifact.h"
#include "io/sample_request.h"
#include "postprocess/postprocess.h"
#include "solvers/nonlinear_solver.cuh"

#include <string>

namespace gpu_femm
{

Status SolveMeshArtifactSingleSample(const MotorSampleRequest &request,
                                     const GpuFemmMeshArtifact &artifact,
                                     NonlinearSolveResult *solution,
                                     FrozenPostprocessResult *postprocess);

int MotorSingleSampleAdapter(const std::string &request_path, const std::string &response_path);

bool PlanarDcRequestMatchesArtifact(const PlanarDcSampleRequest &request,
                                    const GpuFemmMeshArtifact &artifact);

bool WritePlanarDcSampleResponse(const std::string &path, Status status, Status solve_status,
                                 bool postprocess_requested, Status postprocess_status,
                                 const PlanarDcSampleRequest *request,
                                 const NonlinearSolveResult *solution,
                                 const FrozenPostprocessResult *postprocess,
                                 const std::string &diagnostic,
                                 const NonlinearModel *model = nullptr);

std::string CanonicalArtifactPathKey(const std::string &path);

bool SameFilesystemObject(const std::string &first, const std::string &second);

int PlanarDcSingleSampleAdapter(const std::string &request_path, const std::string &response_path,
                                NonlinearAssemblyMode assembly_mode = NonlinearAssemblyMode::kAuto);

} // namespace gpu_femm

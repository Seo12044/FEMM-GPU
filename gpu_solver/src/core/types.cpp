#include "types.h"

namespace gpu_femm
{

const char *StatusName(Status status)
{
  switch (status) {
  case Status::kOk:
    return "OK";
  case Status::kInputIo:
    return "INPUT_IO";
  case Status::kInvalidArgument:
    return "INVALID_ARGUMENT";
  case Status::kUnsupportedFeature:
    return "UNSUPPORTED_FEATURE";
  case Status::kMeshInvalid:
    return "MESH_INVALID";
  case Status::kBoundaryInvalid:
    return "BOUNDARY_INVALID";
  case Status::kGpuUnavailable:
    return "GPU_UNAVAILABLE";
  case Status::kGpuAllocationFailed:
    return "GPU_ALLOCATION_FAILED";
  case Status::kAssemblyFailed:
    return "ASSEMBLY_FAILED";
  case Status::kLinearSolveNotConverged:
    return "LINEAR_SOLVE_NOT_CONVERGED";
  case Status::kLinearSolveBreakdown:
    return "LINEAR_SOLVE_BREAKDOWN";
  case Status::kNumericalNonfinite:
    return "NUMERICAL_NONFINITE";
  case Status::kOutputIo:
    return "OUTPUT_IO";
  case Status::kInternalError:
    return "INTERNAL_ERROR";
  case Status::kNonlinearSolveNotConverged:
    return "NONLINEAR_SOLVE_NOT_CONVERGED";
  case Status::kInvalidMaterial:
    return "INVALID_MATERIAL";
  }
  return "INTERNAL_ERROR";
}

} // namespace gpu_femm

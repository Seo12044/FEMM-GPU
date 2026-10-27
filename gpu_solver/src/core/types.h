#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace gpu_femm
{

enum class Status : int {
  kOk = 0,
  kInputIo = 1,
  kInvalidArgument = 2,
  kUnsupportedFeature = 3,
  kMeshInvalid = 4,
  kBoundaryInvalid = 5,
  kGpuUnavailable = 6,
  kGpuAllocationFailed = 7,
  kAssemblyFailed = 8,
  kLinearSolveNotConverged = 9,
  kLinearSolveBreakdown = 10,
  kNumericalNonfinite = 11,
  kOutputIo = 12,
  kInternalError = 13,
  kNonlinearSolveNotConverged = 14,
  kInvalidMaterial = 15,
};

const char *StatusName(Status status);

struct Node {
  double x_m;
  double y_m;
};

struct Triangle {
  int32_t node[3];
  double reluctivity_m_per_h;
  double source_j_per_a;
};

struct Model {
  std::vector<Node> nodes;
  std::vector<Triangle> triangles;
  std::vector<int32_t> dirichlet_nodes;
  std::vector<double> dirichlet_a_wb_per_m;
  double depth_m = 0.0;
};

struct SolveInfo {
  Status status = Status::kInternalError;
  int iterations = 0;
  double residual_l2 = std::numeric_limits<double>::infinity();
};

struct SolveResult {
  SolveInfo info;
  std::vector<double> a_wb_per_m;
  std::vector<double> bx_t;
  std::vector<double> by_t;
  double flux_linkage_wb = std::numeric_limits<double>::quiet_NaN();
};

} // namespace gpu_femm

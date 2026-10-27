#include "commands.h"
#include "io/mesh_artifact.h"

#include <iostream>

namespace gpu_femm
{

int GpuFemmMeshArtifactTest(const std::string &path)
{
  GpuFemmMeshArtifact artifact;
  std::string error;
  if (!ReadGpuFemmMeshArtifact(path, &artifact, &error)) {
    std::cerr << "FAIL gpu_femm_mesh_v1: " << error << '\n';
    return 1;
  }
  std::cout << "PASS " << artifact.schema_version << "\n"
            << "  nodes=" << artifact.model.nodes.size()
            << " triangles=" << artifact.model.triangles.size()
            << " circuits=" << artifact.model.circuit_count
            << " source_fem_sha256=" << artifact.pose_fem_sha256;
  if (artifact.schema_version == "gpu_femm_mesh_v1" ||
      artifact.schema_version == "gpu_femm_mesh_v2" ||
      artifact.schema_version == "gpu_femm_mesh_v3")
    std::cout << " base_motor_fem_sha256=" << artifact.base_motor_fem_sha256;
  std::cout << '\n';
  return 0;
}

} // namespace gpu_femm

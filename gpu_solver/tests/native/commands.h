#pragma once

#include "solvers/options.h"

#include <string>

namespace gpu_femm
{

int SelfTest();
int PcgRegressionTests(int side, PcgPreconditioner mode = PcgPreconditioner::kJacobi);
int BlockPreconditionerTests(bool repeat_benchmark = true);
int NonlinearToleranceTests();
int PcgSpmvLayoutTests();
int CudssRegressionTests();
int GpuFemmMeshArtifactTest(const std::string &path);
int FemmReferenceTest(const std::string &stem);
int NonlinearFemmReferenceTest(const std::string &stem, const std::string &curve_directory);
int FrozenPostprocessReferenceTest(const std::string &stem, const std::string &curve_directory);
int SingleSampleAdapter(const std::string &request_path, const std::string &response_path);

} // namespace gpu_femm

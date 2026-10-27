#pragma once

namespace gpu_femm
{

// Return assertion failures; SelfTest owns the final process status and summary.
int ProtocolContractTests();
int PhysicsRegressionTests();

} // namespace gpu_femm

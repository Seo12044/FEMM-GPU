#pragma once

#include "core/types.h"
#include "physics/nonlinear_model.h"

namespace gpu_femm
{

Model UnitSquareFixture();
Model LargeStructuredLinearFixture(int cells_per_side);
NonlinearModel LargeStructuredMaskFixture(int cells_per_side);
NonlinearModel NonlinearThreeRegionFixture();
NonlinearModel NonlinearTwoCircuitFixture();
NonlinearModel PostprocessRingFixture();

} // namespace gpu_femm

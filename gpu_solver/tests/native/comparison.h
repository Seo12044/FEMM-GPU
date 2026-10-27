#pragma once

#include "core/types.h"

namespace gpu_femm
{

bool Near(double actual, double expected, double tolerance = 1e-12);
bool NearMixed(double actual, double expected, double absolute_tolerance,
               double relative_tolerance);
bool SameBits(const SolveResult &left, const SolveResult &right);

} // namespace gpu_femm

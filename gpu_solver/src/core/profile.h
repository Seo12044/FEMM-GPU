#pragma once

#include <chrono>

namespace gpu_femm
{

using ProfileClock = std::chrono::steady_clock;

double ProfileSecondsSince(const ProfileClock::time_point &start);

} // namespace gpu_femm

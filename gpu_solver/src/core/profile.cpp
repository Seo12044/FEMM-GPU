#include "profile.h"

namespace gpu_femm
{

double ProfileSecondsSince(const ProfileClock::time_point &start)
{
  return std::chrono::duration<double>(ProfileClock::now() - start).count();
}

} // namespace gpu_femm

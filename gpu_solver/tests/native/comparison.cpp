#include "comparison.h"

#include <cmath>
#include <cstring>

namespace gpu_femm
{

bool NearMixed(double actual, double expected, double absolute_tolerance, double relative_tolerance)
{
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <=
             absolute_tolerance + relative_tolerance * std::abs(expected);
}

bool Near(double actual, double expected, double tolerance)
{
  return std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
}

bool SameBits(const SolveResult &left, const SolveResult &right)
{
  if (left.info.status != right.info.status || left.info.iterations != right.info.iterations ||
      std::memcmp(&left.info.residual_l2, &right.info.residual_l2, sizeof(double)) != 0 ||
      std::memcmp(&left.flux_linkage_wb, &right.flux_linkage_wb, sizeof(double)) != 0 ||
      left.a_wb_per_m.size() != right.a_wb_per_m.size() || left.bx_t.size() != right.bx_t.size() ||
      left.by_t.size() != right.by_t.size()) {
    return false;
  }
  return std::memcmp(left.a_wb_per_m.data(), right.a_wb_per_m.data(),
                     left.a_wb_per_m.size() * sizeof(double)) == 0 &&
         std::memcmp(left.bx_t.data(), right.bx_t.data(), left.bx_t.size() * sizeof(double)) == 0 &&
         std::memcmp(left.by_t.data(), right.by_t.data(), left.by_t.size() * sizeof(double)) == 0;
}

} // namespace gpu_femm

#include "native_airgap_matrix.h"

#include <cmath>

bool BuildNativeFemmAirGapMatrix(double K, double Ki, double ci, double co,
    double matrix[10][10])
{
  if (!(K > 0.0) || !(Ki > 0.0) || !std::isfinite(K) || !std::isfinite(Ki)
      || !std::isfinite(ci) || !std::isfinite(co)) return false;
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j) matrix[i][j] = 0.0;
  double (*MG)[10] = matrix;
  const auto Power = [](double value, int exponent) {
    double result = 1.0;
    for (int i = 0; i < exponent; ++i) result *= value;
    return result;
  };
#include "gpu_femm_native_age_matrix.inc"
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < i; ++j) matrix[i][j] = matrix[j][i];
  return true;
}

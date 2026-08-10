#pragma once

#include <array>
#include <cmath>

namespace gpu_femm {

// A_phi is interpolated linearly over the r-z triangle.  Each quadrature
// point supplies the cylindrical curl basis
//   Br_i = -dN_i/dz, Bz_i = dN_i/dr + N_i/r.
// Seven-point Dunavant integration keeps the 1/r term accurate near the axis
// while never evaluating directly on r=0. Axis nodes are fixed to A_phi=0.
struct AxisymmetricP1Point {
  double weight_area_m2 = 0.0;
  double radius_m = 0.0;
  double shape[3] = {};
  double br_basis_per_m[3] = {};
  double bz_basis_per_m[3] = {};
};

struct AxisymmetricP1Element {
  double area_m2 = 0.0;
  std::array<AxisymmetricP1Point, 7> points;
};

inline bool BuildAxisymmetricP1Element(const double radius_m[3],
    const double axial_m[3], AxisymmetricP1Element* element)
{
  if (element == nullptr)
    return false;
  const double determinant = (radius_m[1] - radius_m[0]) * (axial_m[2] - axial_m[0])
      - (radius_m[2] - radius_m[0]) * (axial_m[1] - axial_m[0]);
  if (!(determinant > 0.0) || !std::isfinite(determinant))
    return false;
  const double dndr[3] = {
    (axial_m[1] - axial_m[2]) / determinant,
    (axial_m[2] - axial_m[0]) / determinant,
    (axial_m[0] - axial_m[1]) / determinant,
  };
  const double dndz[3] = {
    (radius_m[2] - radius_m[1]) / determinant,
    (radius_m[0] - radius_m[2]) / determinant,
    (radius_m[1] - radius_m[0]) / determinant,
  };
  constexpr double barycentric[7][3] = {
    { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 },
    { 0.059715871789770, 0.470142064105115, 0.470142064105115 },
    { 0.470142064105115, 0.059715871789770, 0.470142064105115 },
    { 0.470142064105115, 0.470142064105115, 0.059715871789770 },
    { 0.797426985353087, 0.101286507323456, 0.101286507323456 },
    { 0.101286507323456, 0.797426985353087, 0.101286507323456 },
    { 0.101286507323456, 0.101286507323456, 0.797426985353087 },
  };
  constexpr double weights[7] = {
    0.225,
    0.132394152788506, 0.132394152788506, 0.132394152788506,
    0.125939180544827, 0.125939180544827, 0.125939180544827,
  };
  AxisymmetricP1Element built;
  built.area_m2 = 0.5 * determinant;
  for (int point = 0; point < 7; ++point) {
    AxisymmetricP1Point& output = built.points[point];
    output.weight_area_m2 = built.area_m2 * weights[point];
    for (int local = 0; local < 3; ++local) {
      output.shape[local] = barycentric[point][local];
      output.radius_m += output.shape[local] * radius_m[local];
    }
    if (!(output.radius_m > 0.0) || !std::isfinite(output.radius_m))
      return false;
    for (int local = 0; local < 3; ++local) {
      output.br_basis_per_m[local] = -dndz[local];
      output.bz_basis_per_m[local] = dndr[local] + output.shape[local] / output.radius_m;
    }
  }
  *element = built;
  return true;
}

inline bool EvaluateAxisymmetricP1Centroid(const double radius_m[3],
    const double axial_m[3], const double nodal_a[3], double* br_t, double* bz_t)
{
  AxisymmetricP1Element element;
  if (br_t == nullptr || bz_t == nullptr
      || !BuildAxisymmetricP1Element(radius_m, axial_m, &element))
    return false;
  *br_t = 0.0;
  *bz_t = 0.0;
  const AxisymmetricP1Point& center = element.points[0];
  for (int local = 0; local < 3; ++local) {
    *br_t += nodal_a[local] * center.br_basis_per_m[local];
    *bz_t += nodal_a[local] * center.bz_basis_per_m[local];
  }
  return std::isfinite(*br_t) && std::isfinite(*bz_t);
}

} // namespace gpu_femm

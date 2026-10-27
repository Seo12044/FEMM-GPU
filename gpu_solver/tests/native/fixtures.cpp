#include "fixtures.h"

namespace gpu_femm
{

Model UnitSquareFixture()
{
  Model model;
  model.nodes = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.5, 0.5}};
  model.triangles = {
      {{0, 1, 4}, 1.0, 1.0}, {{1, 2, 4}, 1.0, 1.0}, {{2, 3, 4}, 1.0, 1.0}, {{3, 0, 4}, 1.0, 1.0}};
  model.dirichlet_nodes = {0, 1, 2, 3};
  model.dirichlet_a_wb_per_m = {0.0, 0.0, 0.0, 0.0};
  model.depth_m = 1.0;
  return model;
}

Model LargeStructuredLinearFixture(int cells_per_side)
{
  Model model;
  const int nodes_per_side = cells_per_side + 1;
  model.nodes.reserve(static_cast<size_t>(nodes_per_side) * nodes_per_side);
  model.triangles.reserve(static_cast<size_t>(cells_per_side) * cells_per_side * 2);
  for (int y = 0; y < nodes_per_side; ++y) {
    for (int x = 0; x < nodes_per_side; ++x) {
      model.nodes.push_back({static_cast<double>(x), static_cast<double>(y)});
      if (x == 0 || y == 0 || x == cells_per_side || y == cells_per_side) {
        model.dirichlet_nodes.push_back(static_cast<int32_t>(y * nodes_per_side + x));
        model.dirichlet_a_wb_per_m.push_back(0.0);
      }
    }
  }
  for (int y = 0; y < cells_per_side; ++y) {
    for (int x = 0; x < cells_per_side; ++x) {
      const int32_t lower_left = static_cast<int32_t>(y * nodes_per_side + x);
      const int32_t lower_right = lower_left + 1;
      const int32_t upper_left = lower_left + nodes_per_side;
      const int32_t upper_right = upper_left + 1;
      model.triangles.push_back({{lower_left, lower_right, upper_right}, 1.0, 0.0});
      model.triangles.push_back({{lower_left, upper_right, upper_left}, 1.0, 0.0});
    }
  }
  model.depth_m = 1.0;
  return model;
}

NonlinearModel LargeStructuredMaskFixture(int cells_per_side)
{
  NonlinearModel model;
  const int nodes_per_side = cells_per_side + 1;
  const int center = cells_per_side / 2;
  const int32_t selected_lower_left = static_cast<int32_t>(center * nodes_per_side + center);
  const int32_t selected_lower_right = selected_lower_left + 1;
  const int32_t selected_upper_right = selected_lower_right + nodes_per_side;
  model.nodes.reserve(static_cast<size_t>(nodes_per_side) * nodes_per_side);
  model.triangles.reserve(static_cast<size_t>(cells_per_side) * cells_per_side * 2);
  // Both regions have the same linear material law; their distinct labels
  // exercise WST selection without changing the analytic stencil.
  model.materials = {{1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0}};
  for (int y = 0; y < nodes_per_side; ++y) {
    for (int x = 0; x < nodes_per_side; ++x) {
      const int32_t node = static_cast<int32_t>(y * nodes_per_side + x);
      model.nodes.push_back({static_cast<double>(x), static_cast<double>(y)});
      if (node != selected_lower_left && node != selected_lower_right &&
          node != selected_upper_right) {
        model.dirichlet_nodes.push_back(node);
        model.dirichlet_a_wb_per_m.push_back(0.0);
      }
    }
  }
  for (int y = 0; y < cells_per_side; ++y) {
    for (int x = 0; x < cells_per_side; ++x) {
      const int32_t lower_left = static_cast<int32_t>(y * nodes_per_side + x);
      const int32_t lower_right = lower_left + 1;
      const int32_t upper_left = lower_left + nodes_per_side;
      const int32_t upper_right = upper_left + 1;
      const int32_t material = lower_left == selected_lower_left ? 1 : 0;
      model.triangles.push_back({{lower_left, lower_right, upper_right}, material});
      model.triangles.push_back({{lower_left, upper_right, upper_left}, 0});
    }
  }
  model.depth_m = 1.0;
  return model;
}

NonlinearModel NonlinearThreeRegionFixture()
{
  NonlinearModel model;
  model.nodes = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.5, 0.5}};
  // PM, driven coil, and nonlinear steel all occur in the same small model.
  model.materials = {
      {1.0, 0.0, 0.0, 10.0, 0.0},
      {1.0, 0.0, 12.0, 0.0, 0.0, -1, 0},
      {0.5, 1.0, 0.0, 0.0, 0.0, 0},
  };
  model.bh_curves = {{{0.0, 1.0, 2.0}, {0.0, 1.0, 4.0}, {}}};
  BuildNonlinearBhSpline(&model.bh_curves[0]);
  model.triangles = {{{0, 1, 4}, 0}, {{1, 2, 4}, 1}, {{2, 3, 4}, 2}, {{3, 0, 4}, 2}};
  model.dirichlet_nodes = {0, 1, 2, 3};
  model.dirichlet_a_wb_per_m = {0.0, 0.0, 0.0, 0.0};
  model.depth_m = 1.0;
  return model;
}

NonlinearModel NonlinearTwoCircuitFixture()
{
  NonlinearModel model;
  model.nodes = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.5, 0.5}};
  model.materials = {
      {1.0, 0.0, 0.0, 0.0, 0.0},
      {1.0, 0.0, 12.0, 0.0, 0.0, -1, 0},
      {1.0, 0.0, 6.0, 0.0, 0.0, -1, 1},
  };
  model.triangles = {{{0, 1, 4}, 1}, {{1, 2, 4}, 0}, {{2, 3, 4}, 2}, {{3, 0, 4}, 0}};
  model.dirichlet_nodes = {0, 1, 2, 3};
  model.dirichlet_a_wb_per_m = {0.0, 0.0, 0.0, 0.0};
  model.circuit_count = 2;
  model.depth_m = 1.0;
  return model;
}

NonlinearModel PostprocessRingFixture()
{
  NonlinearModel model;
  model.nodes = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0},
                 {-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}};
  model.materials = {{1.0 / kMu0, 0.0, 0.0, 0.0, 0.0},
                     {1.0 / kMu0, 0.0, 0.0, 0.0, 0.0},
                     {1.0 / kMu0, 0.0, 0.0, 0.0, 0.0},
                     {1.0 / kMu0, 0.0, 0.0, 0.0, 0.0}};
  model.triangles = {{{0, 1, 5}, 3}, {{0, 5, 4}, 3}, {{1, 2, 6}, 3}, {{1, 6, 5}, 3},
                     {{2, 3, 7}, 3}, {{2, 7, 6}, 3}, {{3, 0, 4}, 3}, {{3, 4, 7}, 3},
                     {{4, 5, 6}, 1}, {{4, 6, 7}, 1}};
  model.dirichlet_nodes = {0, 1, 2, 3};
  model.dirichlet_a_wb_per_m = {-1.0, 1.0, 1.0, -1.0};
  model.depth_m = 1.0;
  return model;
}

} // namespace gpu_femm

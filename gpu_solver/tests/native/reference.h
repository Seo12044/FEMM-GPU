#pragma once

#include "physics/nonlinear_model.h"
#include "postprocess/postprocess.h"

#include <string>
#include <vector>

namespace gpu_femm
{

struct FemmReference {
  Model model;
  std::vector<double> expected_a;
  std::vector<double> expected_bx;
  std::vector<double> expected_by;
  double current_a = 0.0;
  double flux_linkage_wb = 0.0;
  Assembly dump;
  std::vector<double> dump_rhs;
};

struct NonlinearFemmReference {
  NonlinearModel model;
  std::vector<double> expected_a;
  std::vector<double> expected_bx;
  std::vector<double> expected_by;
  double current_a = 0.0;
  double flux_linkage_wb = 0.0;
};

struct FrozenPostprocessReference {
  FrozenPostprocessOptions options;
  int32_t expected_node_count = 0;
  int32_t expected_element_count = 0;
  double expected_current_a = std::numeric_limits<double>::quiet_NaN();
  double expected_flux_linkage_wb = std::numeric_limits<double>::quiet_NaN();
  double expected_force_x_n = std::numeric_limits<double>::quiet_NaN();
  double expected_force_y_n = std::numeric_limits<double>::quiet_NaN();
  double expected_torque_nm = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> expected_radial_b_t;
};

bool LoadFemmReference(const std::string &stem, FemmReference *reference, std::string *error);
bool ReadNonlinearFemmReference(const std::string &stem, const std::string &curve_directory,
                                NonlinearFemmReference *reference, std::string *error);
bool ReadFrozenPostprocessReference(const std::string &stem, FrozenPostprocessReference *reference,
                                    std::string *error);

} // namespace gpu_femm

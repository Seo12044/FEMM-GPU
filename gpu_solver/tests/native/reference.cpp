#include "reference.h"
#include "comparison.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <utility>

namespace gpu_femm
{

bool ReadFemmAns(const std::string &path, FemmReference *reference, std::string *error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::string line;
  bool found_solution = false;
  while (std::getline(input, line)) {
    if (line == "[Solution]" || line == "[Solution]\r") {
      found_solution = true;
      break;
    }
  }
  int node_count = 0;
  if (!found_solution || !(input >> node_count) || node_count <= 0) {
    *error = "invalid [Solution] node count";
    return false;
  }
  reference->model.nodes.resize(node_count);
  reference->expected_a.resize(node_count);
  for (int node = 0; node < node_count; ++node) {
    int boundary_code = 0;
    if (!(input >> reference->model.nodes[node].x_m >> reference->model.nodes[node].y_m >>
          reference->expected_a[node] >> boundary_code) ||
        !std::isfinite(reference->expected_a[node])) {
      *error = "invalid FEMM solution node";
      return false;
    }
  }
  int element_count = 0;
  if (!(input >> element_count) || element_count <= 0) {
    *error = "invalid FEMM solution element count";
    return false;
  }
  reference->model.triangles.resize(element_count);
  std::vector<bool> boundary(node_count, false);
  for (int element = 0; element < element_count; ++element) {
    int label = 0;
    int edge[3] = {-1, -1, -1};
    double element_current = 0.0;
    Triangle &triangle = reference->model.triangles[element];
    if (!(input >> triangle.node[0] >> triangle.node[1] >> triangle.node[2] >> label >> edge[0] >>
          edge[1] >> edge[2] >> element_current) ||
        label != 0 || !std::isfinite(element_current)) {
      *error = "invalid or unsupported FEMM solution element";
      return false;
    }
    triangle.reluctivity_m_per_h = 1.0 / kMu0;
    triangle.source_j_per_a = 1.0;
    for (int side = 0; side < 3; ++side) {
      if (edge[side] >= 0) {
        const int first = triangle.node[side];
        const int second = triangle.node[(side + 1) % 3];
        if (first < 0 || first >= node_count || second < 0 || second >= node_count) {
          *error = "FEMM boundary node out of range";
          return false;
        }
        boundary[first] = true;
        boundary[second] = true;
      }
    }
  }
  for (int node = 0; node < node_count; ++node) {
    if (boundary[node]) {
      if (reference->expected_a[node] != 0.0) {
        *error = "fixture requires zero-A outer boundary";
        return false;
      }
      reference->model.dirichlet_nodes.push_back(node);
      reference->model.dirichlet_a_wb_per_m.push_back(0.0);
    }
  }
  reference->model.depth_m = 1.0;
  if (ValidateModel(reference->model) != Status::kOk) {
    *error = "FEMM mesh failed frozen-scope validation";
    return false;
  }

  reference->expected_bx.resize(element_count);
  reference->expected_by.resize(element_count);
  double area_sum = 0.0;
  for (int element = 0; element < element_count; ++element) {
    const Triangle &triangle = reference->model.triangles[element];
    const Node p0 = reference->model.nodes[triangle.node[0]];
    const Node p1 = reference->model.nodes[triangle.node[1]];
    const Node p2 = reference->model.nodes[triangle.node[2]];
    const double determinant =
        (p1.x_m - p0.x_m) * (p2.y_m - p0.y_m) - (p2.x_m - p0.x_m) * (p1.y_m - p0.y_m);
    area_sum += 0.5 * determinant;
    const double a0 = reference->expected_a[triangle.node[0]];
    const double a1 = reference->expected_a[triangle.node[1]];
    const double a2 = reference->expected_a[triangle.node[2]];
    reference->expected_bx[element] =
        (a0 * (p2.x_m - p1.x_m) + a1 * (p0.x_m - p2.x_m) + a2 * (p1.x_m - p0.x_m)) / determinant;
    reference->expected_by[element] =
        -(a0 * (p1.y_m - p2.y_m) + a1 * (p2.y_m - p0.y_m) + a2 * (p0.y_m - p1.y_m)) / determinant;
  }
  if (!NearMixed(area_sum, 1.0, 1e-12, 1e-12)) {
    *error = "fixture source normalization requires one square metre";
    return false;
  }
  return true;
}

bool ReadCircuitReference(const std::string &path, FemmReference *reference, std::string *error)
{
  std::ifstream input(path);
  std::string current_name;
  std::string voltage_name;
  std::string flux_name;
  double voltage = 0.0;
  if (!input ||
      !(input >> current_name >> reference->current_a >> voltage_name >> voltage >> flux_name >>
        reference->flux_linkage_wb) ||
      current_name != "current_A" || voltage_name != "voltage_drop_V" ||
      flux_name != "flux_linkage_Wb" || reference->current_a != 12.0 || voltage != 0.0 ||
      !std::isfinite(reference->flux_linkage_wb)) {
    *error = "invalid frozen circuit reference";
    return false;
  }
  return true;
}

bool ParseKeyValueDouble(const std::string &line, const std::string &key, double *value)
{
  const size_t position = line.find(key + "=");
  if (position == std::string::npos)
    return false;
  try {
    size_t parsed = 0;
    *value = std::stod(line.substr(position + key.size() + 1), &parsed);
    return parsed > 0 && std::isfinite(*value);
  } catch (...) {
    return false;
  }
}

bool ReadFrozenPostprocessReference(const std::string &stem, FrozenPostprocessReference *reference,
                                    std::string *error)
{
  std::ifstream input(stem + ".postprocess.txt");
  if (!input) {
    *error = "cannot open " + stem + ".postprocess.txt";
    return false;
  }
  std::string line;
  bool have_selected = false, have_nodes = false, have_elements = false;
  bool have_current = false, have_flux = false, have_fx = false, have_fy = false;
  bool have_torque = false, have_radius = false;
  while (std::getline(input, line)) {
    double value = 0.0;
    if (ParseKeyValueDouble(line, "selected_material_label", &value)) {
      const int32_t label = static_cast<int32_t>(value);
      if (value != static_cast<double>(label)) {
        *error = "non-integral selected material label";
        return false;
      }
      reference->options.selected_material_label = label;
      have_selected = true;
    } else if (ParseKeyValueDouble(line, "node_count", &value)) {
      reference->expected_node_count = static_cast<int32_t>(value);
      have_nodes = value == static_cast<double>(reference->expected_node_count) &&
                   reference->expected_node_count > 0;
    } else if (ParseKeyValueDouble(line, "element_count", &value)) {
      reference->expected_element_count = static_cast<int32_t>(value);
      have_elements = value == static_cast<double>(reference->expected_element_count) &&
                      reference->expected_element_count > 0;
    } else if (ParseKeyValueDouble(line, "circuit_current_A", &value)) {
      reference->expected_current_a = value;
      have_current = true;
    } else if (ParseKeyValueDouble(line, "flux_linkage_Wb", &value)) {
      reference->expected_flux_linkage_wb = value;
      have_flux = true;
    } else if (ParseKeyValueDouble(line, "air_material_label", &value)) {
      const int32_t label = static_cast<int32_t>(value);
      if (value != static_cast<double>(label)) {
        *error = "non-integral air material label";
        return false;
      }
      reference->options.air_material_label = label;
    } else if (ParseKeyValueDouble(line, "Fx_N", &value)) {
      reference->expected_force_x_n = value;
      have_fx = true;
    } else if (ParseKeyValueDouble(line, "Fy_N", &value)) {
      reference->expected_force_y_n = value;
      have_fy = true;
    } else if (ParseKeyValueDouble(line, "torque_Nm", &value)) {
      reference->expected_torque_nm = value;
      have_torque = true;
    } else if (ParseKeyValueDouble(line, "airgap_radius_mm", &value)) {
      reference->options.airgap_radius_m = value * 1e-3;
      have_radius = true;
    } else if (line.find("airgap_sample_deg=") != std::string::npos) {
      double angle_deg = 0.0, radial_b = 0.0;
      if (!ParseKeyValueDouble(line, "airgap_sample_deg", &angle_deg) ||
          !ParseKeyValueDouble(line, "radial_B_T", &radial_b)) {
        *error = "invalid airgap sample line";
        return false;
      }
      reference->options.airgap_angles_rad.push_back(
          angle_deg * 3.141592653589793238462643383279502884 / 180.0);
      reference->expected_radial_b_t.push_back(radial_b);
    }
  }
  if (!have_selected || !have_nodes || !have_elements || !have_current || !have_flux || !have_fx ||
      !have_fy || !have_torque || !have_radius || reference->options.airgap_angles_rad.empty() ||
      reference->options.airgap_angles_rad.size() != reference->expected_radial_b_t.size()) {
    *error = "incomplete frozen postprocess reference";
    return false;
  }
  return true;
}

bool ReadNonlinearFemmReference(const std::string &stem, const std::string &curve_directory,
                                NonlinearFemmReference *reference, std::string *error)
{
  std::ifstream input(stem + ".ans");
  std::string line;
  while (std::getline(input, line) && line != "[Solution]") {
  }
  int node_count = 0;
  if (!input || !(input >> node_count) || node_count <= 0) {
    *error = "invalid nonlinear [Solution] node count";
    return false;
  }
  reference->model.nodes.resize(node_count);
  reference->expected_a.resize(node_count);
  for (int node = 0; node < node_count; ++node) {
    int ignored = 0;
    if (!(input >> reference->model.nodes[node].x_m >> reference->model.nodes[node].y_m >>
          reference->expected_a[node] >> ignored)) {
      *error = "invalid nonlinear solution node";
      return false;
    }
    reference->model.nodes[node].x_m *= 1e-3;
    reference->model.nodes[node].y_m *= 1e-3;
  }
  int element_count = 0;
  if (!(input >> element_count) || element_count <= 0) {
    *error = "invalid nonlinear solution element count";
    return false;
  }
  reference->model.triangles.resize(element_count);
  std::vector<int> label(element_count, -1);
  bool label_seen[4] = {false, false, false, false};
  std::vector<bool> boundary(node_count, false);
  for (int element = 0; element < element_count; ++element) {
    int edge[3] = {-1, -1, -1};
    double element_current = 0.0;
    NonlinearTriangle &triangle = reference->model.triangles[element];
    if (!(input >> triangle.node[0] >> triangle.node[1] >> triangle.node[2] >> label[element] >>
          edge[0] >> edge[1] >> edge[2] >> element_current) ||
        label[element] < 0 || label[element] > 3 || !std::isfinite(element_current)) {
      *error = "missing or out-of-range nonlinear block label";
      return false;
    }
    triangle.material = label[element];
    label_seen[label[element]] = true;
    for (int side = 0; side < 3; ++side) {
      if (edge[side] >= 0) {
        const int first = triangle.node[side];
        const int second = triangle.node[(side + 1) % 3];
        if (first < 0 || first >= node_count || second < 0 || second >= node_count) {
          *error = "nonlinear boundary node out of range";
          return false;
        }
        boundary[first] = boundary[second] = true;
      }
    }
  }
  for (bool seen : label_seen) {
    if (!seen) {
      *error = "nonlinear fixture is missing an expected material region";
      return false;
    }
  }
  double coil_area = 0.0;
  for (int element = 0; element < element_count; ++element) {
    const NonlinearTriangle &triangle = reference->model.triangles[element];
    Node p[3] = {reference->model.nodes[triangle.node[0]], reference->model.nodes[triangle.node[1]],
                 reference->model.nodes[triangle.node[2]]};
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms)) {
      *error = "invalid nonlinear element geometry";
      return false;
    }
    if (label[element] == 2)
      coil_area += terms.area_m2;
  }
  if (!(coil_area > 0.0)) {
    *error = "nonlinear fixture has no coil material";
    return false;
  }
  NonlinearBhCurve steel_curve;
  if (!ReadNonlinearBhTable(curve_directory + "/35PN230.tab", &steel_curve)) {
    *error = "cannot read 35PN230 B-H table";
    return false;
  }
  reference->model.bh_curves = {std::move(steel_curve)};
  reference->model.materials = {
      {1.0, 0.0, 0.0, 0.0, 0.0, 0},                         // label 0: Steel35PN230
      {1.0 / (kMu0 * 1.05), 0.0, 0.0, 900000.0, 0.0},       // label 1: PM
      {1.0 / kMu0, 0.0, 50.0 / coil_area, 0.0, 0.0, -1, 0}, // label 2: driven coil-air
      {1.0 / kMu0, 0.0, 0.0, 0.0, 0.0},                     // label 3: default air
  };
  reference->model.depth_m = 0.020;
  for (int node = 0; node < node_count; ++node) {
    if (boundary[node]) {
      if (reference->expected_a[node] != 0.0) {
        *error = "fixture requires zero-A outer boundary";
        return false;
      }
      reference->model.dirichlet_nodes.push_back(node);
      reference->model.dirichlet_a_wb_per_m.push_back(0.0);
    }
  }
  if (ValidateNonlinearModel(reference->model) != Status::kOk) {
    *error = "nonlinear frozen mesh validation failed";
    return false;
  }
  std::ifstream summary(stem + ".summary.txt");
  std::string key;
  while (summary >> key) {
    if (key == "circuit_current_A=")
      summary >> reference->current_a;
    else if (key == "flux_linkage_Wb=")
      summary >> reference->flux_linkage_wb;
    else
      summary.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  // Summary is key=value, not whitespace-separated; parse it linewise when needed.
  if (!std::isfinite(reference->current_a) || reference->current_a == 0.0 ||
      !std::isfinite(reference->flux_linkage_wb) || reference->flux_linkage_wb == 0.0) {
    summary.clear();
    summary.seekg(0);
    while (std::getline(summary, line)) {
      const size_t equals = line.find('=');
      if (equals == std::string::npos)
        continue;
      if (line.substr(0, equals) == "circuit_current_A")
        reference->current_a = std::stod(line.substr(equals + 1));
      if (line.substr(0, equals) == "flux_linkage_Wb")
        reference->flux_linkage_wb = std::stod(line.substr(equals + 1));
    }
  }
  if (!std::isfinite(reference->current_a) || !std::isfinite(reference->flux_linkage_wb)) {
    *error = "invalid nonlinear fixture summary";
    return false;
  }
  reference->expected_bx.resize(element_count);
  reference->expected_by.resize(element_count);
  for (int element = 0; element < element_count; ++element) {
    const NonlinearTriangle &triangle = reference->model.triangles[element];
    Node p[3] = {reference->model.nodes[triangle.node[0]], reference->model.nodes[triangle.node[1]],
                 reference->model.nodes[triangle.node[2]]};
    NonlinearElementTerms terms;
    BuildElementTerms(p, &terms);
    for (int i = 0; i < 3; ++i) {
      reference->expected_bx[element] += reference->expected_a[triangle.node[i]] * terms.b_x[i];
      reference->expected_by[element] += reference->expected_a[triangle.node[i]] * terms.b_y[i];
    }
  }
  return true;
}

bool ReadDumpRhs(const std::string &path, std::vector<double> *rhs, std::string *error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::string line;
  bool reading = false;
  while (std::getline(input, line)) {
    if (!reading) {
      const size_t start = line.find("b = [");
      if (start == std::string::npos)
        continue;
      line = line.substr(start + 5);
      reading = true;
    }
    const bool end = line.find("];") != std::string::npos;
    line.erase(std::remove(line.begin(), line.end(), ';'), line.end());
    line.erase(std::remove(line.begin(), line.end(), ']'), line.end());
    std::istringstream value_input(line);
    double value = 0.0;
    if (value_input >> value)
      rhs->push_back(value);
    if (end)
      break;
  }
  if (!reading || rhs->empty()) {
    *error = "invalid FEMM dump RHS";
    return false;
  }
  return true;
}

bool ReadDumpMatrix(const std::string &path, size_t dimension, Assembly *dump, std::string *error)
{
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open " + path;
    return false;
  }
  std::map<std::pair<int, int>, std::vector<double>> entries;
  int one_based_row = 0;
  int one_based_column = 0;
  double value = 0.0;
  while (input >> one_based_row >> one_based_column >> value) {
    const int row = one_based_row - 1;
    const int column = one_based_column - 1;
    if (row < 0 || column < 0 || static_cast<size_t>(row) >= dimension ||
        static_cast<size_t>(column) >= dimension || !std::isfinite(value)) {
      *error = "invalid FEMM dump matrix entry";
      return false;
    }
    entries[{std::min(row, column), std::max(row, column)}].push_back(value);
  }
  std::vector<std::map<int, double>> rows(dimension);
  for (const auto &item : entries) {
    const std::vector<double> &duplicates = item.second;
    if (duplicates.size() != 2 || duplicates[0] != duplicates[1]) {
      *error = "FEMM dump duplicate/mirror invariant failed";
      return false;
    }
    const int row = item.first.first;
    const int column = item.first.second;
    rows[row][column] = duplicates[0];
    if (row != column)
      rows[column][row] = duplicates[0];
  }
  dump->row_offsets.assign(dimension + 1, 0);
  dump->diagonal.assign(dimension, 0.0);
  dump->free_nodes.resize(dimension);
  for (size_t row = 0; row < dimension; ++row) {
    dump->free_nodes[row] = static_cast<int32_t>(row);
    dump->row_offsets[row] = static_cast<int32_t>(dump->values.size());
    for (const auto &entry : rows[row]) {
      dump->column_indices.push_back(entry.first);
      dump->values.push_back(entry.second);
      if (entry.first == static_cast<int>(row))
        dump->diagonal[row] = entry.second;
    }
    if (!(dump->diagonal[row] > 0.0)) {
      *error = "FEMM dump missing positive diagonal";
      return false;
    }
  }
  dump->row_offsets[dimension] = static_cast<int32_t>(dump->values.size());
  return true;
}

bool LoadFemmReference(const std::string &stem, FemmReference *reference, std::string *error)
{
  return ReadFemmAns(stem + ".ans", reference, error) &&
         ReadCircuitReference(stem + ".circuit.txt", reference, error) &&
         ReadDumpRhs(stem + ".m", &reference->dump_rhs, error) &&
         reference->dump_rhs.size() == reference->expected_a.size() &&
         ReadDumpMatrix(stem + ".dat", reference->dump_rhs.size(), &reference->dump, error);
}

} // namespace gpu_femm

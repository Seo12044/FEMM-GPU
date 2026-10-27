#include "postprocess.h"

#include <algorithm>
#include <cmath>

namespace gpu_femm
{

EdgeKey CanonicalEdge(int32_t first, int32_t second)
{
  return {std::min(first, second), std::max(first, second)};
}

bool TriangleBarycentric(const Node p[3], double x_m, double y_m, double lambda[3])
{
  const double determinant =
      (p[1].x_m - p[0].x_m) * (p[2].y_m - p[0].y_m) - (p[2].x_m - p[0].x_m) * (p[1].y_m - p[0].y_m);
  if (!(determinant > 0.0) || !std::isfinite(determinant))
    return false;
  lambda[1] =
      ((x_m - p[0].x_m) * (p[2].y_m - p[0].y_m) - (p[2].x_m - p[0].x_m) * (y_m - p[0].y_m)) /
      determinant;
  lambda[2] =
      ((p[1].x_m - p[0].x_m) * (y_m - p[0].y_m) - (x_m - p[0].x_m) * (p[1].y_m - p[0].y_m)) /
      determinant;
  lambda[0] = 1.0 - lambda[1] - lambda[2];
  return std::isfinite(lambda[0]) && std::isfinite(lambda[1]) && std::isfinite(lambda[2]);
}

// Exact DC subset of FEMM GetNodalB: first use inverse-centroid-distance
// averaging in an equivalent-material patch.  Otherwise traverse CCW/CW from
// the element around the node, preserve Bn from A and Bt from the source-side
// element at each interface, then use FEMM's sharp-corner fallback.
bool SmoothedElementB(const NonlinearModel &model, const std::vector<double> &a,
                      const std::vector<double> &bx, const std::vector<double> &by,
                      const std::vector<std::vector<int32_t>> &incident,
                      const std::map<EdgeKey, std::vector<int32_t>> &edge_elements, int32_t element,
                      double nodal_bx[3], double nodal_by[3])
{
  if (element < 0 || static_cast<size_t>(element) >= model.triangles.size() ||
      a.size() != model.nodes.size() || bx.size() != model.triangles.size() ||
      by.size() != model.triangles.size())
    return false;
  (void)edge_elements;
  const NonlinearTriangle &triangle = model.triangles[element];
  for (int local = 0; local < 3; ++local) {
    const int32_t node = triangle.node[local];
    bool homogeneous = true;
    for (const int32_t near : incident[node])
      // FEMM switches to its interface traversal at a block-label boundary,
      // even when the two labels happen to reference equivalent materials.
      homogeneous =
          homogeneous && model.triangles[element].material == model.triangles[near].material;
    if (homogeneous) {
      double weight_sum = 0.0;
      nodal_bx[local] = 0.0;
      nodal_by[local] = 0.0;
      for (const int32_t near : incident[node]) {
        const NonlinearTriangle &candidate = model.triangles[near];
        const Node &p0 = model.nodes[candidate.node[0]];
        const Node &p1 = model.nodes[candidate.node[1]];
        const Node &p2 = model.nodes[candidate.node[2]];
        const double cx = (p0.x_m + p1.x_m + p2.x_m) / 3.0;
        const double cy = (p0.y_m + p1.y_m + p2.y_m) / 3.0;
        const double weight =
            1.0 / std::hypot(model.nodes[node].x_m - cx, model.nodes[node].y_m - cy);
        if (!(weight > 0.0) || !std::isfinite(weight))
          return false;
        weight_sum += weight;
        nodal_bx[local] += weight * bx[near];
        nodal_by[local] += weight * by[near];
      }
      nodal_bx[local] /= weight_sum;
      nodal_by[local] /= weight_sum;
      continue;
    }

    auto local_index = [&model, node](int32_t candidate) {
      for (int index = 0; index < 3; ++index)
        if (model.triangles[candidate].node[index] == node)
          return index;
      return -1;
    };
    auto next_across = [&model, &incident, node](int32_t current, int32_t other) {
      for (const int32_t candidate : incident[node]) {
        if (candidate == current)
          continue;
        for (int index = 0; index < 3; ++index)
          if (model.triangles[candidate].node[index] == other)
            return candidate;
      }
      return static_cast<int32_t>(-1);
    };
    double weight_sum = 0.0;
    nodal_bx[local] = 0.0;
    nodal_by[local] = 0.0;
    double v1x = 0.0, v1y = 0.0, v2x = 0.0, v2y = 0.0;
    bool special_case = false;
    auto add_interface = [&](int32_t source, int32_t other) {
      const double dx = model.nodes[other].x_m - model.nodes[node].x_m;
      const double dy = model.nodes[other].y_m - model.nodes[node].y_m;
      const double length = std::hypot(dx, dy);
      if (!(length > 0.0))
        return false;
      const double tx = dx / length;
      const double ty = dy / length;
      const double bt = bx[source] * tx + by[source] * ty;
      const double bn = (a[other] - a[node]) / length;
      const double weight = 0.5 / length;
      nodal_bx[local] += weight * (tx * bt + ty * bn);
      nodal_by[local] += weight * (ty * bt - tx * bn);
      weight_sum += weight;
      return true;
    };
    int32_t current = element;
    for (size_t scan = 0; scan < incident[node].size(); ++scan) {
      const int index = local_index(current);
      if (index < 0)
        return false;
      const int32_t other = model.triangles[current].node[(index + 2) % 3];
      const int32_t next = next_across(current, other);
      if (next < 0) {
        nodal_bx[local] = bx[current];
        nodal_by[local] = by[current];
        special_case = true;
        break;
      }
      if (model.triangles[element].material != model.triangles[next].material) {
        if (!add_interface(current, other))
          return false;
        const double length = std::hypot(model.nodes[other].x_m - model.nodes[node].x_m,
                                         model.nodes[other].y_m - model.nodes[node].y_m);
        v1x = (model.nodes[other].x_m - model.nodes[node].x_m) / length;
        v1y = (model.nodes[other].y_m - model.nodes[node].y_m) / length;
        break;
      }
      current = next;
    }
    if (!special_case) {
      current = element;
      for (size_t scan = 0; scan < incident[node].size(); ++scan) {
        const int index = local_index(current);
        if (index < 0)
          return false;
        const int32_t other = model.triangles[current].node[(index + 1) % 3];
        const int32_t next = next_across(current, other);
        if (next < 0) {
          nodal_bx[local] = bx[current];
          nodal_by[local] = by[current];
          special_case = true;
          break;
        }
        if (model.triangles[element].material != model.triangles[next].material) {
          if (!add_interface(current, other))
            return false;
          const double length = std::hypot(model.nodes[other].x_m - model.nodes[node].x_m,
                                           model.nodes[other].y_m - model.nodes[node].y_m);
          v2x = (model.nodes[other].x_m - model.nodes[node].x_m) / length;
          v2y = (model.nodes[other].y_m - model.nodes[node].y_m) / length;
          break;
        }
        current = next;
      }
    }
    if (!special_case) {
      if (!(weight_sum > 0.0))
        return false;
      nodal_bx[local] /= weight_sum;
      nodal_by[local] /= weight_sum;
      const bool simple_corner = std::hypot(v1x, v1y) < 0.9 || std::hypot(v2x, v2y) < 0.9 ||
                                 (-v1x * v2x - v1y * v2y) > 0.985;
      if (!simple_corner) {
        double magnitude = 0.0;
        for (const int32_t near : incident[node]) {
          if (model.triangles[near].material == triangle.material)
            magnitude = std::max(magnitude, std::hypot(bx[near], by[near]));
        }
        const double own_magnitude = std::hypot(bx[element], by[element]);
        if (own_magnitude > 0.0) {
          nodal_bx[local] = magnitude * bx[element] / own_magnitude;
          nodal_by[local] = magnitude * by[element] / own_magnitude;
        } else {
          nodal_bx[local] = 0.0;
          nodal_by[local] = 0.0;
        }
      }
    }
  }
  return true;
}

bool FrozenPostprocessPlanMatches(const FrozenPostprocessPlan &plan, const NonlinearModel &model,
                                  const FrozenPostprocessOptions &options)
{
  return plan.status == Status::kOk && plan.node_count == model.nodes.size() &&
         plan.triangle_count == model.triangles.size() &&
         plan.selected_group_number == options.selected_group_number &&
         plan.air_group_number == options.air_group_number &&
         plan.selected_material_label == options.selected_material_label &&
         plan.air_material_label == options.air_material_label &&
         plan.airgap_radius_m == options.airgap_radius_m &&
         plan.airgap_angles_rad == options.airgap_angles_rad;
}

Status BuildFrozenPostprocessPlan(const NonlinearModel &model,
                                  const FrozenPostprocessOptions &options,
                                  FrozenPostprocessPlan *plan)
{
  if (!plan)
    return Status::kInvalidArgument;
  std::vector<double> mask;
  const Status status = BuildWeightedStressMask(model, options, &mask);
  if (status != Status::kOk) {
    *plan = FrozenPostprocessPlan{};
    plan->status = status;
    return status;
  }
  return BuildFrozenPostprocessPlanWithMask(model, options, std::move(mask), plan);
}

Status BuildFrozenPostprocessPlanWithMask(const NonlinearModel &model,
                                          const FrozenPostprocessOptions &options,
                                          std::vector<double> mask, FrozenPostprocessPlan *plan)
{
  if (plan == nullptr)
    return Status::kInvalidArgument;
  *plan = FrozenPostprocessPlan{};
  plan->node_count = model.nodes.size();
  plan->triangle_count = model.triangles.size();
  plan->selected_group_number = options.selected_group_number;
  plan->air_group_number = options.air_group_number;
  plan->selected_material_label = options.selected_material_label;
  plan->air_material_label = options.air_material_label;
  plan->airgap_radius_m = options.airgap_radius_m;
  plan->airgap_angles_rad = options.airgap_angles_rad;
  if (mask.size() != model.nodes.size() ||
      !std::all_of(mask.begin(), mask.end(), [](double v) { return v == 0 || v == 1; })) {
    plan->status = Status::kInvalidArgument;
    return plan->status;
  }
  plan->mask = std::move(mask);
  plan->incident.assign(model.nodes.size(), {});
  plan->mask_gradient_x.resize(model.triangles.size());
  plan->mask_gradient_y.resize(model.triangles.size());
  plan->element_weight.resize(model.triangles.size());
  plan->element_center_x.resize(model.triangles.size());
  plan->element_center_y.resize(model.triangles.size());
  for (size_t element = 0; element < model.triangles.size(); ++element) {
    const NonlinearTriangle &triangle = model.triangles[element];
    for (int local = 0; local < 3; ++local) {
      plan->incident[triangle.node[local]].push_back(static_cast<int32_t>(element));
      plan->edge_elements[CanonicalEdge(triangle.node[local], triangle.node[(local + 1) % 3])]
          .push_back(static_cast<int32_t>(element));
    }
    Node p[3] = {model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
                 model.nodes[triangle.node[2]]};
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms)) {
      plan->status = Status::kMeshInvalid;
      return plan->status;
    }
    for (int local = 0; local < 3; ++local) {
      plan->mask_gradient_x[element] += plan->mask[triangle.node[local]] * terms.b_y[local];
      plan->mask_gradient_y[element] -= plan->mask[triangle.node[local]] * terms.b_x[local];
    }
    plan->element_weight[element] = terms.area_m2 * model.depth_m;
    plan->element_center_x[element] = (p[0].x_m + p[1].x_m + p[2].x_m) / 3.0;
    plan->element_center_y[element] = (p[0].y_m + p[1].y_m + p[2].y_m) / 3.0;
  }
  if (!(options.airgap_radius_m >= 0.0) || !std::isfinite(options.airgap_radius_m)) {
    plan->status = Status::kInvalidArgument;
    return plan->status;
  }
  // Every requested probe lies on one circle.  Preserve FEMM's established
  // first-containing-element order, but avoid testing triangles whose axis-
  // aligned bounds cannot intersect that circle.  This is a conservative
  // geometry-only filter: every triangle that can contain a probe remains in
  // original element order, including boundary/tolerance cases.
  std::vector<int32_t> airgap_candidate_elements;
  airgap_candidate_elements.reserve(model.triangles.size());
  const double radius_squared = options.airgap_radius_m * options.airgap_radius_m;
  const double radial_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, radius_squared);
  for (size_t element = 0; element < model.triangles.size(); ++element) {
    const NonlinearTriangle &triangle = model.triangles[element];
    const Node &p0 = model.nodes[triangle.node[0]];
    const Node &p1 = model.nodes[triangle.node[1]];
    const Node &p2 = model.nodes[triangle.node[2]];
    const double min_x = std::min({p0.x_m, p1.x_m, p2.x_m});
    const double max_x = std::max({p0.x_m, p1.x_m, p2.x_m});
    const double min_y = std::min({p0.y_m, p1.y_m, p2.y_m});
    const double max_y = std::max({p0.y_m, p1.y_m, p2.y_m});
    const double closest_x = min_x > 0.0 ? min_x : (max_x < 0.0 ? max_x : 0.0);
    const double closest_y = min_y > 0.0 ? min_y : (max_y < 0.0 ? max_y : 0.0);
    const double minimum_squared = closest_x * closest_x + closest_y * closest_y;
    const double farthest_x = std::max(std::abs(min_x), std::abs(max_x));
    const double farthest_y = std::max(std::abs(min_y), std::abs(max_y));
    const double maximum_squared = farthest_x * farthest_x + farthest_y * farthest_y;
    if (radius_squared + radial_tolerance >= minimum_squared &&
        radius_squared <= maximum_squared + radial_tolerance) {
      airgap_candidate_elements.push_back(static_cast<int32_t>(element));
    }
  }
  plan->probes.reserve(options.airgap_angles_rad.size());
  for (const double angle : options.airgap_angles_rad) {
    if (!std::isfinite(angle)) {
      plan->status = Status::kInvalidArgument;
      return plan->status;
    }
    const double cosine = std::cos(angle), sine = std::sin(angle);
    const double x = options.airgap_radius_m * cosine;
    const double y = options.airgap_radius_m * sine;
    FrozenAirgapProbe probe;
    probe.cosine = cosine;
    probe.sine = sine;
    for (const int32_t element : airgap_candidate_elements) {
      const NonlinearTriangle &triangle = model.triangles[element];
      Node p[3] = {model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
                   model.nodes[triangle.node[2]]};
      double candidate[3] = {};
      if (TriangleBarycentric(p, x, y, candidate) && candidate[0] >= -1e-12 &&
          candidate[1] >= -1e-12 && candidate[2] >= -1e-12) {
        probe.containing_element = element;
        std::copy(candidate, candidate + 3, probe.lambda);
        break;
      }
    }
    if (probe.containing_element < 0 ||
        !IsAirPostprocessMaterial(
            model.materials[model.triangles[probe.containing_element].material], options,
            model.triangles[probe.containing_element].material)) {
      plan->status = Status::kInvalidArgument;
      return plan->status;
    }
    plan->probes.push_back(probe);
  }
  plan->status = Status::kOk;
  return plan->status;
}

FrozenPostprocessResult ComputeFrozenPostprocess(const NonlinearModel &model,
                                                 const NonlinearSolveResult &solution,
                                                 const FrozenPostprocessOptions &options,
                                                 const FrozenPostprocessPlan *reusable_plan)
{
  FrozenPostprocessResult result;
  result.force_x_n = 0.0;
  result.force_y_n = 0.0;
  result.torque_nm = 0.0;
  if (solution.a_wb_per_m.size() != model.nodes.size() ||
      solution.bx_t.size() != model.triangles.size() ||
      solution.by_t.size() != model.triangles.size() || !(model.depth_m > 0.0)) {
    result.status = Status::kInvalidArgument;
    return result;
  }
  FrozenPostprocessPlan local_plan;
  const FrozenPostprocessPlan *plan = reusable_plan;
  if (plan == nullptr) {
    result.status = BuildFrozenPostprocessPlan(model, options, &local_plan);
    plan = &local_plan;
  } else if (!FrozenPostprocessPlanMatches(*plan, model, options)) {
    result.status = Status::kInvalidArgument;
    return result;
  }
  result.status = plan->status;
  if (result.status != Status::kOk) {
    return result;
  }
  for (size_t element = 0; element < model.triangles.size(); ++element) {
    const double hx = plan->mask_gradient_x[element];
    const double hy = plan->mask_gradient_y[element];
    const double bx = solution.bx_t[element], by = solution.by_t[element];
    const double fx_density = ((bx * bx - by * by) * hx + 2.0 * bx * by * hy) / (2.0 * kMu0);
    const double fy_density = (2.0 * bx * by * hx + (by * by - bx * bx) * hy) / (2.0 * kMu0);
    const double weight = plan->element_weight[element];
    result.force_x_n += weight * fx_density;
    result.force_y_n += weight * fy_density;
    const double cx = plan->element_center_x[element];
    const double cy = plan->element_center_y[element];
    result.torque_nm += weight * (cx * fy_density - cy * fx_density);
  }
  for (size_t sample = 0; sample < plan->probes.size(); ++sample) {
    const double angle = options.airgap_angles_rad[sample];
    const FrozenAirgapProbe &probe = plan->probes[sample];
    const int32_t containing = probe.containing_element;
    double nodal_bx[3] = {}, nodal_by[3] = {};
    if (!SmoothedElementB(model, solution.a_wb_per_m, solution.bx_t, solution.by_t, plan->incident,
                          plan->edge_elements, containing, nodal_bx, nodal_by)) {
      result.status = Status::kMeshInvalid;
      return result;
    }
    double bx = 0.0, by = 0.0;
    for (int local = 0; local < 3; ++local) {
      bx += probe.lambda[local] * nodal_bx[local];
      by += probe.lambda[local] * nodal_by[local];
    }
    result.airgap_samples.push_back({angle, bx * probe.cosine + by * probe.sine});
  }
  if (!std::isfinite(result.force_x_n) || !std::isfinite(result.force_y_n) ||
      !std::isfinite(result.torque_nm))
    result.status = Status::kNumericalNonfinite;
  return result;
}

TangentPostprocessResult ComputeFrozenTangentPostprocess(const NonlinearModel &model,
                                                         const NonlinearSolveResult &base,
                                                         const NonlinearSolveResult &derivative,
                                                         const FrozenPostprocessOptions &options,
                                                         const std::vector<double> &mask,
                                                         const FrozenPostprocessPlan *reusable_plan)
{
  TangentPostprocessResult result;
  if (base.a_wb_per_m.size() != model.nodes.size() ||
      derivative.a_wb_per_m.size() != model.nodes.size() ||
      base.bx_t.size() != model.triangles.size() ||
      derivative.bx_t.size() != model.triangles.size() || mask.size() != model.nodes.size() ||
      (reusable_plan != nullptr && !FrozenPostprocessPlanMatches(*reusable_plan, model, options))) {
    result.status = Status::kInvalidArgument;
    return result;
  }
  std::map<EdgeKey, std::vector<int32_t>> local_edge_elements;
  std::vector<std::vector<int32_t>> local_incident(model.nodes.size());
  const std::map<EdgeKey, std::vector<int32_t>> *edge_elements =
      reusable_plan == nullptr ? &local_edge_elements : &reusable_plan->edge_elements;
  const std::vector<std::vector<int32_t>> *incident =
      reusable_plan == nullptr ? &local_incident : &reusable_plan->incident;
  for (size_t element = 0; element < model.triangles.size(); ++element) {
    const NonlinearTriangle &triangle = model.triangles[element];
    if (reusable_plan == nullptr)
      for (int local = 0; local < 3; ++local) {
        local_incident[triangle.node[local]].push_back(static_cast<int32_t>(element));
        local_edge_elements[CanonicalEdge(triangle.node[local], triangle.node[(local + 1) % 3])]
            .push_back(static_cast<int32_t>(element));
      }
    double hx = 0.0, hy = 0.0, weight = 0.0, cx = 0.0, cy = 0.0;
    if (reusable_plan != nullptr) {
      hx = reusable_plan->mask_gradient_x[element];
      hy = reusable_plan->mask_gradient_y[element];
      weight = reusable_plan->element_weight[element];
      cx = reusable_plan->element_center_x[element];
      cy = reusable_plan->element_center_y[element];
    } else {
      Node p[3] = {model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
                   model.nodes[triangle.node[2]]};
      NonlinearElementTerms terms;
      if (!BuildElementTerms(p, &terms)) {
        result.status = Status::kMeshInvalid;
        return result;
      }
      for (int local = 0; local < 3; ++local) {
        hx += mask[triangle.node[local]] * terms.b_y[local];
        hy -= mask[triangle.node[local]] * terms.b_x[local];
      }
      weight = terms.area_m2 * model.depth_m;
      cx = (p[0].x_m + p[1].x_m + p[2].x_m) / 3.0;
      cy = (p[0].y_m + p[1].y_m + p[2].y_m) / 3.0;
    }
    const double bx = base.bx_t[element], by = base.by_t[element];
    const double dbx = derivative.bx_t[element];
    const double dby = derivative.by_t[element];
    const double dnormal = 2.0 * (bx * dbx - by * dby);
    const double dshear = 2.0 * (dbx * by + bx * dby);
    const double dfx_density = (dnormal * hx + dshear * hy) / (2.0 * kMu0);
    const double dfy_density = (dshear * hx - dnormal * hy) / (2.0 * kMu0);
    result.force_x_n_per_a += weight * dfx_density;
    result.force_y_n_per_a += weight * dfy_density;
    result.torque_nm_per_a += weight * (cx * dfy_density - cy * dfx_density);
  }
  for (size_t sample = 0; sample < options.airgap_angles_rad.size(); ++sample) {
    const double angle = options.airgap_angles_rad[sample];
    const FrozenAirgapProbe *probe =
        reusable_plan == nullptr ? nullptr : &reusable_plan->probes[sample];
    const double cosine = probe == nullptr ? std::cos(angle) : probe->cosine;
    const double sine = probe == nullptr ? std::sin(angle) : probe->sine;
    const double x = options.airgap_radius_m * cosine;
    const double y = options.airgap_radius_m * sine;
    int32_t containing = -1;
    double lambda[3] = {};
    if (probe != nullptr) {
      containing = probe->containing_element;
      std::copy(probe->lambda, probe->lambda + 3, lambda);
    } else {
      for (size_t element = 0; element < model.triangles.size(); ++element) {
        const NonlinearTriangle &triangle = model.triangles[element];
        Node p[3] = {model.nodes[triangle.node[0]], model.nodes[triangle.node[1]],
                     model.nodes[triangle.node[2]]};
        double candidate[3] = {};
        if (TriangleBarycentric(p, x, y, candidate) && candidate[0] >= -1e-12 &&
            candidate[1] >= -1e-12 && candidate[2] >= -1e-12) {
          containing = static_cast<int32_t>(element);
          std::copy(candidate, candidate + 3, lambda);
          break;
        }
      }
    }
    if (containing < 0 ||
        !IsAirPostprocessMaterial(model.materials[model.triangles[containing].material], options,
                                  model.triangles[containing].material)) {
      result.status = Status::kInvalidArgument;
      return result;
    }
    double nodal_bx[3] = {}, nodal_by[3] = {};
    if (!SmoothedElementB(model, derivative.a_wb_per_m, derivative.bx_t, derivative.by_t, *incident,
                          *edge_elements, containing, nodal_bx, nodal_by)) {
      result.status = Status::kMeshInvalid;
      return result;
    }
    double bx = 0.0, by = 0.0;
    for (int local = 0; local < 3; ++local) {
      bx += lambda[local] * nodal_bx[local];
      by += lambda[local] * nodal_by[local];
    }
    result.airgap_radial_flux_density_t_per_a.push_back(bx * cosine + by * sine);
  }
  result.status = std::isfinite(result.force_x_n_per_a) && std::isfinite(result.force_y_n_per_a) &&
                          std::isfinite(result.torque_nm_per_a)
                      ? Status::kOk
                      : Status::kNumericalNonfinite;
  return result;
}

} // namespace gpu_femm

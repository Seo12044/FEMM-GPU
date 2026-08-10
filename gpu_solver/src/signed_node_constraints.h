#pragma once

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace gpu_femm {

struct SignedNodeConstraint {
  int32_t node_a = -1;
  int32_t node_b = -1;
  // The constrained values satisfy A(node_b) = sign * A(node_a).
  int8_t sign = 1;
};

struct SignedDofMap {
  std::vector<int32_t> node_root;
  std::vector<int8_t> node_sign;
  std::vector<bool> root_fixed;
  std::vector<double> root_value;
  std::vector<int32_t> root_to_free;
  std::vector<int32_t> free_roots;
};

enum class SignedDofStatus {
  kOk,
  kInvalidConstraint,
  kContradictoryCycle,
  kInvalidFixedValue,
};

inline SignedDofStatus BuildSignedDofMap(size_t node_count,
    const std::vector<SignedNodeConstraint>& constraints,
    const std::vector<int32_t>& fixed_nodes,
    const std::vector<double>& fixed_values, SignedDofMap* result)
{
  if (result == nullptr || node_count == 0 || fixed_nodes.size() != fixed_values.size())
    return SignedDofStatus::kInvalidFixedValue;
  std::vector<std::vector<std::pair<int32_t, int8_t>>> adjacency(node_count);
  for (const SignedNodeConstraint& constraint : constraints) {
    if (constraint.node_a < 0 || constraint.node_b < 0
        || static_cast<size_t>(constraint.node_a) >= node_count
        || static_cast<size_t>(constraint.node_b) >= node_count
        || (constraint.sign != 1 && constraint.sign != -1))
      return SignedDofStatus::kInvalidConstraint;
    if (constraint.node_a == constraint.node_b) {
      return constraint.sign < 0 ? SignedDofStatus::kContradictoryCycle
                                 : SignedDofStatus::kInvalidConstraint;
    }
    adjacency[constraint.node_a].push_back({ constraint.node_b, constraint.sign });
    adjacency[constraint.node_b].push_back({ constraint.node_a, constraint.sign });
  }

  SignedDofMap mapped;
  mapped.node_root.assign(node_count, -1);
  mapped.node_sign.assign(node_count, 0);
  for (size_t start = 0; start < node_count; ++start) {
    if (mapped.node_root[start] >= 0)
      continue;
    mapped.node_root[start] = static_cast<int32_t>(start);
    mapped.node_sign[start] = 1;
    std::vector<int32_t> pending { static_cast<int32_t>(start) };
    while (!pending.empty()) {
      const int32_t node = pending.back();
      pending.pop_back();
      for (const auto& edge : adjacency[node]) {
        const int32_t neighbor = edge.first;
        const int8_t neighbor_sign = static_cast<int8_t>(mapped.node_sign[node] * edge.second);
        if (mapped.node_root[neighbor] < 0) {
          mapped.node_root[neighbor] = static_cast<int32_t>(start);
          mapped.node_sign[neighbor] = neighbor_sign;
          pending.push_back(neighbor);
        } else if (mapped.node_root[neighbor] != static_cast<int32_t>(start)
            || mapped.node_sign[neighbor] != neighbor_sign) {
          return SignedDofStatus::kContradictoryCycle;
        }
      }
    }
  }

  mapped.root_fixed.assign(node_count, false);
  mapped.root_value.assign(node_count, 0.0);
  for (size_t index = 0; index < fixed_nodes.size(); ++index) {
    const int32_t node = fixed_nodes[index];
    const double value = fixed_values[index];
    if (node < 0 || static_cast<size_t>(node) >= node_count || !std::isfinite(value))
      return SignedDofStatus::kInvalidFixedValue;
    const int32_t root = mapped.node_root[node];
    const double root_value = static_cast<double>(mapped.node_sign[node]) * value;
    if (mapped.root_fixed[root] && mapped.root_value[root] != root_value)
      return SignedDofStatus::kInvalidFixedValue;
    mapped.root_fixed[root] = true;
    mapped.root_value[root] = root_value;
  }

  mapped.root_to_free.assign(node_count, -1);
  for (size_t node = 0; node < node_count; ++node) {
    if (mapped.node_root[node] == static_cast<int32_t>(node) && !mapped.root_fixed[node]) {
      mapped.root_to_free[node] = static_cast<int32_t>(mapped.free_roots.size());
      mapped.free_roots.push_back(static_cast<int32_t>(node));
    }
  }
  *result = std::move(mapped);
  return SignedDofStatus::kOk;
}

} // namespace gpu_femm

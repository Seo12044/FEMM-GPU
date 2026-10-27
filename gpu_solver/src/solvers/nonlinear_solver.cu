#include "solvers/nonlinear_solver.cuh"
#include "core/profile.h"
#include "axisymmetric_p1.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <system_error>
#include <thread>
#include <utility>

namespace gpu_femm
{

NonlinearSolveResult NonlinearP1Solver::FinalizeResult(NonlinearSolveResult *result) const
{
  result->bx_t.assign(model_.triangles.size(), 0.0);
  result->by_t.assign(model_.triangles.size(), 0.0);
  result->circuit_flux_linkage_wb.assign(static_cast<size_t>(model_.circuit_count), 0.0);
  for (size_t element = 0; element < model_.triangles.size(); ++element) {
    const NonlinearTriangle &triangle = model_.triangles[element];
    Node p[3] = {model_.nodes[triangle.node[0]], model_.nodes[triangle.node[1]],
                 model_.nodes[triangle.node[2]]};
    NonlinearElementTerms terms;
    if (!BuildElementTerms(p, &terms)) {
      result->info.status = Status::kMeshInvalid;
      return *result;
    }
    const NonlinearMaterial &material = model_.materials[triangle.material];
    if (model_.problem_type == MagneticProblemType::kPlanar) {
      for (int i = 0; i < 3; ++i) {
        result->bx_t[element] += result->a_wb_per_m[triangle.node[i]] * terms.b_x[i];
        result->by_t[element] += result->a_wb_per_m[triangle.node[i]] * terms.b_y[i];
      }
      if (material.circuit_index >= 0) {
        const size_t circuit = static_cast<size_t>(material.circuit_index);
        result->circuit_flux_linkage_wb[circuit] +=
            terms.area_m2 * material.source_j_per_a / 3.0 *
            (result->a_wb_per_m[triangle.node[0]] + result->a_wb_per_m[triangle.node[1]] +
             result->a_wb_per_m[triangle.node[2]]);
      }
    } else {
      const double radius[3] = {p[0].x_m, p[1].x_m, p[2].x_m};
      const double axial[3] = {p[0].y_m, p[1].y_m, p[2].y_m};
      const double nodal_a[3] = {
          result->a_wb_per_m[triangle.node[0]],
          result->a_wb_per_m[triangle.node[1]],
          result->a_wb_per_m[triangle.node[2]],
      };
      if (!EvaluateAxisymmetricP1Centroid(radius, axial, nodal_a, &result->bx_t[element],
                                          &result->by_t[element])) {
        result->info.status = Status::kMeshInvalid;
        return *result;
      }
      if (material.circuit_index >= 0) {
        AxisymmetricP1Element axisymmetric;
        if (!BuildAxisymmetricP1Element(radius, axial, &axisymmetric)) {
          result->info.status = Status::kMeshInvalid;
          return *result;
        }
        constexpr double kTwoPi = 6.283185307179586476925286766559005768;
        double flux = 0.0;
        for (const AxisymmetricP1Point &point : axisymmetric.points) {
          double a = 0.0;
          for (int local = 0; local < 3; ++local)
            a += point.shape[local] * nodal_a[local];
          flux += kTwoPi * point.radius_m * point.weight_area_m2 * material.source_j_per_a * a;
        }
        result->circuit_flux_linkage_wb[material.circuit_index] += flux;
      }
    }
  }
  for (double &flux : result->circuit_flux_linkage_wb) {
    if (model_.problem_type == MagneticProblemType::kPlanar)
      flux *= model_.depth_m;
    if (!std::isfinite(flux))
      result->info.status = Status::kNumericalNonfinite;
  }
  if (model_.circuit_count == 1)
    result->flux_linkage_wb = result->circuit_flux_linkage_wb[0];
  return *result;
}

std::vector<TangentSolveResult> NonlinearP1Solver::SolveTangentMultiRhs(
    const NonlinearSolveResult &base,
    const std::vector<std::pair<std::string, std::vector<double>>> &directions,
    const NonlinearOptions &options, int *batched_pcg_launches)
{
  std::vector<TangentSolveResult> results(directions.size());
  for (size_t index = 0; index < directions.size(); ++index) {
    results[index].name = directions[index].first;
    results[index].circuit_current_derivative_a_per_a = directions[index].second;
    results[index].info.status = Status::kInvalidArgument;
  }
  if (!initialized_ || base.info.status != Status::kOk ||
      base.a_wb_per_m.size() != model_.nodes.size() || directions.empty() ||
      !(options.linear_relative_tolerance > 0.0) || options.max_linear_iterations < 0)
    return results;
  for (const auto &direction : directions)
    if (direction.second.size() != static_cast<size_t>(model_.circuit_count) ||
        !std::all_of(direction.second.begin(), direction.second.end(),
                     [](double value) { return std::isfinite(value); }))
      return results;

  Assembly tangent;
  Status status =
      AssembleNonlinearNewton(model_, base.a_wb_per_m, base.circuit_currents_a, true, &tangent);
  if (status != Status::kOk) {
    for (TangentSolveResult &result : results)
      result.info.status = status;
    return results;
  }
  if (!csr_initialized_) {
    status = csr_solver_.Initialize(tangent);
    csr_initialized_ = status == Status::kOk;
  } else if (!csr_solver_.HasMatchingStructure(tangent)) {
    status = Status::kAssemblyFailed;
  } else {
    status = csr_solver_.UpdateValues(tangent);
    if (status == Status::kOk)
      ++csr_symbolic_reuse_count_;
  }
  if (status != Status::kOk) {
    for (TangentSolveResult &result : results)
      result.info.status = status;
    return results;
  }

  std::vector<std::vector<double>> rhs_values(directions.size(),
                                              std::vector<double>(tangent.free_nodes.size(), 0.0));
  for (size_t item = 0; item < directions.size(); ++item) {
    for (size_t circuit = 0; circuit < directions[item].second.size(); ++circuit) {
      const double scale = directions[item].second[circuit];
      if (scale == 0.0)
        continue;
      const std::vector<double> &source = tangent.source_load_per_circuit[circuit];
      for (size_t node = 0; node < source.size(); ++node) {
        const int32_t row = tangent.node_to_free[node];
        if (row >= 0)
          rhs_values[item][row] +=
              scale * static_cast<double>(tangent.node_sign[node]) * source[node];
      }
    }
  }

  std::vector<SolveInfo> infos;
  std::vector<std::vector<double>> free_solutions;
  SharedOperatorBatchEvidence operator_evidence;
  status = csr_solver_.SolveCurrentOperatorBatch(
      rhs_values, options.linear_relative_tolerance, options.max_linear_iterations, &infos,
      &free_solutions, batched_pcg_launches, &operator_evidence);
  if (status != Status::kOk) {
    for (TangentSolveResult &result : results)
      result.info.status = status;
    return results;
  }
  for (size_t item = 0; item < results.size(); ++item) {
    results[item].info = infos[item];
    results[item].operator_evidence = operator_evidence;
    if (infos[item].status != Status::kOk)
      continue;
    NonlinearSolveResult derivative;
    derivative.info = infos[item];
    derivative.a_wb_per_m = ExpandAssemblyDerivative(tangent, free_solutions[item]);
    derivative.circuit_currents_a = directions[item].second;
    derivative = FinalizeResult(&derivative);
    results[item].info = derivative.info;
    results[item].field_derivative = std::move(derivative);
  }
  return results;
}

NonlinearSolveResult NonlinearP1Solver::SolveAccelerated(const std::vector<double> &currents,
                                                         const NonlinearOptions &options,
                                                         NonlinearAssemblyMode mode)
{
  std::vector<NonlinearSolveResult> results =
      SolveBatch({currents}, options, nullptr, nullptr, mode);
  return results.empty() ? NonlinearSolveResult{} : std::move(results.front());
}

std::vector<NonlinearSolveResult>
NonlinearP1Solver::SolveBatch(const std::vector<std::vector<double>> &currents,
                              NonlinearOptions options, int *batched_pcg_launches,
                              NonlinearBatchTiming *timing, NonlinearAssemblyMode requested_mode)
{
  options.adaptive_linear_tolerance &= has_nonlinear_materials_;
  std::vector<NonlinearSolveResult> results(currents.size());
  if (!initialized_ || currents.empty() || !(options.relative_tolerance > 0.0) ||
      !std::isfinite(options.relative_tolerance) || options.max_newton_iterations < 0)
    return results;
  std::vector<std::vector<double>> states(currents.size(),
                                          std::vector<double>(model_.nodes.size(), 0.0));
  std::vector<double> relaxations(currents.size(), 1.0);
  std::vector<double> previous_residuals(currents.size(), std::numeric_limits<double>::infinity());
  std::vector<bool> active(currents.size(), true);
  for (size_t item = 0; item < currents.size(); ++item) {
    results[item].info.status = Status::kInvalidArgument;
    results[item].circuit_currents_a = currents[item];
    if (currents[item].size() != static_cast<size_t>(model_.circuit_count) ||
        !std::all_of(currents[item].begin(), currents[item].end(),
                     [](double value) { return std::isfinite(value); })) {
      active[item] = false;
      continue;
    }
    for (size_t boundary = 0; boundary < model_.dirichlet_nodes.size(); ++boundary)
      states[item][model_.dirichlet_nodes[boundary]] = model_.dirichlet_a_wb_per_m[boundary];
  }
  // The plan owns the signed reduction and both P1 operators.  For tiny
  // one-off models its allocation/setup cost exceeds host assembly, so auto
  // keeps the reference path until there is enough quadrature work. Forced
  // device mode remains available for parity and profiling.
  const size_t quadrature_per_triangle =
      model_.problem_type == MagneticProblemType::kAxisymmetric ? 7 : 1;
  constexpr size_t kMinimumDeviceAssemblyQuadratureWork = 4096;
  const bool device_assembly_worthwhile =
      model_.triangles.size() * quadrature_per_triangle * currents.size() >=
      kMinimumDeviceAssemblyQuadratureWork;
  bool device_assembly_active =
      requested_mode == NonlinearAssemblyMode::kDevice ||
      (requested_mode == NonlinearAssemblyMode::kAuto && device_assembly_worthwhile);
  if (device_assembly_active) {
    device_assembly_ = std::make_unique<DeviceNonlinearAssemblyPlan>();
    const Status plan_status = device_assembly_->Initialize(model_);
    const Status setup_status = plan_status == Status::kOk
                                    ? csr_solver_.Initialize(device_assembly_->symbolic_assembly())
                                    : plan_status;
    if (setup_status == Status::kOk) {
      csr_initialized_ = true;
      // A fixed signed root may prescribe values on nodes other than the
      // literal Dirichlet record.  The device Newton state must start from
      // the same expanded boundary vector as the host reduced assembler.
      const std::vector<double> &boundary = device_assembly_->symbolic_assembly().boundary_values;
      for (size_t item = 0; item < states.size(); ++item)
        if (active[item])
          for (size_t node = 0; node < boundary.size(); ++node)
            if (boundary[node] != 0.0)
              states[item][node] = boundary[node];
    } else if (requested_mode == NonlinearAssemblyMode::kAuto) {
      device_assembly_.reset();
      csr_initialized_ = false;
      device_assembly_active = false;
    } else {
      for (size_t item = 0; item < results.size(); ++item)
        if (active[item]) {
          results[item].info.status = setup_status;
          active[item] = false;
        }
    }
  }
  for (int iteration = 0; iteration < options.max_newton_iterations; ++iteration) {
    const ProfileClock::time_point assembly_start = ProfileClock::now();
    std::vector<size_t> slots;
    std::vector<Assembly> assemblies;
    std::vector<std::vector<double>> rhs_values;
    const bool symbolic_preexisting = csr_initialized_;
    std::vector<size_t> active_slots;
    for (size_t item = 0; item < currents.size(); ++item)
      if (active[item])
        active_slots.push_back(item);
    if (active_slots.empty())
      break;
    if (device_assembly_active) {
      std::vector<std::vector<double>> device_states, device_currents;
      device_states.reserve(active_slots.size());
      device_currents.reserve(active_slots.size());
      for (size_t item : active_slots) {
        device_states.push_back(states[item]);
        device_currents.push_back(currents[item]);
      }
      const Status assembled = device_assembly_->Assemble(
          device_states, device_currents, iteration > 0, &csr_solver_,
          timing == nullptr ? nullptr : &timing->device_assembly_seconds);
      if (assembled != Status::kOk) {
        if (requested_mode == NonlinearAssemblyMode::kAuto) {
          device_assembly_active = false;
          device_assembly_.reset();
          csr_initialized_ = false;
          --iteration;
          continue; // retry this exact Newton iteration on the reference path.
        }
        for (size_t item : active_slots) {
          results[item].info.status = assembled;
          active[item] = false;
        }
        continue;
      }
      std::vector<SolveInfo> infos;
      std::vector<std::vector<double>> free_solutions;
      GpuBatchSolveTiming gpu_timing;
      const double linear_tolerance =
          BatchNewtonLinearTolerance(options, previous_residuals, active_slots);
      const Status solve_status = csr_solver_.SolvePreparedDeviceBatch(
          active_slots.size(), linear_tolerance, options.max_linear_iterations, &infos,
          &free_solutions, batched_pcg_launches, &gpu_timing);
      if (timing != nullptr) {
        timing->gpu_kernel_sync_seconds += gpu_timing.kernel_sync_seconds;
        timing->gpu_download_seconds += gpu_timing.download_seconds;
        AccumulatePreconditionerStats(timing->preconditioner, gpu_timing.preconditioner);
        timing->interleaved_spmv_solves += gpu_timing.interleaved_spmv_solves;
      }
      if (solve_status != Status::kOk) {
        for (size_t item : active_slots) {
          results[item].info.status = solve_status;
          active[item] = false;
        }
        continue;
      }
      csr_symbolic_reuse_count_ += iteration == 0 ? active_slots.size() - 1 : active_slots.size();
      const Assembly &assembly = device_assembly_->symbolic_assembly();
      const ProfileClock::time_point update_start = ProfileClock::now();
      double finalized = 0.0;
      for (size_t local = 0; local < active_slots.size(); ++local) {
        const size_t item = active_slots[local];
        results[item].linear_iterations += infos[local].iterations;
        results[item].last_linear_tolerance = linear_tolerance;
        if (infos[local].status != Status::kOk) {
          results[item].info = infos[local];
          active[item] = false;
          continue;
        }
        std::vector<double> candidate = ExpandAssemblySolution(assembly, free_solutions[local]);
        double difference_sq = 0.0, candidate_sq = 0.0;
        for (size_t node = 0; node < candidate.size(); ++node) {
          const double difference = candidate[node] - states[item][node];
          difference_sq += difference * difference;
          candidate_sq += candidate[node] * candidate[node];
        }
        const double residual =
            std::sqrt(difference_sq) /
            std::max(std::sqrt(candidate_sq), std::numeric_limits<double>::min());
        results[item].residual_history.push_back(residual);
        results[item].info.iterations = iteration + 1;
        results[item].info.residual_l2 = residual;
        if (!std::isfinite(residual)) {
          results[item].info.status = Status::kNumericalNonfinite;
          active[item] = false;
          continue;
        }
        if (iteration > 5) {
          if (residual > previous_residuals[item] && relaxations[item] > 0.125)
            relaxations[item] *= 0.5;
          else
            relaxations[item] += 0.1 * (1.0 - relaxations[item]);
        }
        for (size_t node = 0; node < states[item].size(); ++node)
          states[item][node] += relaxations[item] * (candidate[node] - states[item][node]);
        if (NewtonStepConverged(options, iteration, residual, linear_tolerance)) {
          results[item].info.status = Status::kOk;
          results[item].a_wb_per_m = std::move(states[item]);
          const ProfileClock::time_point finalize_start = ProfileClock::now();
          results[item] = FinalizeResult(&results[item]);
          active[item] = false;
          finalized += ProfileSecondsSince(finalize_start);
        } else
          previous_residuals[item] = residual;
      }
      if (timing != nullptr) {
        timing->finalize_seconds += finalized;
        timing->state_update_seconds +=
            std::max(0.0, ProfileSecondsSince(update_start) - finalized);
      }
      continue;
    }
    std::vector<Assembly> iteration_assemblies(active_slots.size());
    std::vector<Status> assembly_status(active_slots.size(), Status::kInternalError);
    std::atomic<size_t> next_assembly{0};
    const size_t worker_count =
        std::min<size_t>(active_slots.size(),
                         std::min<size_t>(6, std::max(1u, std::thread::hardware_concurrency())));
    auto assemble = [&]() {
      for (;;) {
        const size_t local = next_assembly.fetch_add(1, std::memory_order_relaxed);
        if (local >= active_slots.size())
          return;
        const size_t item = active_slots[local];
        assembly_status[local] = AssembleNonlinearNewton(
            model_, states[item], currents[item], iteration > 0, &iteration_assemblies[local]);
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(worker_count > 0 ? worker_count - 1 : 0);
    for (size_t worker = 1; worker < worker_count; ++worker) {
      try {
        workers.emplace_back(assemble);
      } catch (const std::system_error &) {
        break;
      }
    }
    assemble();
    for (std::thread &worker : workers)
      worker.join();
    for (size_t local = 0; local < active_slots.size(); ++local) {
      const size_t item = active_slots[local];
      const Status status = assembly_status[local];
      if (status != Status::kOk) {
        results[item].info.status = status;
        active[item] = false;
        continue;
      }
      Assembly &assembly = iteration_assemblies[local];
      if (!csr_initialized_) {
        const Status initialize = csr_solver_.Initialize(assembly);
        if (initialize != Status::kOk) {
          results[item].info.status = initialize;
          active[item] = false;
          continue;
        }
        csr_initialized_ = true;
      }
      if (!csr_solver_.HasMatchingStructure(assembly)) {
        results[item].info.status = Status::kAssemblyFailed;
        active[item] = false;
        continue;
      }
      std::vector<double> rhs(assembly.free_nodes.size());
      for (size_t row = 0; row < rhs.size(); ++row)
        rhs[row] = assembly.rhs_per_amp[row] + assembly.rhs_offset[row];
      slots.push_back(item);
      assemblies.push_back(std::move(assembly));
      rhs_values.push_back(std::move(rhs));
    }
    if (timing != nullptr)
      timing->host_assembly_seconds += ProfileSecondsSince(assembly_start);
    if (slots.empty())
      break;
    std::vector<SolveInfo> infos;
    std::vector<std::vector<double>> free_solutions;
    GpuBatchSolveTiming gpu_timing;
    const double linear_tolerance = BatchNewtonLinearTolerance(options, previous_residuals, slots);
    const Status batch_status = csr_solver_.SolveBatch(
        assemblies, rhs_values, linear_tolerance, options.max_linear_iterations, &infos,
        &free_solutions, batched_pcg_launches, &gpu_timing);
    if (timing != nullptr) {
      timing->gpu_upload_seconds += gpu_timing.upload_seconds;
      timing->gpu_kernel_sync_seconds += gpu_timing.kernel_sync_seconds;
      timing->gpu_download_seconds += gpu_timing.download_seconds;
      AccumulatePreconditionerStats(timing->preconditioner, gpu_timing.preconditioner);
      timing->interleaved_spmv_solves += gpu_timing.interleaved_spmv_solves;
    }
    if (batch_status != Status::kOk) {
      for (size_t item : slots) {
        results[item].info.status = batch_status;
        active[item] = false;
      }
      break;
    }
    csr_symbolic_reuse_count_ += symbolic_preexisting ? slots.size() : slots.size() - 1;
    const ProfileClock::time_point update_start = ProfileClock::now();
    double iteration_finalize_seconds = 0.0;
    for (size_t local = 0; local < slots.size(); ++local) {
      const size_t item = slots[local];
      results[item].linear_iterations += infos[local].iterations;
      results[item].last_linear_tolerance = linear_tolerance;
      if (infos[local].status != Status::kOk) {
        results[item].info = infos[local];
        active[item] = false;
        continue;
      }
      std::vector<double> candidate =
          ExpandAssemblySolution(assemblies[local], free_solutions[local]);
      double difference_sq = 0.0, candidate_sq = 0.0;
      for (size_t node = 0; node < candidate.size(); ++node) {
        const double difference = candidate[node] - states[item][node];
        difference_sq += difference * difference;
        candidate_sq += candidate[node] * candidate[node];
      }
      const double residual = std::sqrt(difference_sq) /
                              std::max(std::sqrt(candidate_sq), std::numeric_limits<double>::min());
      results[item].residual_history.push_back(residual);
      results[item].info.iterations = iteration + 1;
      results[item].info.residual_l2 = residual;
      if (!std::isfinite(residual)) {
        results[item].info.status = Status::kNumericalNonfinite;
        active[item] = false;
        continue;
      }
      if (iteration > 5) {
        if (residual > previous_residuals[item] && relaxations[item] > 0.125)
          relaxations[item] *= 0.5;
        else
          relaxations[item] += 0.1 * (1.0 - relaxations[item]);
      }
      for (size_t node = 0; node < states[item].size(); ++node)
        states[item][node] += relaxations[item] * (candidate[node] - states[item][node]);
      if (NewtonStepConverged(options, iteration, residual, linear_tolerance)) {
        results[item].info.status = Status::kOk;
        results[item].a_wb_per_m = std::move(states[item]);
        const ProfileClock::time_point finalize_start = ProfileClock::now();
        results[item] = FinalizeResult(&results[item]);
        active[item] = false;
        iteration_finalize_seconds += ProfileSecondsSince(finalize_start);
      } else
        previous_residuals[item] = residual;
    }
    if (timing != nullptr) {
      timing->finalize_seconds += iteration_finalize_seconds;
      timing->state_update_seconds +=
          std::max(0.0, ProfileSecondsSince(update_start) - iteration_finalize_seconds);
    }
  }
  for (size_t item = 0; item < results.size(); ++item)
    if (active[item]) {
      results[item].info.status = Status::kNonlinearSolveNotConverged;
      results[item].a_wb_per_m = std::move(states[item]);
      const ProfileClock::time_point finalize_start = ProfileClock::now();
      results[item] = FinalizeResult(&results[item]);
      if (timing != nullptr)
        timing->finalize_seconds += ProfileSecondsSince(finalize_start);
    }
  if (options.recover_nonconvergence && options.max_newton_iterations > 0) {
    for (size_t item = 0; item < results.size(); ++item) {
      if (results[item].info.status != Status::kNonlinearSolveNotConverged)
        continue;
      const int original_iterations = results[item].info.iterations;
      const ProfileClock::time_point recovery_start = ProfileClock::now();
      NonlinearSolveResult recovered = SolveSafeguarded(currents[item], options);
      if (timing != nullptr)
        timing->recovery_seconds += ProfileSecondsSince(recovery_start);
      recovered.info.iterations += original_iterations;
      recovered.linear_iterations += results[item].linear_iterations;
      results[item] = std::move(recovered);
    }
  }
  return results;
}

// Only exhausted nonlinear states enter recovery. Armijo backtracking measures
// the equation residual so a small damped step cannot certify convergence.
NonlinearSolveResult NonlinearP1Solver::SolveSafeguarded(const std::vector<double> &currents,
                                                         const NonlinearOptions &options)
{
  NonlinearSolveResult result;
  result.info.status = Status::kInvalidArgument;
  result.circuit_currents_a = currents;
  if (!initialized_ || currents.size() != static_cast<size_t>(model_.circuit_count) ||
      !std::all_of(currents.begin(), currents.end(), [](double x) { return std::isfinite(x); }) ||
      !(options.relative_tolerance > 0) || !std::isfinite(options.relative_tolerance) ||
      options.max_newton_iterations < 0)
    return result;
  std::vector<double> state(model_.nodes.size(), 0.0);
  for (size_t i = 0; i < model_.dirichlet_nodes.size(); ++i)
    state[model_.dirichlet_nodes[i]] = model_.dirichlet_a_wb_per_m[i];

  auto equation_norm = [](const Assembly &assembly, const std::vector<double> &a) {
    double norm = 0.0;
    for (size_t row = 0; row < assembly.free_nodes.size(); ++row) {
      double value = -assembly.rhs_per_amp[row] - assembly.rhs_offset[row];
      for (int slot = assembly.row_offsets[row]; slot < assembly.row_offsets[row + 1]; ++slot)
        value += assembly.values[slot] * a[assembly.free_nodes[assembly.column_indices[slot]]];
      norm = std::hypot(norm, value);
    }
    return norm;
  };
  Assembly assembly;
  Status status = AssembleNonlinearNewton(model_, state, currents, true, &assembly);
  if (status != Status::kOk) {
    result.info.status = status;
    return result;
  }
  // Expand prescribed signed roots before the first Newton direction.
  std::vector<double> free_state(assembly.free_nodes.size(), 0.0);
  for (size_t i = 0; i < free_state.size(); ++i)
    free_state[i] = state[assembly.free_nodes[i]];
  state = ExpandAssemblySolution(assembly, free_state);
  status = AssembleNonlinearNewton(model_, state, currents, true, &assembly);
  if (status != Status::kOk) {
    result.info.status = status;
    return result;
  }
  double norm = equation_norm(assembly, state);
  const double scale = std::max(norm, std::numeric_limits<double>::min());
  const double equation_tolerance = options.relative_tolerance;
  const double step_tolerance = 100.0 * options.relative_tolerance;
  GpuCsrSolver recovery_solver;
  status = recovery_solver.Initialize(assembly);
  if (status != Status::kOk) {
    result.info.status = status;
    return result;
  }
  result.info.status = Status::kNonlinearSolveNotConverged;
  for (int iteration = 0; iteration < options.max_newton_iterations; ++iteration) {
    if (!std::isfinite(norm)) {
      result.info.status = Status::kNumericalNonfinite;
      break;
    }
    std::vector<double> rhs(assembly.free_nodes.size());
    for (size_t row = 0; row < rhs.size(); ++row)
      rhs[row] = assembly.rhs_per_amp[row] + assembly.rhs_offset[row];
    status = recovery_solver.UpdateValues(assembly);
    if (status != Status::kOk) {
      result.info.status = status;
      break;
    }
    std::vector<double> free_solution;
    const SolveInfo linear = recovery_solver.Solve(rhs, options.linear_relative_tolerance,
                                                   options.max_linear_iterations, &free_solution);
    result.linear_iterations += linear.iterations;
    result.last_linear_tolerance = options.linear_relative_tolerance;
    if (linear.status != Status::kOk) {
      result.info = linear;
      break;
    }
    const std::vector<double> candidate = ExpandAssemblySolution(assembly, free_solution);
    double step_norm = 0.0, candidate_norm = 0.0;
    for (size_t node = 0; node < state.size(); ++node) {
      step_norm = std::hypot(step_norm, candidate[node] - state[node]);
      candidate_norm = std::hypot(candidate_norm, candidate[node]);
    }
    const double relative_step =
        step_norm / std::max(candidate_norm, std::numeric_limits<double>::min());
    result.info.iterations = iteration + 1;
    result.info.residual_l2 = relative_step;
    result.residual_history.push_back(relative_step);
    if (!std::isfinite(relative_step)) {
      result.info.status = Status::kNumericalNonfinite;
      break;
    }
    // Check both the undamped Newton correction and the equations at the
    // returned state. The original 1e-6 step gate is not relaxed.
    if (relative_step < step_tolerance && norm / scale <= equation_tolerance) {
      result.info.status = Status::kOk;
      break;
    }
    bool accepted = false;
    double alpha = 1.0;
    constexpr int kMaximumBacktracks = 20;
    constexpr double kArmijoDecrease = 1e-4;
    for (int backtrack = 0; backtrack < kMaximumBacktracks; ++backtrack, alpha *= 0.5) {
      std::vector<double> trial(state.size());
      for (size_t node = 0; node < state.size(); ++node)
        trial[node] = state[node] + alpha * (candidate[node] - state[node]);
      Assembly trial_assembly;
      status = AssembleNonlinearNewton(model_, trial, currents, true, &trial_assembly);
      if (status != Status::kOk) {
        result.info.status = status;
        break;
      }
      const double trial_norm = equation_norm(trial_assembly, trial);
      if (std::isfinite(trial_norm) && (trial_norm <= (1.0 - kArmijoDecrease * alpha) * norm ||
                                        trial_norm / scale <= equation_tolerance)) {
        state = std::move(trial);
        assembly = std::move(trial_assembly);
        norm = trial_norm;
        accepted = true;
        break;
      }
    }
    if (!accepted)
      break;
  }
  result.a_wb_per_m = std::move(state);
  result.verified_equation_residual_relative = norm / scale;
  return FinalizeResult(&result);
}

NonlinearSolveResult NonlinearP1Solver::Solve(double current_a, const NonlinearOptions &options,
                                              const std::vector<double> *warm_start)
{
  return Solve(std::vector<double>{current_a}, options, warm_start);
}

NonlinearSolveResult NonlinearP1Solver::Solve(const std::vector<double> &circuit_currents_a,
                                              NonlinearOptions options,
                                              const std::vector<double> *warm_start)
{
  options.adaptive_linear_tolerance &= has_nonlinear_materials_;
  NonlinearSolveResult result;
  result.info.status = Status::kInvalidArgument;
  if (!initialized_ || circuit_currents_a.size() != static_cast<size_t>(model_.circuit_count) ||
      !(options.relative_tolerance > 0.0) || !std::isfinite(options.relative_tolerance) ||
      options.max_newton_iterations < 0)
    return result;
  for (const double current_a : circuit_currents_a) {
    if (!std::isfinite(current_a))
      return result;
  }
  result.circuit_currents_a = circuit_currents_a;
  std::vector<double> a(model_.nodes.size(), 0.0);
  for (size_t i = 0; i < model_.dirichlet_nodes.size(); ++i)
    a[model_.dirichlet_nodes[i]] = model_.dirichlet_a_wb_per_m[i];
  const bool has_warm_start = warm_start != nullptr;
  if (has_warm_start) {
    if (warm_start->size() != a.size())
      return result;
    a = *warm_start;
    for (size_t i = 0; i < model_.dirichlet_nodes.size(); ++i)
      a[model_.dirichlet_nodes[i]] = model_.dirichlet_a_wb_per_m[i];
  }
  double relaxation = 1.0;
  double previous_residual = std::numeric_limits<double>::infinity();
  for (int iteration = 0; iteration < options.max_newton_iterations; ++iteration) {
    Assembly assembly;
    const bool use_newton = has_warm_start || iteration > 0;
    Status status = AssembleNonlinearNewton(model_, a, circuit_currents_a, use_newton, &assembly);
    if (status != Status::kOk) {
      result.info.status = status;
      return result;
    }
    if (!csr_initialized_) {
      status = csr_solver_.Initialize(assembly);
      if (status == Status::kOk)
        csr_initialized_ = true;
    } else if (!csr_solver_.HasMatchingStructure(assembly)) {
      // Geometry cache identity promises this cannot change.  Fail rather
      // than silently re-uploading a different symbolic matrix.
      status = Status::kAssemblyFailed;
    } else {
      status = csr_solver_.UpdateValues(assembly);
      if (status == Status::kOk)
        ++csr_symbolic_reuse_count_;
    }
    if (status != Status::kOk) {
      result.info.status = status;
      return result;
    }
    std::vector<double> free_solution;
    std::vector<double> rhs(assembly.free_nodes.size());
    for (size_t row = 0; row < rhs.size(); ++row)
      rhs[row] = assembly.rhs_per_amp[row] + assembly.rhs_offset[row];
    const double linear_tolerance = NewtonLinearTolerance(options, previous_residual);
    const SolveInfo linear_info =
        csr_solver_.Solve(rhs, linear_tolerance, options.max_linear_iterations, &free_solution);
    result.linear_iterations += linear_info.iterations;
    result.last_linear_tolerance = linear_tolerance;
    if (linear_info.status != Status::kOk) {
      result.info = linear_info;
      return result;
    }
    std::vector<double> candidate = ExpandAssemblySolution(assembly, free_solution);
    double difference_sq = 0.0;
    double candidate_sq = 0.0;
    for (size_t i = 0; i < candidate.size(); ++i) {
      const double difference = candidate[i] - a[i];
      difference_sq += difference * difference;
      candidate_sq += candidate[i] * candidate[i];
    }
    const double residual = std::sqrt(difference_sq) /
                            std::max(std::sqrt(candidate_sq), std::numeric_limits<double>::min());
    if (!std::isfinite(residual)) {
      result.info.status = Status::kNumericalNonfinite;
      return result;
    }
    result.residual_history.push_back(residual);
    if (iteration > 5) {
      if (residual > previous_residual && relaxation > 0.125)
        relaxation *= 0.5;
      else
        relaxation += 0.1 * (1.0 - relaxation);
    }
    for (size_t i = 0; i < a.size(); ++i)
      a[i] += relaxation * (candidate[i] - a[i]);
    result.info.iterations = iteration + 1;
    result.info.residual_l2 = residual;
    if (NewtonStepConverged(options, iteration, residual, linear_tolerance)) {
      result.info.status = Status::kOk;
      result.a_wb_per_m = std::move(a);
      return FinalizeResult(&result);
    }
    previous_residual = residual;
  }
  result.info.status = Status::kNonlinearSolveNotConverged;
  result.a_wb_per_m = std::move(a);
  return FinalizeResult(&result);
}

size_t NonlinearP1Solver::csr_symbolic_reuse_count() const { return csr_symbolic_reuse_count_; }

Status NonlinearP1Solver::Initialize(NonlinearModel model)
{
  const Status status = ValidateNonlinearModel(model);
  if (status != Status::kOk)
    return status;
  model_ = std::move(model);
  csr_initialized_ = false;
  has_nonlinear_materials_ = std::any_of(
      model_.materials.begin(), model_.materials.end(), [](const NonlinearMaterial &material) {
        return material.bh_curve_index >= 0 || material.alpha_per_t2 != 0.0;
      });
  csr_symbolic_reuse_count_ = 0;
  device_assembly_.reset();
  initialized_ = true;
  return Status::kOk;
}

} // namespace gpu_femm

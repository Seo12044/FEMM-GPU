#include "app/motor_batch.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "core/profile.h"
#include "io/mesh_artifact.h"
#include "io/sample_request.h"
#include "physics/nonlinear_model.h"
#include "postprocess/postprocess.h"
#include "solvers/nonlinear_solver.cuh"
#include "solvers/options.h"

namespace gpu_femm
{

int MotorBatchAdapter(const std::string &request_path, const std::string &response_path,
                      bool include_timing, NonlinearAssemblyMode assembly_mode,
                      std::unique_ptr<MotorBatchPrefetch> prefetch)
{
  const ProfileClock::time_point total_start = ProfileClock::now();
  MotorBatchTiming timing;
  MotorBatchRequest request;
  std::string error;
  if (prefetch) {
    request = std::move(prefetch->request);
  } else {
    std::ifstream input(request_path);
    std::stringstream bytes;
    bytes << input.rdbuf();
    if (!input || !ReadMotorBatchRequestJson(bytes.str(), &request, &error)) {
      std::cerr << "FAIL motor batch request: " << error << '\n';
      return 1; // No trusted item identity exists to write a resumable response.
    }
  }
  timing.request_read_parse_seconds = ProfileSecondsSince(total_start);
  const ProfileClock::time_point preflight_start = ProfileClock::now();
  const MotorBatchArtifactLoadPlan artifact_load_plan =
      prefetch ? std::move(prefetch->load_plan) : MakeMotorBatchArtifactLoadPlan(request);
  std::vector<std::unique_ptr<MotorBatchArtifactLoad>> cached_artifact_loads(
      artifact_load_plan.canonical_paths.size());
  const size_t preflight_slot = artifact_load_plan.item_slots.front();
  if (prefetch) {
    cached_artifact_loads = std::move(prefetch->loads);
  } else {
    cached_artifact_loads[preflight_slot] = std::make_unique<MotorBatchArtifactLoad>();
    LoadMotorBatchArtifact(artifact_load_plan.representative_paths[preflight_slot],
                           cached_artifact_loads[preflight_slot].get());
  }
  // Derive the batch cap from the real CSR footprint, not a sample-count
  // heuristic.  The preflight is read-only and repeats the normal strict
  // request validation before any CUDA allocation, while reusing the parsed
  // artifact during the actual batch execution.
  size_t values_per_item = 0;
  size_t nodes_per_item = 0;
  size_t triangles_per_item = 0;
  size_t circuits_per_item = 0;
  {
    const MotorSampleRequest &first = request.items.front().request;
    const MotorBatchArtifactLoad &load = *cached_artifact_loads[preflight_slot];
    if (load.status == Status::kOk && load.artifact_sha256 == first.mesh_artifact_sha256 &&
        RequestMatchesArtifact(first, load.artifact)) {
      NonlinearModel model;
      if (!ApplySlidingBandRotorAngle(first, load.artifact, &model))
        model.nodes.clear();
      std::vector<double> initial_a(model.nodes.size(), 0.0);
      for (size_t boundary = 0; boundary < model.dirichlet_nodes.size(); ++boundary)
        initial_a[model.dirichlet_nodes[boundary]] = model.dirichlet_a_wb_per_m[boundary];
      Assembly estimate;
      if (AssembleNonlinearNewton(model, initial_a, first.circuit_currents_a, false, &estimate) ==
          Status::kOk) {
        values_per_item = estimate.values.size();
        nodes_per_item = estimate.free_nodes.size();
        triangles_per_item = model.triangles.size();
        circuits_per_item = static_cast<size_t>(model.circuit_count);
      }
    }
  }
  timing.preflight_seconds = ProfileSecondsSince(preflight_start);
  size_t free_bytes = 0, total_bytes = 0;
  const bool memory_known = cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
  constexpr size_t kMotorBatchSafetyReserveBytes = 512ULL * 1024ULL * 1024ULL;
  constexpr int32_t kMotorBatchMaxParallelWidth = 32;
  // Values plus original/inverse diagonal, RHS, and five PCG state vectors.
  // The inverse is extra workspace; keep the original diagonal for validation.
  // Device nonlinear assembly additionally keeps one full nodal state, four
  // triangle fields (Bx, By, reluctivity, derivative), and the circuit vector
  // per item. Allow two extra numeric CSR-sized buffers for Sliced ELL padding
  // (the layout refuses padding above 2x). Static topology/contributor routes
  // remain covered by the 512 MiB safety reserve below.
  const size_t bytes_per_item =
      values_per_item == 0 || nodes_per_item == 0
          ? 0
          : sizeof(double) * (3 * values_per_item + 9 * nodes_per_item + 4 * triangles_per_item +
                              circuits_per_item + 2 * kPcgMaxCooperativeShardsPerItem + 3) +
                sizeof(int) * 5;
  const int32_t vram_limit =
      memory_known && bytes_per_item > 0
          ? static_cast<int32_t>(std::max<size_t>(
                1,
                std::min<size_t>(kMotorBatchMaxParallelWidth,
                                 free_bytes > kMotorBatchSafetyReserveBytes
                                     ? (free_bytes - kMotorBatchSafetyReserveBytes) / bytes_per_item
                                     : 1)))
          : 1;
  // v2 items are partitioned by their exact artifact-and-pose operator below.
  // The chunk cap therefore remains usable for samples that share an angle,
  // while a different angle can never inherit the prior AGE matrix.
  const int32_t effective_chunk =
      std::max(1, std::min({request.max_items_per_chunk, vram_limit, kMotorBatchMaxParallelWidth}));
  // Only paths that recur across chunks (plus the already-required preflight
  // path) remain resident.  A mixed request with thousands of distinct
  // artifacts therefore retains bounded host-memory behavior, while one
  // geometry artifact is parsed once for the whole batch regardless of CUDA
  // chunk width.
  std::vector<MotorBatchResponseDto> responses(request.items.size());
  NonlinearP1Solver cached_solver;
  std::string cached_fingerprint;
  int32_t cache_hits = 0;
  int32_t chunk_count = 0;
  int actual_parallel_width = 0;
  int batched_pcg_launches = 0;
  size_t csr_symbolic_cache_hits = 0;
  size_t accounted_csr_symbolic_reuse_count = 0;
  for (size_t first = 0; first < request.items.size();
       first += static_cast<size_t>(effective_chunk)) {
    ++chunk_count;
    const size_t end = std::min(request.items.size(), first + static_cast<size_t>(effective_chunk));
    std::vector<MotorBatchPreparedItem> prepared;
    prepared.reserve(end - first);
    const ProfileClock::time_point artifact_start = ProfileClock::now();
    const size_t candidate_count = end - first;
    std::vector<MotorBatchArtifactLoad *> artifact_by_slot(
        artifact_load_plan.canonical_paths.size(), nullptr);
    std::vector<size_t> load_slots;
    std::vector<size_t> load_index_by_slot(artifact_load_plan.canonical_paths.size(),
                                           std::numeric_limits<size_t>::max());
    for (size_t local = 0; local < candidate_count; ++local) {
      const size_t slot = artifact_load_plan.item_slots[first + local];
      if (cached_artifact_loads[slot] != nullptr) {
        artifact_by_slot[slot] = cached_artifact_loads[slot].get();
      } else if (load_index_by_slot[slot] == std::numeric_limits<size_t>::max()) {
        load_index_by_slot[slot] = load_slots.size();
        load_slots.push_back(slot);
      }
    }
    std::vector<std::unique_ptr<MotorBatchArtifactLoad>> loaded_artifacts;
    loaded_artifacts.reserve(load_slots.size());
    for (size_t ignored : load_slots) {
      (void)ignored;
      loaded_artifacts.push_back(std::make_unique<MotorBatchArtifactLoad>());
    }
    std::atomic<size_t> next_artifact{0};
    const size_t artifact_worker_count = std::min<size_t>(
        load_slots.size(), std::min<size_t>(6, std::max(1u, std::thread::hardware_concurrency())));
    auto load_artifact = [&]() {
      for (;;) {
        const size_t load_index = next_artifact.fetch_add(1, std::memory_order_relaxed);
        if (load_index >= load_slots.size())
          return;
        const size_t slot = load_slots[load_index];
        LoadMotorBatchArtifact(artifact_load_plan.representative_paths[slot],
                               loaded_artifacts[load_index].get());
      }
    };
    std::vector<std::thread> artifact_workers;
    artifact_workers.reserve(artifact_worker_count > 0 ? artifact_worker_count - 1 : 0);
    for (size_t worker = 1; worker < artifact_worker_count; ++worker) {
      try {
        artifact_workers.emplace_back(load_artifact);
      } catch (const std::system_error &) {
        break;
      }
    }
    load_artifact();
    for (std::thread &worker : artifact_workers)
      worker.join();
    for (size_t load_index = 0; load_index < load_slots.size(); ++load_index) {
      const size_t slot = load_slots[load_index];
      if (artifact_load_plan.slot_use_counts[slot] > 1) {
        cached_artifact_loads[slot] = std::move(loaded_artifacts[load_index]);
        artifact_by_slot[slot] = cached_artifact_loads[slot].get();
      } else {
        artifact_by_slot[slot] = loaded_artifacts[load_index].get();
      }
    }
    timing.artifact_read_validate_seconds += ProfileSecondsSince(artifact_start);
    for (size_t local = 0; local < candidate_count; ++local) {
      const size_t index = first + local;
      const MotorSampleRequest &item = request.items[index].request;
      MotorBatchArtifactLoad *load = artifact_by_slot[artifact_load_plan.item_slots[index]];
      responses[index].status = load == nullptr ? Status::kInternalError : load->status;
      if (responses[index].status != Status::kOk)
        continue;
      if (load->artifact_sha256 != item.mesh_artifact_sha256 ||
          !RequestMatchesArtifact(item, load->artifact)) {
        responses[index].status = Status::kInvalidArgument;
        continue;
      }
      const GpuFemmMeshArtifact &artifact = load->artifact;
      NonlinearModel model;
      if (!ApplySlidingBandRotorAngle(item, artifact, &model)) {
        responses[index].status = Status::kInvalidArgument;
        continue;
      }
      NonlinearModel tangent_model;
      if (item.tangent_requested && item.sector_performance.enabled &&
          !artifact.has_native_age_sector &&
          !MakeMotorSectorTangentModel(artifact, model, &tangent_model))
        tangent_model.nodes.clear();
      const std::string operator_key = MotorBatchOperatorKey(item, *load, model);
      prepared.push_back({index, &artifact, std::move(model), std::move(tangent_model),
                          item.circuit_currents_a, operator_key});
    }
    // A SolveBatch instance owns exactly one CSR operator.  Partition the
    // request chunk before initializing it, so only samples with identical
    // v2 artifact-and-pose operators share a CUDA batch.  The geometry-only
    // WST plans are geometry-only CPU work.  Prepare a bounded window of them
    // ahead while the current nonlinear state occupies the GPU.  Four plans
    // keep memory bounded while allowing independent posed geometries to use
    // otherwise-idle host cores; each result is released after its group is
    // consumed.  Solve and postprocess math remain unchanged.
    const std::vector<std::vector<size_t>> operator_groups =
        GroupMotorBatchPreparedItemsByOperator(prepared);
    struct PreparedFrozenPostprocessPlan {
      FrozenPostprocessOptions options;
      FrozenPostprocessPlan plan;
      double compute_seconds = 0.0;
      bool available = false;
    };
    auto build_group_frozen_plan = [&](size_t group_index) {
      PreparedFrozenPostprocessPlan result;
      const std::vector<size_t> &group = operator_groups[group_index];
      if (group.empty())
        return result;
      const MotorBatchPreparedItem &representative = prepared[group.front()];
      const MotorSampleRequest &item = request.items[representative.request_index].request;
      if (representative.artifact->has_sliding_band)
        return result;
      result.options.selected_group_number = item.selected_group_number;
      result.options.air_group_number = item.air_group_number;
      result.options.max_mask_iterations = 4096;
      result.options.airgap_radius_m = item.airgap_radius_mm * 1e-3;
      for (double angle : item.airgap_angles_deg)
        result.options.airgap_angles_rad.push_back(angle * 3.141592653589793238462643383279502884 /
                                                   180.0);
      if (item.sector_performance.enabled) {
        MapSectorAirgapAnglesToRepresentative(representative.artifact->sector_start_angle_deg *
                                                  3.141592653589793238462643383279502884 / 180.0,
                                              representative.artifact->sector_boundary_sign,
                                              &result.options);
      }
      const ProfileClock::time_point start = ProfileClock::now();
      BuildFrozenPostprocessPlan(representative.model, result.options, &result.plan);
      result.compute_seconds = ProfileSecondsSince(start);
      result.available = true;
      return result;
    };
    auto launch_group_frozen_plan = [&](size_t group_index) {
      try {
        return std::async(std::launch::async, build_group_frozen_plan, group_index);
      } catch (const std::system_error &) {
        return std::async(std::launch::deferred, build_group_frozen_plan, group_index);
      }
    };
    const size_t frozen_plan_prefetch_window =
        std::min<size_t>(operator_groups.size(),
                         std::min<size_t>(4, std::max(1u, std::thread::hardware_concurrency())));
    std::vector<std::future<PreparedFrozenPostprocessPlan>> pending_frozen_plans(
        operator_groups.size());
    for (size_t group_index = 0; group_index < frozen_plan_prefetch_window; ++group_index) {
      pending_frozen_plans[group_index] = launch_group_frozen_plan(group_index);
    }
    for (size_t group_index = 0; group_index < operator_groups.size(); ++group_index) {
      const std::vector<size_t> &group = operator_groups[group_index];
      if (group.empty())
        continue;
      const ProfileClock::time_point plan_wait_start = ProfileClock::now();
      PreparedFrozenPostprocessPlan prepared_frozen_plan = pending_frozen_plans[group_index].get();
      const double plan_wait_seconds = ProfileSecondsSince(plan_wait_start);
      timing.postprocess_plan_compute_seconds += prepared_frozen_plan.compute_seconds;
      timing.postprocess_plan_seconds += plan_wait_seconds;
      timing.postprocess_seconds += plan_wait_seconds;
      const size_t launch_index = group_index + frozen_plan_prefetch_window;
      if (launch_index < operator_groups.size()) {
        pending_frozen_plans[launch_index] = launch_group_frozen_plan(launch_index);
      }
      const MotorBatchPreparedItem &representative = prepared[group.front()];
      if (cached_fingerprint != representative.operator_key) {
        const ProfileClock::time_point initialize_start = ProfileClock::now();
        const Status initialize_status = cached_solver.Initialize(representative.model);
        timing.solver_initialize_seconds += ProfileSecondsSince(initialize_start);
        cached_fingerprint = initialize_status == Status::kOk ? representative.operator_key : "";
        if (initialize_status != Status::kOk) {
          for (size_t prepared_index : group)
            responses[prepared[prepared_index].request_index].status = initialize_status;
          continue;
        }
        // Initialize starts a fresh solver-local counter.  Keep the response
        // metric process-wide so multiple v2 pose groups do not hide prior
        // symbolic CSR reuse.
        accounted_csr_symbolic_reuse_count = 0;
        cache_hits += static_cast<int32_t>(group.size() - 1);
      } else {
        cache_hits += static_cast<int32_t>(group.size());
      }
      actual_parallel_width = std::max(actual_parallel_width, static_cast<int>(group.size()));
      std::vector<std::vector<double>> group_currents;
      group_currents.reserve(group.size());
      for (size_t prepared_index : group)
        group_currents.push_back(prepared[prepared_index].currents);
      const NonlinearOptions options = ProductionSampleOptions();
      NonlinearBatchTiming chunk_timing;
      std::vector<NonlinearSolveResult> chunk_solutions = cached_solver.SolveBatch(
          group_currents, options, &batched_pcg_launches, &chunk_timing, assembly_mode);
      const size_t current_csr_symbolic_reuse_count = cached_solver.csr_symbolic_reuse_count();
      if (current_csr_symbolic_reuse_count >= accounted_csr_symbolic_reuse_count) {
        csr_symbolic_cache_hits +=
            current_csr_symbolic_reuse_count - accounted_csr_symbolic_reuse_count;
      }
      accounted_csr_symbolic_reuse_count = current_csr_symbolic_reuse_count;
      timing.nonlinear.host_assembly_seconds += chunk_timing.host_assembly_seconds;
      timing.nonlinear.device_assembly_seconds += chunk_timing.device_assembly_seconds;
      timing.nonlinear.gpu_upload_seconds += chunk_timing.gpu_upload_seconds;
      timing.nonlinear.gpu_kernel_sync_seconds += chunk_timing.gpu_kernel_sync_seconds;
      timing.nonlinear.gpu_download_seconds += chunk_timing.gpu_download_seconds;
      AccumulatePreconditionerStats(timing.nonlinear.preconditioner, chunk_timing.preconditioner);
      timing.nonlinear.interleaved_spmv_solves += chunk_timing.interleaved_spmv_solves;
      timing.nonlinear.state_update_seconds += chunk_timing.state_update_seconds;
      timing.nonlinear.finalize_seconds += chunk_timing.finalize_seconds;
      timing.nonlinear.recovery_seconds += chunk_timing.recovery_seconds;
      for (size_t local = 0; local < group.size(); ++local) {
        const MotorBatchPreparedItem &prepared_item = prepared[group[local]];
        const size_t index = prepared_item.request_index;
        const MotorSampleRequest &item = request.items[index].request;
        NonlinearSolveResult &solution = chunk_solutions[local];
        timing.nonlinear_pcg_iterations += solution.linear_iterations;
        if (solution.info.status != Status::kOk) {
          responses[index] =
              MakeMotorBatchResponseDto(solution.info.status, item, &solution, nullptr);
          continue;
        }
        FrozenPostprocessOptions post_options;
        post_options.selected_group_number = item.selected_group_number;
        post_options.air_group_number = item.air_group_number;
        post_options.max_mask_iterations = 4096;
        post_options.airgap_radius_m = item.airgap_radius_mm * 1e-3;
        for (double angle : item.airgap_angles_deg)
          post_options.airgap_angles_rad.push_back(angle * 3.141592653589793238462643383279502884 /
                                                   180.0);
        FrozenPostprocessOptions base_post_options = post_options;
        std::vector<int8_t> base_airgap_repeat_signs;
        if (item.sector_performance.enabled && prepared_item.artifact->has_native_age_sector) {
          base_airgap_repeat_signs = MapNativeAgeSectorAirgapAnglesToRepresentative(
              *prepared_item.artifact, &base_post_options);
        } else if (item.sector_performance.enabled && !prepared_item.artifact->has_sliding_band) {
          base_airgap_repeat_signs = MapSectorAirgapAnglesToRepresentative(
              prepared_item.artifact->sector_start_angle_deg *
                  3.141592653589793238462643383279502884 / 180.0,
              prepared_item.artifact->sector_boundary_sign, &base_post_options);
        }
        const ProfileClock::time_point postprocess_start = ProfileClock::now();
        FrozenPostprocessPlan item_frozen_plan;
        const FrozenPostprocessPlan *frozen_plan = nullptr;
        if (!prepared_item.artifact->has_sliding_band) {
          if (prepared_frozen_plan.available &&
              FrozenPostprocessPlanMatches(prepared_frozen_plan.plan, prepared_item.model,
                                           base_post_options)) {
            frozen_plan = &prepared_frozen_plan.plan;
          } else {
            const ProfileClock::time_point plan_start = ProfileClock::now();
            BuildFrozenPostprocessPlan(prepared_item.model, base_post_options, &item_frozen_plan);
            const double plan_seconds = ProfileSecondsSince(plan_start);
            timing.postprocess_plan_compute_seconds += plan_seconds;
            timing.postprocess_plan_seconds += plan_seconds;
            frozen_plan = &item_frozen_plan;
          }
        }
        const ProfileClock::time_point base_postprocess_start = ProfileClock::now();
        FrozenPostprocessResult postprocess;
        if (prepared_item.artifact->has_sliding_band) {
          postprocess =
              ComputeAirGapElementPostprocess(prepared_item.model, solution, base_post_options);
        } else if (frozen_plan->status != Status::kOk) {
          postprocess.status = frozen_plan->status;
        } else {
          postprocess = ComputeFrozenPostprocess(prepared_item.model, solution, base_post_options,
                                                 frozen_plan);
        }
        timing.base_postprocess_seconds += ProfileSecondsSince(base_postprocess_start);
        if (postprocess.status == Status::kOk && !base_airgap_repeat_signs.empty() &&
            !ApplySectorAirgapRepeatSigns(base_airgap_repeat_signs, &postprocess))
          postprocess.status = Status::kInternalError;
        timing.postprocess_seconds += ProfileSecondsSince(postprocess_start);
        responses[index] =
            MakeMotorBatchResponseDto(postprocess.status, item, &solution,
                                      postprocess.status == Status::kOk ? &postprocess : nullptr);
        if (item.sector_performance.enabled) {
          ApplyBaseSectorReconstruction(*prepared_item.artifact, item, &responses[index]);
          responses[index].sector_reconstruction_status = responses[index].status;
        }
        if (postprocess.status == Status::kOk && item.tangent_requested) {
          responses[index].tangent_status = Status::kOk;
          FrozenPostprocessOptions tangent_post_options = post_options;
          std::vector<int8_t> tangent_airgap_repeat_signs;
          if (item.sector_performance.enabled && !prepared_item.artifact->has_sliding_band) {
            tangent_airgap_repeat_signs = MapSectorAirgapAnglesToRepresentative(
                prepared_item.artifact->sector_start_angle_deg *
                    3.141592653589793238462643383279502884 / 180.0,
                static_cast<int8_t>(
                    (prepared_item.artifact->suspension_spatial_order % 2) == 0 ? 1 : -1),
                &tangent_post_options);
          }
          FrozenPostprocessPlan tangent_frozen_plan;
          const FrozenPostprocessPlan *tangent_plan = frozen_plan;
          if (!prepared_item.artifact->has_sliding_band &&
              !FrozenPostprocessPlanMatches(*tangent_plan, prepared_item.model,
                                            tangent_post_options)) {
            responses[index].tangent_status = BuildFrozenPostprocessPlan(
                prepared_item.model, tangent_post_options, &tangent_frozen_plan);
            tangent_plan = &tangent_frozen_plan;
          }
          NonlinearP1Solver sector_tangent_solver;
          NonlinearP1Solver *tangent_solver = &cached_solver;
          if (item.sector_performance.enabled) {
            if (prepared_item.tangent_model.nodes.empty()) {
              responses[index].tangent_status = Status::kInvalidArgument;
            } else {
              responses[index].tangent_status =
                  sector_tangent_solver.Initialize(prepared_item.tangent_model);
              tangent_solver = &sector_tangent_solver;
            }
          }
          const ProfileClock::time_point tangent_operator_start = ProfileClock::now();
          std::vector<TangentSolveResult> tangent_results =
              responses[index].tangent_status == Status::kOk
                  ? tangent_solver->SolveTangentMultiRhs(solution, item.tangent_multi_rhs.rhs,
                                                         options, &batched_pcg_launches)
                  : std::vector<TangentSolveResult>{};
          timing.tangent_operator_seconds += ProfileSecondsSince(tangent_operator_start);
          if (responses[index].tangent_status == Status::kOk &&
              tangent_results.size() != item.tangent_multi_rhs.rhs.size())
            responses[index].tangent_status = Status::kInternalError;
          if (item.sector_performance.enabled && !tangent_results.empty())
            responses[index].tangent_operator_evidence = tangent_results.front().operator_evidence;
          for (TangentSolveResult &tangent : tangent_results) {
            if (tangent.info.status != Status::kOk) {
              responses[index].tangent_status = tangent.info.status;
              continue;
            }
            const ProfileClock::time_point tangent_postprocess_start = ProfileClock::now();
            TangentPostprocessResult tangent_postprocess =
                prepared_item.artifact->has_sliding_band
                    ? ComputeAirGapElementTangentPostprocess(
                          prepared_item.model,
                          item.sector_performance.enabled ? prepared_item.tangent_model
                                                          : prepared_item.model,
                          solution, tangent.field_derivative, tangent_post_options)
                    : ComputeFrozenTangentPostprocess(
                          prepared_item.model, solution, tangent.field_derivative,
                          tangent_post_options, tangent_plan->mask, tangent_plan);
            const double tangent_postprocess_seconds =
                ProfileSecondsSince(tangent_postprocess_start);
            timing.postprocess_seconds += tangent_postprocess_seconds;
            timing.tangent_postprocess_seconds += tangent_postprocess_seconds;
            if (tangent_postprocess.status != Status::kOk) {
              responses[index].tangent_status = tangent_postprocess.status;
              continue;
            }
            if (!tangent_airgap_repeat_signs.empty() &&
                !ApplySectorTangentAirgapRepeatSigns(tangent_airgap_repeat_signs,
                                                     &tangent_postprocess)) {
              responses[index].tangent_status = Status::kInternalError;
              continue;
            }
            TangentRhsResponseDto dto;
            dto.name = tangent.name;
            dto.circuit_current_derivative_a_per_a = tangent.circuit_current_derivative_a_per_a;
            dto.force_x_n_per_a = tangent_postprocess.force_x_n_per_a;
            dto.force_y_n_per_a = tangent_postprocess.force_y_n_per_a;
            dto.torque_nm_per_a = tangent_postprocess.torque_nm_per_a;
            dto.circuit_flux_linkage_derivative_wb_per_a =
                tangent.field_derivative.circuit_flux_linkage_wb;
            dto.airgap_radial_flux_density_derivative_t_per_a =
                std::move(tangent_postprocess.airgap_radial_flux_density_t_per_a);
            dto.linear_convergence = tangent.info;
            if (item.sector_performance.enabled &&
                !ApplyTangentSectorReconstruction(*prepared_item.artifact, responses[index],
                                                  &dto)) {
              responses[index].tangent_status = Status::kInternalError;
              continue;
            }
            responses[index].tangent_rhs.push_back(std::move(dto));
          }
          if (responses[index].tangent_status == Status::kOk &&
              responses[index].tangent_rhs.size() != item.tangent_multi_rhs.rhs.size())
            responses[index].tangent_status = Status::kInternalError;
          if (item.sector_performance.enabled)
            responses[index].sector_reconstruction_status = responses[index].tangent_status;
        }
      }
    }
  }
  timing.measured_before_response_write_seconds = ProfileSecondsSince(total_start);
  if (!WriteMotorBatchResponse(response_path, &request, responses, request.max_items_per_chunk,
                               effective_chunk, cache_hits, chunk_count, csr_symbolic_cache_hits,
                               actual_parallel_width, batched_pcg_launches,
                               include_timing ? &timing : nullptr))
    return 1;
  return std::all_of(
             responses.begin(), responses.end(),
             [](const MotorBatchResponseDto &response) { return response.status == Status::kOk; })
             ? 0
             : 1;
}

} // namespace gpu_femm

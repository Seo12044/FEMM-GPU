#include "commands.h"
#include "comparison.h"
#include "fixtures.h"
#include "suites.h"
#include "solvers/linear_solver.cuh"
#include "solvers/nonlinear_solver.cuh"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace gpu_femm
{

int SelfTest()
{
  int failures = PcgSpmvLayoutTests() + PcgRegressionTests(67) + PcgRegressionTests(129) +
                 NonlinearToleranceTests() + BlockPreconditionerTests() +
                 PcgRegressionTests(67, PcgPreconditioner::kBlockJacobi) +
                 PcgRegressionTests(129, PcgPreconditioner::kBlockJacobi);
  auto expect = [&failures](bool condition, const std::string &message) {
    if (!condition) {
      std::cerr << "FAIL: " << message << '\n';
      ++failures;
    }
  };

  failures += ProtocolContractTests();
  failures += PhysicsRegressionTests();

  Assembly three_by_three;
  three_by_three.row_offsets = {0, 2, 5, 7};
  three_by_three.column_indices = {0, 1, 0, 1, 2, 1, 2};
  three_by_three.values = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0};
  three_by_three.diagonal = {4.0, 4.0, 3.0};
  three_by_three.free_nodes = {0, 1, 2};
  GpuCsrSolver csr_solver;
  Status csr_status = csr_solver.Initialize(three_by_three);
  expect(csr_status == Status::kOk, "3x3 CSR initialization");
  std::vector<double> csr_solution;
  const SolveInfo csr_info = csr_solver.Solve({2.0, 4.0, 7.0}, 1e-13, 64, &csr_solution);
  expect(csr_info.status == Status::kOk, "3x3 PCG convergence");
  expect(csr_solution.size() == 3 && Near(csr_solution[0], 1.0) && Near(csr_solution[1], 2.0) &&
             Near(csr_solution[2], 3.0),
         "3x3 PCG reference parity");
  std::vector<SolveInfo> small_batch_infos;
  std::vector<std::vector<double>> small_batch_solutions;
  int small_batch_launches = 0;
  expect(csr_solver.SolveBatch({three_by_three, three_by_three}, {{2.0, 4.0, 7.0}, {2.0, 4.0, 7.0}},
                               1e-13, 64, &small_batch_infos, &small_batch_solutions,
                               &small_batch_launches) == Status::kOk,
         "3x3 batch buffer allocation");
  std::vector<SolveInfo> shared_operator_infos;
  std::vector<std::vector<double>> shared_operator_solutions;
  SharedOperatorBatchEvidence shared_operator_evidence;
  int shared_operator_launches = 0;
  const Status shared_operator_status =
      csr_solver.UpdateValues(three_by_three) == Status::kOk
          ? csr_solver.SolveCurrentOperatorBatch(
                {{2.0, 4.0, 7.0}, {4.0, 8.0, 14.0}}, 1e-13, 64, &shared_operator_infos,
                &shared_operator_solutions, &shared_operator_launches, &shared_operator_evidence)
          : Status::kInternalError;
  expect(shared_operator_status == Status::kOk && shared_operator_solutions.size() == 2 &&
             shared_operator_solutions[0] == std::vector<double>({1.0, 2.0, 3.0}) &&
             shared_operator_solutions[1] == std::vector<double>({2.0, 4.0, 6.0}) &&
             shared_operator_launches == 1 &&
             shared_operator_evidence.physical_operator_count == 1 &&
             shared_operator_evidence.operator_upload_count == 1 &&
             shared_operator_evidence.rhs_count == 2,
         "shared tangent operator stores/uploads one K and solves two RHS in one launch");
  std::vector<SolveInfo> legacy_after_shared_infos;
  std::vector<std::vector<double>> legacy_after_shared_solutions;
  int legacy_after_shared_launches = 0;
  expect(csr_solver.SolveBatch({three_by_three, three_by_three},
                               {{2.0, 4.0, 7.0}, {4.0, 8.0, 14.0}}, 1e-13, 64,
                               &legacy_after_shared_infos, &legacy_after_shared_solutions,
                               &legacy_after_shared_launches) == Status::kOk &&
             legacy_after_shared_solutions == shared_operator_solutions,
         "legacy replicated batch safely reallocates after shared-operator mode");

  // 257 exercises the strided reduction tail beyond one 256-thread block.
  Assembly reduction_tail;
  constexpr int kReductionTailN = kPcgBlockThreads + 1;
  reduction_tail.row_offsets.resize(kReductionTailN + 1);
  reduction_tail.column_indices.resize(kReductionTailN);
  reduction_tail.values.assign(kReductionTailN, 1.0);
  reduction_tail.diagonal.assign(kReductionTailN, 1.0);
  reduction_tail.free_nodes.resize(kReductionTailN);
  std::vector<double> reduction_rhs(kReductionTailN);
  for (int i = 0; i < kReductionTailN; ++i) {
    reduction_tail.row_offsets[i] = i;
    reduction_tail.column_indices[i] = i;
    reduction_tail.free_nodes[i] = i;
    reduction_rhs[i] = static_cast<double>((i % 11) - 5);
  }
  reduction_tail.row_offsets[kReductionTailN] = kReductionTailN;
  std::vector<SolveInfo> resized_batch_infos;
  std::vector<std::vector<double>> resized_batch_solutions;
  int resized_batch_launches = 0;
  const Status resized_batch_status =
      csr_solver.Initialize(reduction_tail) == Status::kOk
          ? csr_solver.SolveBatch({reduction_tail, reduction_tail}, {reduction_rhs, reduction_rhs},
                                  1e-13, 16, &resized_batch_infos, &resized_batch_solutions,
                                  &resized_batch_launches)
          : Status::kInternalError;
  expect(resized_batch_status == Status::kOk && resized_batch_solutions.size() == 2 &&
             resized_batch_solutions[0] == reduction_rhs &&
             resized_batch_solutions[1] == reduction_rhs,
         "batch buffers resize after CSR topology change");
  GpuCsrSolver reduction_solver;
  std::vector<double> reduction_solution;
  const Status reduction_initialize = reduction_solver.Initialize(reduction_tail);
  const SolveInfo reduction_info =
      reduction_initialize == Status::kOk
          ? reduction_solver.Solve(reduction_rhs, 1e-13, 16, &reduction_solution)
          : SolveInfo{};
  bool reduction_matches =
      reduction_info.status == Status::kOk && reduction_solution.size() == reduction_rhs.size();
  for (size_t index = 0; reduction_matches && index < reduction_rhs.size(); ++index)
    reduction_matches = reduction_solution[index] == reduction_rhs[index];
  expect(reduction_matches, "deterministic reduction handles non-multiple-of-block-width RHS");

  Assembly indefinite;
  indefinite.row_offsets = {0, 1};
  indefinite.column_indices = {0};
  indefinite.values = {-1.0};
  indefinite.diagonal = {-1.0};
  indefinite.free_nodes = {0};
  GpuCsrSolver breakdown_solver;
  Status breakdown_status = breakdown_solver.Initialize(indefinite);
  expect(breakdown_status == Status::kOk, "breakdown fixture initialization");
  std::vector<double> breakdown_solution;
  const SolveInfo breakdown_info = breakdown_solver.Solve({1.0}, 1e-13, 8, &breakdown_solution);
  expect(breakdown_info.status == Status::kLinearSolveBreakdown,
         "indefinite matrix returns LINEAR_SOLVE_BREAKDOWN");

  LinearP1Solver solver;
  Status initialize_status = solver.Initialize(UnitSquareFixture());
  expect(initialize_status == Status::kOk,
         std::string("fixture initialization: ") + StatusName(initialize_status));
  if (initialize_status != Status::kOk)
    return failures + 1;

  const SolveResult positive = solver.Solve(12.0);
  expect(positive.info.status == Status::kOk,
         std::string("positive solve: ") + StatusName(positive.info.status));
  const double expected_a[5] = {0.0, 0.0, 0.0, 0.0, 1.0};
  const double expected_bx[4] = {2.0, 0.0, -2.0, 0.0};
  const double expected_by[4] = {0.0, 2.0, 0.0, -2.0};
  expect(positive.a_wb_per_m.size() == 5 && positive.bx_t.size() == 4 && positive.by_t.size() == 4,
         "positive result dimensions");
  if (positive.a_wb_per_m.size() == 5 && positive.bx_t.size() == 4 && positive.by_t.size() == 4) {
    for (int i = 0; i < 5; ++i) {
      expect(Near(positive.a_wb_per_m[i], expected_a[i]), "nodal A parity");
    }
    for (int i = 0; i < 4; ++i) {
      expect(Near(positive.bx_t[i], expected_bx[i]), "Bx sign/parity");
      expect(Near(positive.by_t[i], expected_by[i]), "By sign/parity");
    }
  }
  expect(Near(positive.flux_linkage_wb, 1.0 / 3.0), "flux-linkage parity");
  expect(positive.info.residual_l2 <= 1e-13, "linear residual tolerance");

  const SolveResult negative = solver.Solve(-12.0);
  expect(negative.info.status == Status::kOk, "negative-current solve");
  expect(negative.a_wb_per_m.size() == 5 && negative.bx_t.size() == 4 && negative.by_t.size() == 4,
         "negative result dimensions");
  if (negative.a_wb_per_m.size() == 5 && negative.bx_t.size() == 4 && negative.by_t.size() == 4) {
    for (int i = 0; i < 5; ++i) {
      expect(Near(negative.a_wb_per_m[i], -expected_a[i]), "negative A sign");
    }
    for (int i = 0; i < 4; ++i) {
      expect(Near(negative.bx_t[i], -expected_bx[i]), "negative Bx sign");
      expect(Near(negative.by_t[i], -expected_by[i]), "negative By sign");
    }
  }
  expect(Near(negative.flux_linkage_wb, -1.0 / 3.0), "negative flux sign");

  const SolveResult tiny = solver.Solve(1e-14);
  expect(tiny.info.status == Status::kOk && tiny.info.iterations > 0,
         "small nonzero RHS is solved, not accepted at x=0");
  expect(tiny.a_wb_per_m.size() == 5 && Near(tiny.a_wb_per_m[4], 1e-14 / 12.0, 1e-28),
         "small nonzero RHS parity");
  const SolveResult subnormal_square = solver.Solve(1e-200);
  expect(subnormal_square.info.status == Status::kOk && subnormal_square.info.iterations > 0,
         "subnormal-square RHS is solved, not accepted at x=0");
  expect(subnormal_square.a_wb_per_m.size() == 5 &&
             Near(subnormal_square.a_wb_per_m[4], 1e-200 / 12.0, 1e-214),
         "subnormal-square RHS parity");
  const SolveResult invalid_tolerance =
      solver.Solve(12.0, std::numeric_limits<double>::infinity(), 8);
  expect(invalid_tolerance.info.status == Status::kInvalidArgument,
         "infinite tolerance returns INVALID_ARGUMENT");

  const SolveResult not_converged = solver.Solve(12.0, 1e-13, 0);
  expect(not_converged.info.status == Status::kLinearSolveNotConverged,
         "iteration limit returns LINEAR_SOLVE_NOT_CONVERGED");

  NonlinearP1Solver nonlinear_solver;
  const Status nonlinear_initialize = nonlinear_solver.Initialize(NonlinearThreeRegionFixture());
  expect(nonlinear_initialize == Status::kOk,
         std::string("nonlinear fixture initialization: ") + StatusName(nonlinear_initialize));
  if (nonlinear_initialize == Status::kOk) {
    const NonlinearSolveResult nonlinear = nonlinear_solver.Solve(2.0);
    expect(nonlinear.info.status == Status::kOk,
           std::string("nonlinear PM+coil solve: ") + StatusName(nonlinear.info.status));
    expect(nonlinear.a_wb_per_m.size() == 5 && nonlinear.bx_t.size() == 4 &&
               nonlinear.by_t.size() == 4 && nonlinear.residual_history.size() >= 2,
           "nonlinear result dimensions and history");
    expect(std::isfinite(nonlinear.flux_linkage_wb), "nonlinear circuit flux");
    if (nonlinear.info.status == Status::kOk) {
      const NonlinearSolveResult warm = nonlinear_solver.Solve(2.0, {}, &nonlinear.a_wb_per_m);
      expect(warm.info.status == Status::kOk, "nonlinear warm-start solve");
      expect(nonlinear_solver.csr_symbolic_reuse_count() > 0,
             "nonlinear Newton corrections reuse uploaded CSR structure");
      expect(warm.info.iterations <= nonlinear.info.iterations,
             "warm start does not add Newton steps");
      expect(warm.a_wb_per_m.size() == nonlinear.a_wb_per_m.size(), "warm result dimensions");
      if (warm.a_wb_per_m.size() == nonlinear.a_wb_per_m.size()) {
        for (size_t i = 0; i < warm.a_wb_per_m.size(); ++i)
          expect(Near(warm.a_wb_per_m[i], nonlinear.a_wb_per_m[i], 1e-9),
                 "warm-start solution parity");
      }
    }
    NonlinearOptions no_iterations;
    no_iterations.max_newton_iterations = 0;
    const NonlinearSolveResult nonlinear_limited = nonlinear_solver.Solve(2.0, no_iterations);
    expect(nonlinear_limited.info.status == Status::kNonlinearSolveNotConverged,
           "nonlinear iteration cap returns NONLINEAR_SOLVE_NOT_CONVERGED");
    const NonlinearSolveResult guarded = nonlinear_solver.SolveSafeguarded({2.0}, {});
    expect(guarded.info.status == Status::kOk &&
               guarded.verified_equation_residual_relative <= 1e-12,
           "safeguarded Newton verifies the nonlinear equations at the returned state");
    expect(guarded.a_wb_per_m.size() == nonlinear.a_wb_per_m.size() &&
               Near(guarded.flux_linkage_wb, nonlinear.flux_linkage_wb, 1e-9),
           "safeguarded nonlinear PM/coil recovery matches reference flux");
    const NonlinearSolveResult guarded_limited =
        nonlinear_solver.SolveSafeguarded({2.0}, no_iterations);
    expect(guarded_limited.info.status == Status::kNonlinearSolveNotConverged,
           "safeguarded recovery obeys the nonlinear iteration budget");
    no_iterations.recover_nonconvergence = true;
    const auto bounded = nonlinear_solver.SolveBatch({{2.0}}, no_iterations);
    expect(bounded[0].info.status == Status::kNonlinearSolveNotConverged &&
               bounded[0].info.iterations == 0,
           "automatic recovery cannot bypass a zero iteration budget");
    const auto normal = nonlinear_solver.SolveBatch({{2.0}, {-2.0}}, ProductionSampleOptions());
    NonlinearOptions no_recovery = ProductionSampleOptions();
    no_recovery.recover_nonconvergence = false;
    const auto control = nonlinear_solver.SolveBatch({{2.0}, {-2.0}}, no_recovery);
    expect(normal[0].info.status == Status::kOk && normal[1].info.status == Status::kOk &&
               normal[0].a_wb_per_m == control[0].a_wb_per_m &&
               normal[1].a_wb_per_m == control[1].a_wb_per_m,
           "recovery leaves already converged batch states bit-identical");
    expect(nonlinear_solver.SolveSafeguarded({std::numeric_limits<double>::infinity()}, {})
                   .info.status == Status::kInvalidArgument,
           "safeguarded recovery rejects nonfinite current input");
  }

  NonlinearModel invalid_circuit = NonlinearTwoCircuitFixture();
  invalid_circuit.materials[2].circuit_index = 2;
  expect(ValidateNonlinearModel(invalid_circuit) == Status::kInvalidMaterial,
         "out-of-range nonlinear circuit label is rejected");
  NonlinearP1Solver two_circuit_solver;
  const Status two_circuit_initialize = two_circuit_solver.Initialize(NonlinearTwoCircuitFixture());
  expect(two_circuit_initialize == Status::kOk, "two-circuit fixture initialization");
  if (two_circuit_initialize == Status::kOk) {
    const NonlinearSolveResult circuit_0 = two_circuit_solver.Solve(std::vector<double>{2.0, 0.0});
    const NonlinearSolveResult circuit_1 = two_circuit_solver.Solve(std::vector<double>{0.0, -3.0});
    const NonlinearSolveResult combined = two_circuit_solver.Solve(std::vector<double>{2.0, -3.0});
    expect(circuit_0.info.status == Status::kOk && circuit_1.info.status == Status::kOk &&
               combined.info.status == Status::kOk,
           "two-circuit signed source solves");
    expect(combined.circuit_currents_a.size() == 2 && combined.circuit_flux_linkage_wb.size() == 2,
           "two-circuit ordered current and flux dimensions");
    if (circuit_0.a_wb_per_m.size() == 5 && circuit_1.a_wb_per_m.size() == 5 &&
        combined.a_wb_per_m.size() == 5) {
      expect(Near(circuit_0.a_wb_per_m[4], 0.5) && Near(circuit_1.a_wb_per_m[4], -0.375) &&
                 Near(combined.a_wb_per_m[4], 0.125) &&
                 Near(combined.a_wb_per_m[4], circuit_0.a_wb_per_m[4] + circuit_1.a_wb_per_m[4]),
             "two-circuit independent signed source superposition");
    }
    if (combined.circuit_flux_linkage_wb.size() == 2) {
      expect(Near(combined.circuit_flux_linkage_wb[0], 0.125) &&
                 Near(combined.circuit_flux_linkage_wb[1], 0.0625) &&
                 std::isnan(combined.flux_linkage_wb),
             "two-circuit independent flux linkage vector");
    }
    const NonlinearSolveResult wrong_current_count =
        two_circuit_solver.Solve(std::vector<double>{2.0});
    expect(wrong_current_count.info.status == Status::kInvalidArgument,
           "wrong nonlinear circuit-current vector length is rejected");
  }

  solver.Solve(12.0); // warm up all runtime paths before memory accounting.
  size_t free_before = 0;
  size_t total_before = 0;
  const cudaError_t memory_before_status = cudaMemGetInfo(&free_before, &total_before);
  expect(memory_before_status == cudaSuccess, "pre-repeat cudaMemGetInfo");
  for (int repeat = 0; repeat < 100; ++repeat) {
    const SolveResult repeated = solver.Solve(12.0);
    expect(SameBits(positive, repeated), "100-run bitwise determinism");
  }
  size_t free_after = 0;
  size_t total_after = 0;
  const cudaError_t memory_after_status = cudaMemGetInfo(&free_after, &total_after);
  expect(memory_after_status == cudaSuccess, "post-repeat cudaMemGetInfo");
  if (memory_before_status == cudaSuccess && memory_after_status == cudaSuccess) {
    expect(free_after >= free_before, "no GPU-memory growth across 100 solves");
  }

  if (failures == 0) {
    std::cout << "PASS gpu_linear_p1_selftest\n"
              << "  status=" << StatusName(positive.info.status)
              << " iterations=" << positive.info.iterations
              << " residual=" << positive.info.residual_l2 << '\n'
              << "  A_center=" << positive.a_wb_per_m.at(4)
              << " flux_Wb=" << positive.flux_linkage_wb << '\n'
              << "  deterministic_repeats=100 gpu_memory_delta_bytes="
              << static_cast<long long>(free_before) - static_cast<long long>(free_after) << '\n';
  }
  return failures == 0 ? 0 : 1;
}

} // namespace gpu_femm

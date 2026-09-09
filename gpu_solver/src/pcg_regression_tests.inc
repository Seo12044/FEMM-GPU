// Included by SelfTest after the CSR solver is defined. Synthetic SPD systems
// exercise row/shard tails and cross-shard dependencies without any FEMM/CAD run.
int PcgRegressionTests(int side)
{
  int failures = 0;
  auto expect = [&failures](bool ok, const char* message) {
    if (!ok) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
  };
  expect(PcgCooperativeShardCount(50000, 1, 34) == 32
      && PcgCooperativeShardCount(50000, 2, 34) == 16
      && PcgCooperativeShardCount(50000, 4, 34) == 8
      && PcgCooperativeShardCount(4489, 1, 108) == 8
      && PcgCooperativeShardCount(50000, 4, 4) == 0
      && PcgCooperativeShardCount(2048, 1, 34) == 0
      && PcgCooperativeShardCount(50000, 32, 108) == 0,
      "cooperative partition respects shape/device bounds");
  auto multiply = [](const Assembly& matrix, const std::vector<double>& x) {
    std::vector<double> rhs(x.size(), 0.0);
    for (size_t row = 0; row < x.size(); ++row)
      for (int entry = matrix.row_offsets[row]; entry < matrix.row_offsets[row + 1]; ++entry)
        rhs[row] += matrix.values[entry] * x[matrix.column_indices[entry]];
    return rhs;
  };
  auto verified = [&](const Assembly& matrix, const std::vector<double>& rhs,
                      const std::vector<double>& solution, const SolveInfo& info) {
    if (info.status != Status::kOk || solution.size() != rhs.size()) return false;
    const std::vector<double> ax = multiply(matrix, solution);
    double error = 0.0, scale = 0.0;
    for (size_t row = 0; row < rhs.size(); ++row) {
      if (!std::isfinite(solution[row])) return false;
      error = std::hypot(error, ax[row] - rhs[row]);
      scale = std::hypot(scale, rhs[row]);
    }
    return error <= 1.01e-10 * scale;
  };

  // 67*67 and 129*129 exercise 8 and up to 32 shards with non-divisible tails.
  // A five-point stencil couples rows owned by different shards.
  const int count = side * side;
  Assembly matrix;
  matrix.row_offsets.push_back(0);
  std::vector<double> exact(count);
  for (int row = 0; row < count; ++row) {
    matrix.free_nodes.push_back(row);
    const double diagonal = 4.05 + 0.01 * (row % 7);
    matrix.diagonal.push_back(diagonal);
    auto entry = [&](int column, double value) {
      matrix.column_indices.push_back(column); matrix.values.push_back(value);
    };
    if (row >= side) entry(row - side, -1.0);
    if (row % side != 0) entry(row - 1, -1.0);
    entry(row, diagonal);
    if (row % side != side - 1) entry(row + 1, -1.0);
    if (row < count - side) entry(row + side, -1.0);
    matrix.row_offsets.push_back(static_cast<int32_t>(matrix.values.size()));
    exact[row] = std::sin(0.17 * row) + 0.3 * std::cos(0.071 * row);
  }
  const auto rhs = multiply(matrix, exact);
  GpuCsrSolver solver;
  expect(solver.Initialize(matrix) == Status::kOk, "large synthetic CSR initialize");
  std::vector<double> single;
  SolveInfo single_info = solver.Solve(rhs, 1e-10, 2000, &single);
  expect(verified(matrix, rhs, single, single_info), "single large CSR true residual");
  double max_error = 0.0;
  for (size_t i = 0; i < single.size(); ++i)
    max_error = std::max(max_error, std::abs(single[i] - exact[i]));
  expect(max_error < 1e-8, "single large CSR manufactured solution");

  for (int width : { 1, 2, 4, 5, 32 }) {
    std::vector<Assembly> matrices(width, matrix);
    std::vector<std::vector<double>> rhs_values(width, rhs);
    for (int item = 0; item < width; ++item) {
      const double factor = item == 1 ? 0.0 : (item % 2 ? -0.75 : 1.0 + 0.1 * item);
      for (double& value : rhs_values[item]) value *= factor;
    }
    std::vector<SolveInfo> infos;
    std::vector<std::vector<double>> solutions;
    int launches = 0;
    expect(solver.SolveCurrentOperatorBatch(rhs_values, 1e-10, 2000,
        &infos, &solutions, &launches) == Status::kOk && launches == 1,
        "shared operator widths 1/2/4/5/32");
    bool valid = infos.size() == static_cast<size_t>(width) && solutions.size() == infos.size();
    for (int item = 0; valid && item < width; ++item)
      valid = verified(matrix, rhs_values[item], solutions[item], infos[item]);
    expect(valid, "shared operator mixed and zero RHS true residuals");

    // A different numeric operator per RHS catches accidental shared-K reuse.
    for (int item = 0; item < width; ++item) {
      const double factor = 1.0 + 0.2 * item;
      for (double& value : matrices[item].values) value *= factor;
      for (double& value : matrices[item].diagonal) value *= factor;
    }
    expect(solver.SolveBatch(matrices, rhs_values, 1e-10, 2000,
        &infos, &solutions, &launches) == Status::kOk, "replicated operators after shared mode");
    valid = infos.size() == static_cast<size_t>(width) && solutions.size() == infos.size();
    for (int item = 0; valid && item < width; ++item)
      valid = verified(matrices[item], rhs_values[item], solutions[item], infos[item]);
    expect(valid, "distinct operator mixed RHS true residuals");

    single_info = solver.Solve(rhs, 1e-10, 2000, &single);
    expect(verified(matrix, rhs, single, single_info), "single solve after replicated batch");
    std::vector<double> repeat;
    const auto repeat_info = solver.Solve(rhs, 1e-10, 2000, &repeat);
    expect(single == repeat && single_info.iterations == repeat_info.iterations,
        "large single solve repeat is deterministic");
  }

  std::vector<double> output;
  expect(solver.Solve(rhs, 1e-10, 0, &output).status == Status::kLinearSolveNotConverged,
      "large zero-iteration solve cannot pass");
  expect(solver.Solve(std::vector<double>(count, 0.0), 1e-10, 0, &output).status == Status::kOk,
      "large zero RHS can pass without an iteration");
  expect(solver.Solve(rhs, -1.0, 10, &output).status == Status::kInvalidArgument,
      "large invalid tolerance remains invalid");
  auto nonfinite_rhs = rhs;
  nonfinite_rhs[17] = std::numeric_limits<double>::quiet_NaN();
  expect(solver.Solve(nonfinite_rhs, 1e-10, 10, &output).status == Status::kNumericalNonfinite,
      "large nonfinite RHS is rejected");
  Assembly indefinite = matrix;
  indefinite.diagonal[19] = -1.0;
  expect(solver.UpdateValues(indefinite) == Status::kOk
          && solver.Solve(rhs, 1e-10, 10, &output).status == Status::kLinearSolveBreakdown,
      "large bad diagonal retains breakdown status");
  expect(solver.UpdateValues(matrix) == Status::kOk, "restore valid operator");
  const SolveInfo restored = solver.Solve(rhs, 1e-10, 2000, &output);
  expect(verified(matrix, rhs, output, restored),
      "large solve can recover after rejected inputs");

  std::vector<SolveInfo> mixed_infos;
  std::vector<std::vector<double>> mixed_solutions;
  const Status mixed = solver.SolveCurrentOperatorBatch(
      { rhs, nonfinite_rhs, std::vector<double>(count, 0.0), rhs }, 1e-10, 2000,
      &mixed_infos, &mixed_solutions, nullptr);
  expect(mixed == Status::kOk && mixed_infos.size() == 4
      && mixed_infos[1].status == Status::kNumericalNonfinite
      && verified(matrix, rhs, mixed_solutions[0], mixed_infos[0])
      && mixed_infos[2].status == Status::kOk
      && verified(matrix, rhs, mixed_solutions[3], mixed_infos[3]),
      "failed/zero/live RHS share a cooperative grid without losing PASS items");

  // Inverse-diagonal multiplication is unsafe for a subnormal reciprocal.
  // Exercise the preserved division template with a known diagonal solution.
  Assembly extreme;
  extreme.row_offsets.push_back(0);
  for (int row = 0; row < count; ++row) {
    extreme.free_nodes.push_back(row);
    extreme.row_offsets.push_back(row + 1);
    extreme.column_indices.push_back(row);
    extreme.values.push_back(1e308);
    extreme.diagonal.push_back(1e308);
  }
  const std::vector<double> extreme_rhs(count, 1.0);
  expect(solver.Initialize(extreme) == Status::kOk, "subnormal reciprocal initialize");
  const auto extreme_info = solver.Solve(extreme_rhs, 1e-10, 10, &output);
  expect(verified(extreme, extreme_rhs, output, extreme_info),
      "cooperative division fallback preserves extreme-scale true residual");
  return failures;
}

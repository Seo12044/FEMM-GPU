// No CAD or saved motor data: known SPD systems and an independent Ax-b check.
int BlockPreconditionerTests(bool repeat_benchmark = true)
{
  int failures = 0;
  auto expect = [&](bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL block preconditioner: " << message << '\n'; ++failures; }
  };
  auto make_matrix = [](int n) {
    Assembly a;
    a.row_offsets.push_back(0);
    for (int row = 0; row < n; ++row) {
      a.free_nodes.push_back(row);
      a.diagonal.push_back(2.01);
      if (row > 0) { a.column_indices.push_back(row - 1); a.values.push_back(-1.0); }
      a.column_indices.push_back(row); a.values.push_back(2.01);
      if (row + 1 < n) { a.column_indices.push_back(row + 1); a.values.push_back(-1.0); }
      a.row_offsets.push_back(static_cast<int>(a.values.size()));
    }
    return a;
  };
  auto multiply = [](const Assembly& a, const std::vector<double>& x) {
    std::vector<double> b(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i)
      for (int j = a.row_offsets[i]; j < a.row_offsets[i + 1]; ++j)
        b[i] += a.values[j] * x[a.column_indices[j]];
    return b;
  };
  auto verified = [&](const Assembly& a, const std::vector<double>& b,
                       const std::vector<double>& x, const SolveInfo& info) {
    if (info.status != Status::kOk || b.size() != x.size()) return false;
    const auto ax = multiply(a, x);
    double error = 0.0, norm = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
      if (!std::isfinite(x[i])) return false;
      error = std::hypot(error, ax[i] - b[i]); norm = std::hypot(norm, b[i]);
    }
    return error <= 1.01e-10 * norm;
  };
  for (int n : {1, 7, 8, 9, 31, 32, 33, 67, 4097}) {
    auto a = make_matrix(n);
    std::vector<double> exact(n);
    for (int i = 0; i < n; ++i) exact[i] = std::sin(0.3 * i) + 0.2;
    auto rhs = multiply(a, exact);
    GpuCsrSolver block(PcgPreconditioner::kBlockJacobi), jacobi(PcgPreconditioner::kJacobi);
    expect(block.Initialize(a) == Status::kOk && jacobi.Initialize(a) == Status::kOk, "initialize");
    std::vector<double> xb, xj;
    const auto ib = block.Solve(rhs, 1e-10, 2000, &xb);
    const auto ij = jacobi.Solve(rhs, 1e-10, 2000, &xj);
    expect(verified(a, rhs, xb, ib) && verified(a, rhs, xj, ij), "independent true residual");
    double max_error = 0;
    for (size_t i = 0; i < xb.size(); ++i) max_error = std::max(max_error, std::abs(xb[i] - exact[i]));
    expect(max_error < 1e-8, "manufactured solution");
    expect(block.preconditioner_stats().builds == 1 && block.preconditioner_stats().fallback_blocks == 0,
        "one SPD operator build, no fallback");
    if (n >= 31) expect(ib.iterations < ij.iterations, "stronger preconditioner reduces iterations");
    if (n == 4097) std::cout << "block_jacobi manufactured iterations: " << ij.iterations << " -> " << ib.iterations << '\n';
    const auto saved = xb;
    const auto repeat = block.Solve(rhs, 1e-10, 2000, &xb);
    expect(repeat.status == ib.status && repeat.iterations == ib.iterations && xb == saved,
        "deterministic repeat");
    expect(block.preconditioner_stats().builds == 1 && block.preconditioner_stats().reuses == 1,
        "unchanged matrix reused");
    if (n == 4097 && repeat_benchmark) {
      size_t free_before = 0, total = 0, free_after = 0;
      const size_t cache_bytes_before = block.preconditioner_bytes();
      cudaMemGetInfo(&free_before, &total);
      const auto start = ProfileClock::now();
      for (int repeat_index = 0; repeat_index < 100; ++repeat_index) {
        const auto info = block.Solve(rhs, 1e-10, 2000, &xb);
        expect(info.status == ib.status && info.iterations == ib.iterations && xb == saved,
            "100 repeats stay deterministic");
      }
      const double block_seconds = ProfileSecondsSince(start);
      cudaMemGetInfo(&free_after, &total);
      expect(block.preconditioner_bytes() == cache_bytes_before && block.preconditioner_stats().builds == 1,
          "repeated RHS does not allocate or rebuild");
      const auto jacobi_start = ProfileClock::now();
      for (int repeat_index = 0; repeat_index < 100; ++repeat_index)
        expect(jacobi.Solve(rhs, 1e-10, 2000, &xj).status == Status::kOk, "Jacobi benchmark solve");
      std::cout << "shared-operator 100 solves: jacobi=" << ProfileSecondsSince(jacobi_start)
                << " s block=" << block_seconds << " s builds=" << block.preconditioner_stats().builds
                << " reuses=" << block.preconditioner_stats().reuses << " memory_delta="
                << static_cast<long long>(free_before) - static_cast<long long>(free_after) << '\n';
    }
    expect(block.UpdateValues(a) == Status::kOk, "identical numeric update");
    for (double& b : rhs) b *= -0.7;
    const auto changed_rhs = block.Solve(rhs, 1e-10, 2000, &xb);
    expect(verified(a, rhs, xb, changed_rhs) && block.preconditioner_stats().builds == 1,
        "new current/RHS reuses exact matrix");
    if (n < 33) continue;
    // Off-diagonal-only change must invalidate even if the diagonal matches.
    a.values[1] *= 0.9;
    a.values[2] *= 0.9;
    expect(block.UpdateValues(a) == Status::kOk, "off-diagonal update");
    const auto changed_matrix = block.Solve(rhs, 1e-10, 2000, &xb);
    expect(verified(a, rhs, xb, changed_matrix) && block.preconditioner_stats().builds == 2,
        "changed coefficient rebuilds");
    // Different CSR ordering with the same dimension/nnz cannot reuse a plan.
    std::swap(a.column_indices[0], a.column_indices[1]);
    std::swap(a.values[0], a.values[1]);
    expect(block.Initialize(a) == Status::kOk, "new CSR structure");
    const auto structure = block.Solve(rhs, 1e-10, 2000, &xb);
    expect(verified(a, rhs, xb, structure) && block.preconditioner_stats().builds == 3,
        "initialize invalidates old structure");
    const auto zero_budget = block.Solve(rhs, 1e-10, 0, &xb);
    expect(zero_budget.status == Status::kLinearSolveNotConverged, "iteration limit not bypassed");
  }

  const auto a = make_matrix(67);
  GpuCsrSolver batch(PcgPreconditioner::kBlockJacobi);
  expect(batch.Initialize(a) == Status::kOk, "batch initialize");
  std::vector<Assembly> matrices(4, a);
  std::vector<std::vector<double>> rhs(4, std::vector<double>(67, 1.0));
  std::fill(rhs[1].begin(), rhs[1].end(), 0.0);
  std::fill(rhs[2].begin(), rhs[2].end(), -1.0);
  std::vector<SolveInfo> infos;
  std::vector<std::vector<double>> solutions;
  auto check = [&]() {
    expect(infos.size() == rhs.size() && solutions.size() == rhs.size(), "batch dimensions");
    for (size_t i = 0; i < infos.size(); ++i)
      expect(verified(matrices[i], rhs[i], solutions[i], infos[i]), "batch true residual");
  };
  expect(batch.SolveBatch(matrices, rhs, 1e-10, 2000, &infos, &solutions, nullptr) == Status::kOk,
      "duplicate operator batch");
  check();
  expect(batch.preconditioner_stats().builds == 1 && batch.preconditioner_stats().reuses == 3,
      "four equal matrices share one build");
  expect(batch.SolveBatch(matrices, rhs, 1e-10, 2000, &infos, &solutions, nullptr) == Status::kOk,
      "repeat batch");
  check();
  expect(batch.preconditioner_stats().builds == 1 && batch.preconditioner_stats().reuses == 7,
      "cross-call reuse");
  for (double& v : matrices[1].values) v *= 1.2;
  for (double& v : matrices[1].diagonal) v *= 1.2;
  matrices[3] = matrices[1];
  expect(batch.SolveBatch(matrices, rhs, 1e-10, 2000, &infos, &solutions, nullptr) == Status::kOk,
      "mixed operator batch");
  check();
  expect(batch.preconditioner_stats().builds == 2, "only new unique matrix rebuilt");
  // Old slot 1 was not an owner on the first call. After reordering, neither
  // changed slot may accidentally use factors from an overwritten owner.
  std::swap(matrices[0], matrices[1]);
  expect(batch.SolveBatch(matrices, rhs, 1e-10, 2000, &infos, &solutions, nullptr) == Status::kOk,
      "reordered operator batch");
  check();
  expect(batch.preconditioner_stats().builds == 4, "changed owners rebuilt safely");
  expect(batch.PrepareDeviceBatch(4) == Status::kOk, "device capacity");
  for (size_t i = 0; i < 4; ++i) {
    std::vector<double> inverse;
    BuildJacobiInverseDiagonal(matrices[i].diagonal, &inverse);
    expect(CopyToDevice(batch.device_batch_values() + i * a.values.size(), matrices[i].values.data(),
        a.values.size() * sizeof(double)) == Status::kOk, "device values");
    expect(CopyToDevice(batch.device_batch_diagonal() + i * 67, matrices[i].diagonal.data(), 67 * sizeof(double)) == Status::kOk,
        "device diagonal");
    expect(CopyToDevice(batch.device_batch_inverse_diagonal() + i * 67, inverse.data(), 67 * sizeof(double)) == Status::kOk,
        "device inverse diagonal");
    expect(CopyToDevice(batch.device_batch_rhs() + i * 67, rhs[i].data(), 67 * sizeof(double)) == Status::kOk, "device RHS");
  }
  expect(batch.SolvePreparedDeviceBatch(4, 1e-10, 2000, &infos, &solutions, nullptr) == Status::kOk,
      "device-assembled path");
  check();
  expect(batch.preconditioner_stats().builds == 4, "device path reuses same exact matrices");

  auto indefinite = make_matrix(2);
  indefinite.diagonal = {1.0, 1.0};
  indefinite.values = {1.0, 2.0, 2.0, 1.0};
  GpuCsrSolver invalid(PcgPreconditioner::kBlockJacobi);
  expect(invalid.Initialize(indefinite) == Status::kOk, "indefinite fixture initialize");
  std::vector<double> x;
  expect(invalid.Solve({1.0, -1.0}, 1e-10, 100, &x).status == Status::kLinearSolveBreakdown,
      "non-SPD matrix not falsely accepted");
  expect(invalid.preconditioner_stats().fallback_blocks == 1, "failed factorization is counted");
  return failures;
}

int PcgSpmvLayoutTests()
{
  int failures = 0;
  auto expect = [&](bool ok, const char* message) {
    if (!ok) { std::cerr << "FAIL PCG storage: " << message << '\n'; ++failures; }
  };
  for (int n : {511, 512, 513, 4097}) {
    Assembly matrix;
    matrix.row_offsets.push_back(0);
    std::vector<double> rhs(n);
    for (int row = 0; row < n; ++row) {
      matrix.free_nodes.push_back(row);
      const double diagonal = 4.25 + (row % 7) * 0.01;
      matrix.diagonal.push_back(diagonal);
      auto entry = [&](int column, double value) {
        matrix.column_indices.push_back(column); matrix.values.push_back(value);
      };
      if (row >= 17) entry(row - 17, -0.75);
      if (row > 0) entry(row - 1, -1.0);
      entry(row, diagonal);
      if (row + 1 < n) entry(row + 1, -1.0);
      if (row + 17 < n) entry(row + 17, -0.75);
      matrix.row_offsets.push_back(static_cast<int32_t>(matrix.values.size()));
      rhs[row] = std::sin(0.37 * row) + 0.2;
    }
    GpuCsrSolver packed(PcgPreconditioner::kJacobi, true);
    GpuCsrSolver csr(PcgPreconditioner::kJacobi, false);
    expect(packed.Initialize(matrix) == Status::kOk && csr.Initialize(matrix) == Status::kOk,
        "CSR and packed initialize");
    expect(packed.interleaved_spmv() == (n >= 512) && !csr.interleaved_spmv(),
        "small-matrix threshold and explicit CSR path");
    auto identical = [&](const SolveInfo& a, const SolveInfo& b,
                         const std::vector<double>& x, const std::vector<double>& y) {
      return a.status == Status::kOk && a.status == b.status
          && a.iterations == b.iterations && a.residual_l2 == b.residual_l2 && x == y;
    };
    for (int width : {1, 4, 5, 32}) {
      std::vector<std::vector<double>> loads(width, rhs), xp, xc;
      for (int item = 0; item < width; ++item)
        for (double& value : loads[item]) value *= item == 1 ? 0.0 : 1.0 + item * 0.1;
      std::vector<SolveInfo> ip, ic;
      SharedOperatorBatchEvidence evidence;
      expect(packed.SolveCurrentOperatorBatch(loads, 1e-10, 1000, &ip, &xp, nullptr, &evidence) == Status::kOk
          && csr.SolveCurrentOperatorBatch(loads, 1e-10, 1000, &ic, &xc, nullptr) == Status::kOk,
          "shared operator solve");
      for (int item = 0; item < width; ++item)
        expect(identical(ip[item], ic[item], xp[item], xc[item]), "shared operator bitwise parity");
      expect(evidence.physical_operator_count == 1, "shared numeric operator retained");

      std::vector<Assembly> matrices(width, matrix);
      for (int item = 0; item < width; ++item) {
        for (double& value : matrices[item].values) value *= 1.0 + item * 0.2;
        for (double& value : matrices[item].diagonal) value *= 1.0 + item * 0.2;
      }
      expect(packed.SolveBatch(matrices, loads, 1e-10, 1000, &ip, &xp, nullptr) == Status::kOk
          && csr.SolveBatch(matrices, loads, 1e-10, 1000, &ic, &xc, nullptr) == Status::kOk,
          "distinct numeric operators");
      for (int item = 0; item < width; ++item)
        expect(identical(ip[item], ic[item], xp[item], xc[item]), "distinct operator bitwise parity");

      // Simulate direct device assembly replacing every coefficient. A pointer
      // identity cache would silently reuse the previous numeric matrix here.
      expect(packed.PrepareDeviceBatch(width) == Status::kOk, "device batch prepare");
      for (int item = 0; item < width; ++item) {
        for (double& value : matrices[item].values) value *= 1.3;
        for (double& value : matrices[item].diagonal) value *= 1.3;
        std::vector<double> inverse;
        expect(BuildJacobiInverseDiagonal(matrices[item].diagonal, &inverse), "inverse diagonal");
        const size_t offset = static_cast<size_t>(item);
        expect(CopyToDevice(packed.device_batch_values() + offset * matrix.values.size(),
                   matrices[item].values.data(), matrix.values.size() * sizeof(double)) == Status::kOk
            && CopyToDevice(packed.device_batch_diagonal() + offset * n,
                   matrices[item].diagonal.data(), n * sizeof(double)) == Status::kOk
            && CopyToDevice(packed.device_batch_inverse_diagonal() + offset * n,
                   inverse.data(), n * sizeof(double)) == Status::kOk
            && CopyToDevice(packed.device_batch_rhs() + offset * n,
                   loads[item].data(), n * sizeof(double)) == Status::kOk, "direct assembly buffers");
      }
      expect(packed.SolvePreparedDeviceBatch(width, 1e-10, 1000, &ip, &xp, nullptr) == Status::kOk
          && csr.SolveBatch(matrices, loads, 1e-10, 1000, &ic, &xc, nullptr) == Status::kOk,
          "solve after device coefficients changed");
      for (int item = 0; item < width; ++item)
        expect(identical(ip[item], ic[item], xp[item], xc[item]), "device update bitwise parity");
    }
    // A graph change discards all prior layout and numeric buffers.
    Assembly diagonal = matrix;
    diagonal.values = diagonal.diagonal;
    diagonal.column_indices = diagonal.free_nodes;
    for (int row = 0; row <= n; ++row) diagonal.row_offsets[row] = row;
    expect(packed.Initialize(diagonal) == Status::kOk && csr.Initialize(diagonal) == Status::kOk,
        "changed CSR graph reinitializes");
    std::vector<double> xp, xc;
    const auto ip = packed.Solve(rhs, 1e-10, 1000, &xp);
    const auto ic = csr.Solve(rhs, 1e-10, 1000, &xc);
    expect(identical(ip, ic, xp, xc), "reinitialized graph parity");
    // The parallel validator must report the first bad row, just as the
    // original serial scan did, including on the single-block path.
    diagonal.diagonal[19] = -1.0;
    expect(packed.UpdateValues(diagonal) == Status::kOk, "invalid diagonal upload");
    auto bad_rhs = rhs;
    bad_rhs[17] = std::numeric_limits<double>::quiet_NaN();
    expect(packed.Solve(bad_rhs, 1e-10, 1000, &xp).status == Status::kNumericalNonfinite,
        "first invalid row is nonfinite");
    bad_rhs[17] = rhs[17];
    bad_rhs[n - 1] = std::numeric_limits<double>::quiet_NaN();
    expect(packed.Solve(bad_rhs, 1e-10, 1000, &xp).status == Status::kLinearSolveBreakdown,
        "first invalid row is a nonpositive diagonal");
  }
  // One long row would multiply storage by almost 32. Keep only that slice
  // in CSR; regular slices must still use the packed layout without padding.
  std::vector<int32_t> rows(513, 0), columns;
  for (int row = 0; row < 512; ++row) {
    for (int entry = 0; entry < (row == 0 ? 512 : 1); ++entry) columns.push_back(entry);
    rows[row + 1] = static_cast<int32_t>(columns.size());
  }
  PcgSpmvLayout irregular;
  expect(irregular.Initialize(rows, columns, true) == Status::kOk && irregular.enabled()
      && irregular.slots() == columns.size(), "irregular slice uses bounded CSR fallback");
  Assembly arrow;
  arrow.row_offsets.push_back(0);
  for (int row = 0; row < 513; ++row) {
    arrow.free_nodes.push_back(row);
    arrow.diagonal.push_back(row == 0 ? 514.0 : 4.0);
    if (row > 0) { arrow.column_indices.push_back(0); arrow.values.push_back(-0.1); }
    arrow.column_indices.push_back(row); arrow.values.push_back(arrow.diagonal.back());
    if (row == 0) for (int col = 1; col < 513; ++col) {
      arrow.column_indices.push_back(col); arrow.values.push_back(-0.1);
    }
    arrow.row_offsets.push_back(static_cast<int32_t>(arrow.values.size()));
  }
  GpuCsrSolver hybrid(PcgPreconditioner::kJacobi, true), plain(PcgPreconditioner::kJacobi, false);
  expect(hybrid.Initialize(arrow) == Status::kOk && plain.Initialize(arrow) == Status::kOk,
      "mixed CSR/packed slices initialize");
  std::vector<double> xp, xc, rhs(513, 1.0);
  const auto ip = hybrid.Solve(rhs, 1e-10, 1000, &xp);
  const auto ic = plain.Solve(rhs, 1e-10, 1000, &xc);
  expect(ip.status == Status::kOk && ic.status == Status::kOk && xp == xc
      && ip.iterations == ic.iterations && ip.residual_l2 == ic.residual_l2,
      "mixed CSR/packed slices preserve every row exactly");
  std::cout << "PCG CSR/packed parity and invalidation: " << (failures ? "FAIL" : "PASS") << '\n';
  return failures;
}

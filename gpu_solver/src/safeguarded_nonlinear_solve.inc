// Included inside NonlinearP1FixtureSolver. Only exhausted nonlinear items
// enter this bounded recovery; successful batch states are never replayed.
// The host reference assembler supplies J and F, CUDA PCG solves each Newton
// system. Armijo backtracking measures the *equation* residual, not a damped
// step (which can become small while the equations remain unsatisfied).
NonlinearSolveResult SolveSafeguarded(const std::vector<double>& currents,
    const NonlinearOptions& options)
{
  NonlinearSolveResult result;
  result.info.status = Status::kInvalidArgument;
  result.circuit_currents_a = currents;
  if (!initialized_ || currents.size() != static_cast<size_t>(model_.circuit_count)
      || !std::all_of(currents.begin(), currents.end(), [](double x) { return std::isfinite(x); })
      || !(options.relative_tolerance > 0) || !std::isfinite(options.relative_tolerance)
      || options.max_newton_iterations < 0)
    return result;
  std::vector<double> state(model_.nodes.size(), 0.0);
  for (size_t i = 0; i < model_.dirichlet_nodes.size(); ++i)
    state[model_.dirichlet_nodes[i]] = model_.dirichlet_a_wb_per_m[i];

  auto equation_norm = [](const Assembly& assembly, const std::vector<double>& a) {
    double norm = 0.0;
    for (size_t row = 0; row < assembly.free_nodes.size(); ++row) {
      double value = -assembly.rhs_per_amp[row] - assembly.rhs_offset[row];
      for (int slot = assembly.row_offsets[row]; slot < assembly.row_offsets[row + 1]; ++slot)
        value += assembly.values[slot]
            * a[assembly.free_nodes[assembly.column_indices[slot]]];
      norm = std::hypot(norm, value);
    }
    return norm;
  };
  Assembly assembly;
  Status status = AssembleNonlinearNewton(model_, state, currents, true, &assembly);
  if (status != Status::kOk) { result.info.status = status; return result; }
  // Expand prescribed signed roots before the first Newton direction.
  std::vector<double> free_state(assembly.free_nodes.size(), 0.0);
  for (size_t i = 0; i < free_state.size(); ++i)
    free_state[i] = state[assembly.free_nodes[i]];
  state = ExpandAssemblySolution(assembly, free_state);
  status = AssembleNonlinearNewton(model_, state, currents, true, &assembly);
  if (status != Status::kOk) { result.info.status = status; return result; }
  double norm = equation_norm(assembly, state);
  const double scale = std::max(norm, std::numeric_limits<double>::min());
  const double equation_tolerance = options.relative_tolerance;
  const double step_tolerance = 100.0 * options.relative_tolerance;
  GpuCsrSolver recovery_solver;
  status = recovery_solver.Initialize(assembly);
  if (status != Status::kOk) { result.info.status = status; return result; }
  result.info.status = Status::kNonlinearSolveNotConverged;
  for (int iteration = 0; iteration < options.max_newton_iterations; ++iteration) {
    if (!std::isfinite(norm)) { result.info.status = Status::kNumericalNonfinite; break; }
    std::vector<double> rhs(assembly.free_nodes.size());
    for (size_t row = 0; row < rhs.size(); ++row)
      rhs[row] = assembly.rhs_per_amp[row] + assembly.rhs_offset[row];
    status = recovery_solver.UpdateValues(assembly);
    if (status != Status::kOk) { result.info.status = status; break; }
    std::vector<double> free_solution;
    const SolveInfo linear = recovery_solver.Solve(rhs, options.linear_relative_tolerance,
        options.max_linear_iterations, &free_solution);
    if (linear.status != Status::kOk) { result.info = linear; break; }
    const std::vector<double> candidate = ExpandAssemblySolution(assembly, free_solution);
    double step_norm = 0.0, candidate_norm = 0.0;
    for (size_t node = 0; node < state.size(); ++node) {
      step_norm = std::hypot(step_norm, candidate[node] - state[node]);
      candidate_norm = std::hypot(candidate_norm, candidate[node]);
    }
    const double relative_step = step_norm
        / std::max(candidate_norm, std::numeric_limits<double>::min());
    result.info.iterations = iteration + 1;
    result.info.residual_l2 = relative_step;
    result.residual_history.push_back(relative_step);
    if (!std::isfinite(relative_step)) { result.info.status = Status::kNumericalNonfinite; break; }
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
      if (status != Status::kOk) { result.info.status = status; break; }
      const double trial_norm = equation_norm(trial_assembly, trial);
      if (std::isfinite(trial_norm) && (trial_norm <= (1.0 - kArmijoDecrease * alpha) * norm
              || trial_norm / scale <= equation_tolerance)) {
        state = std::move(trial);
        assembly = std::move(trial_assembly);
        norm = trial_norm;
        accepted = true;
        break;
      }
    }
    if (!accepted) break;
  }
  result.a_wb_per_m = std::move(state);
  result.verified_equation_residual_relative = norm / scale;
  return FinalizeResult(&result);
}

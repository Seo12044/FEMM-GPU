# Solver source layout

`src/linear_p1_poc.cu` is the executable entry point. It delegates command
dispatch to `src/app/cli.cpp`; it contains no numerical implementation.
Command names, protocols, exit codes, solver defaults and executable names
are unchanged by this split.

## Where to change code

| Directory | Responsibility |
| --- | --- |
| `src/core/` | Status and model primitives, CUDA buffer ownership, profiling clock |
| `src/physics/` | Linear and nonlinear element assembly, B-H material evaluation, signed-system reduction, device assembly plan |
| `src/solvers/` | CSR ownership, PCG kernels, Jacobi/block-Jacobi, optional cuDSS plan, nonlinear iteration and recovery |
| `src/postprocess/` | Weighted-stress mask, field probes and smoothing, force/torque, native air-gap and tangent responses |
| `src/io/` | Strict JSON and hashing, immutable mesh validation, sample requests, pose/sector mapping, JSON escaping |
| `src/app/` | Command dispatch, single-sample execution, batch preparation/grouping, response serialization and batch execution |
| `tests/native/` | Manufactured models, FEMM reference readers, numerical and protocol regression cases |

Each implementation is compiled separately. Headers declare interfaces;
implementation `.cpp`/`.cu` files are not included as source fragments.
The generated native AGE matrix remains one deliberate exception: CMake
extracts its coefficients from the tracked FEMM `fkn/prob1big.cpp` into the
build directory, and `native_airgap_matrix.cpp` includes those expressions.

The standalone B-H fixture executable and FEMM mesh-only helper remain
separate targets. They are not part of `femm_gpu_core`.

## Targets and ownership

- `femm_gpu_core` owns numerical execution, input validation, postprocessing
  and sample/batch adapters. It has no dependency on test fixtures.
- `femm_gpu_test_support` links the core and supplies existing self-test and
  frozen-reference commands. These commands remain available even when
  `BUILD_TESTING=OFF`; that switch controls CTest registration.
- `gpu_linear_p1_poc` links both and provides command dispatch. The
  `femm_gpu` executable alias remains byte-identical.

Solver buffers belong to solver instances. A cuDSS plan borrows its owning
CSR solver's buffers; it is destroyed before those buffers and invalidated
before coefficients change. The backend selection and diagnostic counters
retain one definition in `solvers/options.cpp`. The cuDSS compile definition
is propagated consistently because it affects the CSR class layout.

The solver types are `LinearP1Solver` and `NonlinearP1Solver`; manufactured
models live in `tests/native`, not in their implementations. The nonlinear
type previously had a `FixtureSolver` name despite handling production batches.

## Verification

Run the ordinary CTest suite in both cuDSS-enabled and disabled builds.
`gpu_femm_source_layout` also guards the small entry point, independently
compiled sources, header guards and production/test boundary without CUDA.

For a structural change, compare old and new executables on the same saved
batch, keeping the linear backend the same:

```powershell
python gpu_solver/tests/benchmark_saved_batch.py `
  --baseline old/femm_gpu.exe --candidate new/femm_gpu.exe `
  --baseline-linear-solver pcg --candidate-linear-solver pcg `
  --request saved_batch.json --output comparison-pcg `
  --rounds 1 --strict-parity
```

Repeat with `cudss` when enabled. `--strict-parity` requires exact numeric
results, convergence records and cache counters; only timing is excluded.
The ordinary benchmark gate remains unchanged for intentional solver
comparisons. Save response files and input/executable hashes with the report.

Build into a new directory while an existing analysis is running. Do not
overwrite its executable or rebind its checkpoints as part of a refactor.

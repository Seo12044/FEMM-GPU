# GPU solver performance

## Functional module split, 2026-10-27

The former 11,797-line `src/linear_p1_poc.cu` is now a six-line entry point.
The core, physics, solvers, IO, postprocessors and application adapters are
independently compiled modules; native fixtures and regression commands are
in `tests/native`. See [ARCHITECTURE.md](ARCHITECTURE.md) for ownership.
PCG kernels are separate from CSR management. The largest numerical source
is 825 lines after applying the checked-in formatting rules.

This is a structural change, not a new convergence policy. FP64 equations,
element/reduction order, kernel configurations, tolerances, backend defaults,
CLI protocols and checkpoint-visible responses were retained. Removed code
was limited to nine obsolete textual-include/header fragments, the unread
`mapped_regions` accumulator, and two uncalled sector-compatibility helpers.
File-local helpers no longer have unnecessary external declarations.

Final Release validation with CUDA 13.3/MSVC 19.44:

| Check | Result |
| --- | --- |
| cuDSS-enabled CTest suite | 17/17 PASS |
| cuDSS-disabled CTest suite | 15/15 PASS; no cuDSS DLL dependency |
| PCG old/new saved batches | 6 requests, 81 item pairs, exact equality |
| cuDSS old/new saved batches | 6 requests, 81 item pairs, exact equality |
| Complete nodal A and element B output | 2 operating points x 2 backends, 4 exact comparisons |
| CUDA memcheck, PCG and cuDSS | 0 errors, 0 leaked bytes |
| CUDA synccheck, PCG self-test | 0 errors |

The saved batches cover no-load poses, suspension-current perturbations,
native cyclic sectors, eccentric dynamics and 32-item no-load/load batches.
Each backend was compared with the same backend before the split, not with
the other linear solver. `--strict-parity` ignored timing only: force,
torque, sampled B, linkage, residuals, iteration counts and cache counters
all matched exactly. Maximum numeric difference was zero. Full-field checks
also matched every returned nodal and element value. The capability JSON
and executable alias were unchanged. These are result-preservation checks,
not a speedup measurement.

Local evidence under `../../refactor-20261027/`:

- `ctest-enabled-final.log`, `ctest-disabled-final.log`.
- `pcg-final/report.json`, `cudss-final/report.json` and retained responses.
- `full-fields/report.json`, derived requests and full-field responses.
- `memcheck-pcg-final.log`, `memcheck-cudss-final.log`, `synccheck-pcg-final.log`.

The original production executable was not overwritten:
`91d0413e80929bfd8d20227b243922e1cb67206b77aa265a7bdf378ebfb4947b`.
The tested refactor binary is `build-refactor-20261027/Release`, SHA-256
`c169322bb01f668a1ff72becc7eb51b8e13f816677c5be882ef15d90b40d6826`.

## Optional cuDSS direct backend, 2026-10-27

The default remains Jacobi PCG. `GPU_SOLVER_ENABLE_CUDSS=ON` adds an explicit
`--linear-solver cudss` option; it does not change any MATLAB run, result
signature, request, mesh, material table, or original executable. The direct
backend is experimental: convergence/reference tests pass, but some saved
motor force components exceed the existing binary-comparison gate. Do not
describe that benchmark as an accuracy-gate PASS or silently replace a
production PCG executable.

Implementation: `src/solvers/cudss_direct_solver.cu`. cuDSS solves the FP64 SPD inner
system with sparse Cholesky, retaining the existing nonlinear solve and all
postprocessors. Deterministic mode is enabled; factor downcasting, hybrid
CPU execution, and pivot perturbation are disabled. cuDSS 1.0 does not allow
its iterative refinement in deterministic mode, so library refinement is
disabled. Each returned solution still has to pass the caller's original
full-CSR `||Ax-b||/||b||` tolerance. An inaccurate direct result fails; it
does not receive a looser tolerance or a CPU fallback.

The full CSR graph must have strictly sorted columns and reciprocal entries.
Numeric skew is checked on the local diagonal/operator scale, not relative
to a nearly cancelled off-diagonal. The first strict entry-relative check
rejected an actual motor matrix with maximum skew `2.3283064e-10` against a
largest diagonal `5.1421439e6`. The final guard accepts roundoff-scale skew
only and still checks the original, unmodified matrix in the true residual.
Significant asymmetry, nonfinite values, non-positive diagonals, and failed
Cholesky minors are rejected. The original matrix is never symmetrized or
shifted in application memory.

Symbolic analysis is reused for one initialized graph. Numerical factors
are retained only for a private, unchanged operator. A coefficient update
invalidates them, including off-diagonal-only changes and writes to the same
device address. Distinct batch operators refactor sequentially using one
factor workspace per solver; they cannot inherit a previous slot's factors.
The real nonlinear benchmarks therefore have many factorizations but few
analyses, not one numerical factorization for an entire nonlinear campaign.

Tests cover manufactured solutions and an independent residual at
1/7/33/4097 rows, shared and distinct operators, zero and failed RHS slots,
host/device coefficient updates, reinitialization, singular and malformed
systems, matrix scales `1e-140` and `1e140`, invalid tolerances and budgets,
and 100 deterministic refactorizations with no warmed GPU-memory growth.
Four concurrent independent plans also verify solution isolation and exact
phase accounting. Geometry-mask prefetch threads publish their diagnostics
under a mutex; each plan retains its own factors and buffers.
Neutral periodic, anti-periodic, and axisymmetric fixtures compare full
fields across host/device assembly and PCG/direct solves. Default and
explicit PCG results are identical on those fixtures. The frozen nonlinear
FEMM field/flux reference passes with cuDSS.
The frozen linear FEMM reference and nonlinear force/torque/air-gap
postprocessing reference also pass with direct solving selected explicitly.
With `CUDA_VISIBLE_DEVICES=-1`, direct solving returns
`GPU_FEMM_GPU_UNAVAILABLE` rather than switching to a CPU solve.

Final CTest: **16/16 passed** with cuDSS and **14/14 passed** with it disabled.
The disabled build does not link/load cuDSS and rejects its option.
The Python benchmark parity/option tests pass 14/14. Compute Sanitizer
memcheck reports **0 errors, 0 leaked bytes** and synccheck **0 errors** on
the direct regression, including singular-factor recovery and 100 repeated
factorizations. NVIDIA's [configuration restrictions](https://docs.nvidia.com/cuda/cudss/latest/types.html)
and [runtime requirements](https://docs.nvidia.com/cuda/cudss/latest/getting_started.html)
are reflected in the build and documented in [README.md](README.md).

Two saved full-field requests from the current optimization, Cases 0334 and
0466 (`id=0, iq=1 A-command`, 0 mechanical degrees), were also solved without
preparing new meshes. Both PCG/direct runs passed with 21 nonlinear iterations.
Maximum PCG/direct differences:

| Case | Nodal A [Wb/m] | Element B component [T] | Torque [N m] | Force component [N] |
| --- | ---: | ---: | ---: | ---: |
| 0334 | 4.289e-12 | 5.657e-8 | 2.159e-10 | 1.012e-8 |
| 0466 | 1.539e-11 | 3.855e-8 | 1.803e-10 | 3.118e-9 |

These are measured differences, not an assertion that the old PCG field is
an exact solution or that every full-field element passes the benchmark's
flat `1e-9` absolute gate. Original saturation fields and plots were not
overwritten. The new enabled binary in explicit PCG mode also reproduces
the saved 3-item no-load and 4-item suspension PCG responses with **zero**
observable difference, identical convergence records, and identical PCG
iteration totals.

### Final saved-motor measurements

RTX 4060 Ti 8 GB, driver 610.62, CUDA 13.3, cuDSS 1.0.0.35, MSVC 19.44,
Release (`sm_86` plus PTX). Optimization and sanitizer processes were stopped.
Each row is the median complete-process time over three alternating A/B
rounds, including parsing, symbolic analysis, numerical factorization,
nonlinear iteration, and postprocessing. Baseline is optimized Jacobi PCG.

| Saved request | Items | PCG [s] | cuDSS [s] | Speedup | Time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Case 118, no-load poses | 3 | 7.315 | 3.013 | 2.428x | 58.81% |
| Case 118, suspension-current perturbations | 4 | 4.650 | 2.258 | 2.059x | 51.44% |
| Case 882, native cyclic sector | 6 | 2.230 | 1.407 | 1.585x | 36.91% |
| Case 117, eccentric dynamics | 4 | 12.224 | 5.972 | 2.047x | 51.14% |
| Case 118, complete saved no-load batch | 32 | 73.113 | 22.844 | 3.201x | 68.76% |
| Case 118, complete saved suspension batch | 32 | 32.993 | 10.917 | 3.022x | 66.91% |

All **486 item executions** converged below the unchanged `1e-6` gate;
maximum reported nonlinear residual was `9.984785278794787e-7`. Small requests
overlap the larger batches, so these are not 81 independent operating states.
Nonlinear iteration counts matched PCG. The three direct responses per
request, including all returned convergence records, were exactly repeatable.
The largest cuDSS plan device-memory estimate was `66,043,580` bytes
(62.98 MiB), excluding application buffers and other live plans.

Maximum differences across the batch comparisons:

- Torque: `1.373e-10 N m`.
- Fx/Fy component: `2.817e-8 N`.
- Sampled radial air-gap B: `4.189e-10 T`.
- Circuit flux linkage: `7.926e-12 Wb`.

The unchanged comparison uses `rtol=1e-8`, `atol=1e-9`. Force-component
differences exceeded it in five of six request sets (15 candidate process
runs). The sector set passed. The final comparison report is therefore
**FAILED**, despite every physical solve converging. The gate was not
relaxed and the failed comparisons were not discarded. These measurements
establish a saved-input speed benefit and quantify the numerical differences;
they do not establish strict binary parity, a full-optimization speedup, or
mesh-discretization accuracy. PCG remains the default.

Local evidence under `../../cudss-20261027/`:

- `final-threadsafe-ab/report.json`: final timings, raw response paths,
  per-run differences, and input/executable hashes. Earlier `smoke-*` and
  `final-ab` directories are intermediate evidence, not this release's timings.
- `motor-fields/`: the two current-design full-field comparisons.
- `default-pcg-parity/report.json`: unchanged PCG outputs, **PASS**.
- `ctest-delivery.log`, `ctest-off-threadsafe.log`,
  `memcheck-direct-threadsafe.log`, `synccheck-direct-threadsafe.log`.

Executable SHA-256 values:

- Baseline: `a0d2a58945a752ac28555342b233e64322bbda9af8547ec5f3d7c88488071f39`.
- Candidate: `91d0413e80929bfd8d20227b243922e1cb67206b77aa265a7bdf378ebfb4947b`.

Candidate: `build-cudss-20261027/Release`; `femm_gpu.exe` and
`gpu_linear_p1_poc.exe` are byte-identical. The mode must be selected explicitly.

The production optimization was already `CANCELLED` when inspected and was
left stopped for these benchmarks. The user subsequently accepted the measured
differences and requested an explicit cuDSS continuation. MATLAB integration
now forwards the option and separates its checkpoint identity; a validated
observation-migration receipt retains 1,033 completed PCG cases without
rebasing their solver checkpoints. Integration/reuse regressions passed
140/140, and Python frontend tests passed 15/15. Production continuation is
recorded in the MATLAB repository's `docs/agent-handoff.md`. This acceptance
does not change the failed strict comparison report or the native PCG default.

## Jacobi execution and sparse-storage optimization, 2026-10-21

This change keeps Jacobi, FP64, CSR row order, scalar reduction order, the
adaptive inner tolerance, and the final true `Ax-b` acceptance check. It
targets the execution cost of each iteration, not a different equation or
a weaker convergence criterion. The comparison baseline is the preceding
`build-pcg-block-20261021` executable in its default Jacobi mode.

Changes:

- Fused row-owned matrix products/dot products and residual/preconditioner
  updates remove redundant vector passes. Cooperative Jacobi sums exchange
  two sets of partials together. Each sum retains the old block tree and
  shard order. A true-residual restart recomputes its preconditioner and rho;
  it cannot reuse the preceding recursive-residual values.
- Initial RHS/diagonal validation is parallel. An integer-valued priority
  reduction preserves the original first-invalid-row error precedence.
- A row-ordered hybrid Sliced-ELL/CSR layout groups 32 neighboring rows for
  memory access. Each row visits only its actual entries, in the original
  order. There is no row sorting, FP32 arithmetic, or padded-zero summation.
  Slices needing more than twice their original entry count stay CSR locally;
  short regular rows can still use the interleaved format beside long AGE
  rows. Matrices below 512 rows retain the original CSR path.
- Topology is constructed once per initialization; numerical values are
  repacked after host or device assembly, including unchanged-pointer writes.
  Storage is bounded to the current batch. The VRAM chunk estimate includes
  the worst-case extra numeric storage, and shared-operator byte diagnostics
  include the packed values.

Implementation: `src/pcg_spmv_layout.cuh` and the PCG kernels in
`src/solvers/pcg_kernels.cu`. No new library dependency or motor-specific heuristic
is introduced. The layout follows the general memory-access principles in
NVIDIA's [Sliced ELL storage description](https://docs.nvidia.com/cuda/cusparse/storage-formats.html)
and [coalesced-access guidance](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/);
it does not call cuSPARSE or replace the deterministic reduction with a library reduction.

Tests in `src/pcg_spmv_layout_tests.inc` compare packed/CSR solutions, iteration
counts and residuals exactly at 511/512/513/4097 rows and batch widths
1/4/5/32. They cover shared operators, distinct operators, zero RHS,
device coefficient replacement, changed topology, mixed CSR/packed slices,
and invalid-row error precedence. The independent manufactured-solution,
true-residual, nonlinear/recovery, tangent and FEMM reference tests remain.

### Final saved-motor comparison

RTX 4060 Ti, CUDA 13.3, MSVC 19.44, Release (`sm_86` plus PTX). The optimization
was stopped, and no sanitizer ran during timing. Both binaries use default
Jacobi. These are median complete-process times from three alternating A/B
rounds, including input verification and postprocessing.

| Saved request | Items | Previous Jacobi [s] | Optimized Jacobi [s] | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| Case 118, no-load rotor poses | 3 | 8.317 | 7.409 | 10.92% |
| Case 118, suspension-current perturbations | 4 | 5.203 | 4.902 | 5.79% |
| Case 882, native cyclic-sector validation | 6 | 2.553 | 2.441 | 4.39% |
| Case 117, eccentric rotor dynamics | 4 | 14.194 | 12.403 | 12.62% |
| Case 118, full saved no-load angle batch | 32 | 78.992 | 68.791 | 12.91% |
| Case 118, full saved suspension-current batch | 32 | 35.983 | 32.062 | 10.90% |

All 36 process runs passed, with 486 item executions. Small requests overlap
the full batches; this is not 81 unique operating states. Maximum reported
differences were **zero** for torque, Fx/Fy, circuit flux linkage, and sampled
air-gap B. All per-item convergence records and ordinary nonlinear PCG counts
also matched exactly across all six repeats per request. The largest reported
nonlinear residual was `9.984785278794787e-7`, unchanged from the baseline and
below the existing `1e-6` gate. Neither solver nor comparison tolerances changed.

Final CTest: **13/13 passed**. Compute Sanitizer memcheck and synccheck pass on
the complete self-test (0 errors, 0 leaked bytes). Focused initcheck and both
storage/block-preconditioner racechecks report 0 errors/hazards.

Local evidence: `../../pcg-fused-20261021/final-ab/report.json`, its 36 raw
responses, `../../pcg-fused-20261021/ctest-final.log`, and the `*-final.log`
sanitizer records in that directory. Intermediate probe directories are not
the final measurements. Executable SHA-256 values:

- Baseline: `b0ccc5c74c463a85e4113ab29f88a3fd6cf9188f5eda553a47804c28b35b5d30`.
- Candidate: `a0d2a58945a752ac28555342b233e64322bbda9af8547ec5f3d7c88488071f39`.

The candidate is built separately in `build-pcg-fused-20261021/Release`;
`gpu_linear_p1_poc.exe` and `femm_gpu.exe` are identical. Original executables,
caller configurations and analysis checkpoints remain untouched. This is a
saved-input solver improvement, not a measured full-optimization speedup or a
change to mesh accuracy.

## Selectable block preconditioner and numeric reuse, 2026-10-21

The standalone solver now accepts `--preconditioner jacobi|block-jacobi`
before the command. Jacobi remains the default; no launcher, saved request,
analysis checkpoint, or original solver executable is changed.

`block-jacobi` uses FP64 Cholesky inverses of contiguous 8-node principal
blocks. It is not a port of FEMM 5's CPU incomplete-LDL or an algebraic
multigrid preconditioner. The kernel, cache and tests live in
`src/solvers/block_jacobi_preconditioner.cu` and
`tests/native/block_preconditioner_regression.cu`.
Both one-block and cooperative PCG paths support it. Jacobi keeps a separate
kernel specialization and does not allocate the block cache.

Reuse rules:

- Identical numeric operators in a batch share one factorization. Equality
  checks every coefficient and diagonal bit, not an approximate hash.
- The previous batch can reuse a matching owner slot. Changed coefficients,
  changed slot ownership, CSR reinitialization, or batch dimensions require
  rebuilding. Storage is bounded by the current batch, not campaign length.
- Repeated RHS solves on private, unmodified operator buffers bypass the
  comparison and construction. Every operator update invalidates this shortcut.
- A shared mesh does not imply a shared nonlinear Jacobian. Distinct currents,
  saturation states, rotor poses and eccentric offsets are not assumed equal.
- An unusable/asymmetric block falls back to diagonal preconditioning and is
  counted. Original diagonal validation, iteration limits and true `Ax-b`
  convergence tests still apply. The equation itself is never shifted.

Batch timing diagnostics expose the block size, construction/reuse/fallback
counts and setup time. These counts cover ordinary batched nonlinear solves;
separate recovery/tangent work is not included. Returned nonlinear PCG
iterations are diagnostic only, not a new acceptance criterion.

An initial 32-node dense implementation was slower on every saved workload.
A zero-entry-compressed inverse reduced its four-current cost but was still
slower than Jacobi, so that extra representation was removed. The retained
8-node version is simpler and has lower application cost. It still must not
be assumed faster than Jacobi on an arbitrary mesh or GPU.

Tests cover 1/7/8/9/31/32/33/67/4097 unknowns, 67x67 and 129x129 grids,
shared and distinct operators at batch widths 1/2/4/5/32, identical and
off-diagonal-only updates, owner reordering, CSR reinitialization, device
assembly, malformed/zero/negative RHS, factorization failure, extreme-scale
division fallback, and iteration limits. CLI tests cover planar periodic,
antiperiodic and axisymmetric host/device solves, explicit/default Jacobi
identity, invalid mode rejection and executable overwrite protection.
Both default and block-mode complete self-tests exercise nonlinear recovery
and tangent solves as well. CTest passes 13/13, including the benchmark's
13 Python unit tests.

CUDA/MSVC incremental builds now explicitly track local `.inc`/`.cuh` files.
The toolchain's dependency scan had omitted them, allowing a header-only edit
to leave an old kernel in the executable. The new dependencies are expressed
through CMake's [VS_SETTINGS](https://cmake.org/cmake/help/latest/prop_sf/VS_SETTINGS.html),
without editing generated project files.

### Saved-motor A/B results

On an RTX 4060 Ti, with the MATLAB optimization stopped, the same final
executable ran Jacobi and block-Jacobi in alternating order for three rounds.
These are median complete-process times, including loading and postprocessing.
The set contains 17 saved operating states, or 102 item executions across
both modes and all rounds.

| Saved request | Items | Jacobi [s] | Block-Jacobi [s] | Block time increase |
| --- | ---: | ---: | ---: | ---: |
| Case 118, no-load rotor poses | 3 | 8.451 | 9.040 | 6.97% |
| Case 118, suspension-current perturbations | 4 | 5.162 | 6.144 | 19.02% |
| Case 882, native cyclic-sector validation | 6 | 2.492 | 2.662 | 6.80% |
| Case 117, eccentric rotor dynamics | 4 | 13.244 | 14.061 | 6.17% |

For the no-load request, returned nonlinear PCG iterations fell from 214,787
to 203,396 (5.3%), but factor setup took 0.210 s and the solve kernels also
took longer. Suspension and sector requests each reused three equal numeric
operators; no-load and offset requests had no exact matches. Reusing a mesh
is therefore not enough to avoid nonlinear factorization in these inputs.

All items converged under the unchanged solver acceptance rules. However,
the unchanged observable comparison (`rtol=1e-8`, `atol=1e-9`) **failed** for
some force components in the no-load, suspension and offset requests. The
sector request passed. Maximum absolute differences across the four requests:

| Observable | Maximum absolute Jacobi/block difference |
| --- | ---: |
| Force component | 3.220e-9 N |
| Torque | 2.845e-10 N m |
| Air-gap radial flux density | 2.269e-10 T |
| Circuit flux linkage | 5.880e-13 Wb |

The benchmark's optional `--continue-on-parity-failure` collects numerical
differences without hiding them: individual comparisons remain `FAIL`, the
report remains `FAILED`, and the exit code is nonzero. No tolerance was
relaxed to accept this mode. **Block-Jacobi remains experimental and opt-in;
Jacobi remains the production default.**

The manufactured coupled-matrix fixture shows where the feature can help:
PCG iterations fell from 206 to 49. After initialization/warm-up, 100 solves
of the same operator took 0.361102 s with Jacobi versus 0.147991 s with blocks
(2.44x), with one construction and no GPU-memory growth. This is a synthetic
same-matrix measurement, not a motor-campaign speedup.

### Final validation and evidence

Final CTest: 13/13 passed. Compute Sanitizer reports zero errors for full
block-mode self-tests under memcheck and synccheck, and zero errors/hazards
for the focused preconditioner tests under initcheck and racecheck. Memcheck
also reports zero leaked bytes. Racecheck initially found shared reduction
scratch being reused before every lane read the result. Both deterministic
reduction helpers now have a final read barrier; the failed log is retained
and the corrected build passed the checks above.

A separate compatibility run compared default Jacobi in the previous inexact
Newton executable (`c3a4f10a...`) against default Jacobi in this build across
all 17 saved states. It passed with zero differences in every compared
observable. Timing, cache counters and convergence diagnostics are excluded
from observable comparison; this is not a byte-identical JSON claim.

Build: MSVC 19.44, CUDA 13.3, Release, CUDA architecture 86. Final executable:
`build-pcg-block-20261021/Release/gpu_linear_p1_poc.exe`, with SHA-256
`b0ccc5c74c463a85e4113ab29f88a3fd6cf9188f5eda553a47804c28b35b5d30`.
`femm_gpu.exe` is byte-identical. Local evidence under `pcg-block-20261021`:

- `final-ab/report.json`: final same-binary mode comparison, raw responses,
  input hashes, timings and retained numerical failures.
- `jacobi-compatibility/report.json`: previous/new default-mode comparison,
  all 17 states passing with zero observable differences.
- `ctest-fixed.log`: complete verbose regression test output.
- `memcheck-fixed.log`, `synccheck-fixed.log`, `initcheck-fixed.log`,
  `racecheck-fixed.log`: final sanitizer summaries.

## Inexact Newton, 2026-10-21

Baseline: `e38810b`, the `build-pcg-adaptive-20260909` executable. This change
reduces inner PCG work during nonlinear iteration; it does not port FEMM 5's
CPU incomplete-LDL preconditioner or change the GPU kernels.

For a previous relative Newton step `s`, the next PCG tolerance is
`max(original_tolerance, 1e-8 * min(1, s))`. The first solve, linear-material
models, and near-converged steps use the original tolerance. Batched states
use the tightest tolerance among active items. A nonlinear item cannot pass
until an original-precision correction satisfies the existing step test.
Production final linear tolerance remains `1e-10`, nonlinear step threshold
remains `1e-6`, and iteration limits are unchanged. Tangent solves and Armijo
recovery retain their original precision and residual checks.

The initially tested `1e-4` forcing factor failed two of three saved motor
poses and was rejected. FEMM 5's preconditioned residual criterion is not
interchangeable with this solver's unpreconditioned `||Ax-b|| / ||b||` check.
Do not loosen the factor based on the FEMM 5 number alone.
An intermediate `1e-7` factor passed the 17-item timing set but failed the
unchanged observable gate on two of 32 no-load angles (maximum force-component
difference `2.057e-9 N`, versus the `1e-9 N` absolute comparison tolerance).
The final factor is therefore `1e-8`; the comparison tolerance was not relaxed.

The nonlinear tolerance tests cover the policy bounds, exact final correction,
fixed-precision option, warm starts, host/device batches, zero and malformed
currents, recovery equation residual, and both iteration budgets. Their
structured nonlinear fixture uses 282 PCG iterations versus 319 at fixed
precision (11.6% fewer). Linear-model batching and existing CLI/reference
tests remain part of the CTest suite.

The implementation shares tolerance selection and convergence checks across
solve paths. Unreachable device-support branches and stale comments were
removed; the JSON request/response schemas are unchanged.

### Isolated saved-input measurements (final `1e-8` factor)

On an RTX 4060 Ti, with the MATLAB optimization stopped, baseline/candidate
order alternated over three rounds. Each row is the median elapsed time of
a complete solver process, including artifact loading and postprocessing.

| Saved request | Items | Baseline [s] | Candidate [s] | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| Case 118, no-load rotor poses | 3 | 8.851 | 8.099 | 8.50% |
| Case 118, suspension-current perturbations | 4 | 5.390 | 5.219 | 3.17% |
| Case 882, native cyclic-sector validation | 6 | 2.647 | 2.671 | -0.91% |
| Case 117, eccentric rotor dynamics | 4 | 14.297 | 14.076 | 1.55% |

All 102 item executions passed the unchanged convergence gates and the
observable comparison (`rtol=1e-8`, `atol=1e-9`). Newton iteration counts
were unchanged. Maximum absolute old/new differences across these requests:

- Torque: `1.845e-11 N m`.
- Force component: `7.924e-10 N`.
- Air-gap radial flux density: `3.002e-11 T`.
- Circuit flux linkage: `7.536e-14 Wb`.

A separate complete 32-angle Case 118 no-load batch also passed the same
comparison against a retained baseline response. Its maximum force-component
difference was `3.240e-10 N`. This larger check is for parity only: its baseline
was recorded during concurrent GPU use and is not a valid timing reference.

CTest passed 10/10, including ten Python benchmark tests. Compute Sanitizer
memcheck reported zero errors and zero leaked bytes; synccheck reported zero
errors. The self-test also repeated its deterministic solve 100 times with
zero GPU-memory growth. Benchmark failures, timeouts, and Ctrl+C now write
`FAILED` and the reason instead of leaving a misleading `RUNNING` report.

Build: MSVC 19.44, CUDA 13.3, Release, CUDA architecture 86. Local evidence is
`pcg-inexact-20261021/isolated-final-ab/report.json`, with raw responses and
input hashes. `full-batch-parity.md`, `memcheck-final.log`, and
`synccheck-final.log` are in `pcg-inexact-20261021`.

- Baseline executable SHA-256:
  `db2971cb0098b357a4b84204ba3c065f312f01c0e7c7ca07e0415f4a0d25e70e`.
- Candidate executable SHA-256:
  `c3a4f10abbeb7c4b734073b9773c07292ef3f62dca05d52efd6d31edf2a67849`.
- Candidate build directory: `build-pcg-inexact-20261021/Release`.
  `gpu_linear_p1_poc.exe` and `femm_gpu.exe` are byte-identical.

The earlier concurrent measurements are excluded: another optimization was
using the GPU. These short saved-input measurements do not establish an
end-to-end optimization speedup. Existing executables and analysis settings
were not replaced. This measurement predates the block-preconditioner work
above. Tangle and FEMM 5 communication remain unimplemented.
The sector timing does not show a benefit; the small dynamics difference is
also within the run-to-run spread. Do not describe this as a uniform speedup.

## PCG kernels, 2026-09-09

Baseline: `c3312ae` (bounded nonlinear recovery). Implementation and the bounded
validation below are complete.

## Scope and contracts

Optimize the common CUDA CSR solver, not a rotor/profile-specific path. Keep
the standalone CLI, planar/axisymmetric and periodic models, FP64 arithmetic,
linear true-residual verification, nonlinear convergence thresholds, and
unsupported-device fallback. Do not change source meshes, material curves,
user checkpoints, executable provenance, or stock CPU FEMM.

Existing profile evidence: fifteen saved 32-item suspension batches spent
528.56 of 621.47 seconds inside PCG kernel/synchronization (85.05%); device
assembly was 9.81 seconds (1.58%). The effective same-operator width was four.
These are solver timings, not CAD/mesh or complete optimization timings.

## Implementation and validation plan

1. Fuse cooperative-kernel row-local producer/consumer work. Remove barriers
   only where the same thread writes and consumes the same row; retain every
   cross-row dependency, scalar reduction, collective branch, and true-residual
   check. Keep reduction order fixed within a chosen partition.
2. Measure reuse of the existing cooperative shared-operator path for a large
   single RHS, including nonlinear recovery. Keep small solves on the existing
   single-block path and retain capability/occupancy fallback.
   Also test a deterministic shape/device-based shard count (up to 32), so a
   single large system is not restricted to eight blocks on a larger GPU.
3. Add synthetic CSR tests covering tails, distinct/mixed RHS, early exit,
   failure, shared/replicated operators, and buffer reuse. Run all CUDA/neutral
   CLI reference tests, CUDA memory/synchronization checks, and short saved-input
   old/new benchmarks with explicit observable parity checks.
4. Keep only measured improvements. Record final evidence and limitations here;
   build a separate executable and leave existing executable hashes untouched.

No claim of a defect-free solver or hardware-independent speedup is made.

## Intermediate checks

- Row-local fusion alone: saved four-RHS request, alternating A/B three times,
  median wall 6.447 -> 5.829 s; all compared observables bit-identical.
- Fusion plus large-single shared-operator execution: separate three-round
  A/B, four RHS 6.704 -> 6.127 s; saved Case 11 recovery 39.079 -> 27.958 s.
- The final CUDA/neutral CLI suite passed 10/10 (4.87 s). It includes 4,489-
  and 16,641-row synthetic SPD matrices, widths 1/2/4/5/32, manufactured
  solutions and recomputed true residuals, mixed failed/zero/live RHS,
  reciprocal/division paths, shared/replicated operators, buffer reuse, and
  repeatability. Existing planar/axisymmetric/P/AP and material/postprocessing
  references also pass. The Python parity gate passes 8/8 unit tests.
- Compute Sanitizer memcheck: 0 errors, 0 bytes leaked. Synccheck: 0 errors.
  Both run the complete self-test, including 8/16/32-shard partitions.
- With `CUDA_VISIBLE_DEVICES=-1`, a valid batch returns exit code 1 and
  `GPU_FEMM_GPU_UNAVAILABLE`; it does not silently run a different backend.
- CPU-only configuration succeeds with CUDA disabled. Building legacy `fkn`
  is blocked in MSVC by missing MFC components (MSB8041), and in the existing
  MinGW build by `liblua/complex.h`'s MSVC `__int64` declarations. Those sources
  are unchanged and outside this performance patch; do not claim CPU build PASS.

## Release measurements

RTX 4060 Ti, CUDA 13.3, MSVC 19.44, Release, `sm_86` plus PTX. Compared with
the previous recovery executable SHA-256
`9db5ecad1120b5c247cb379f306ea022acf49fc098a18fadb5a5446aee800cbc`.
Candidate SHA-256:
`db2971cb0098b357a4b84204ba3c065f312f01c0e7c7ca07e0415f4a0d25e70e`.
Both `femm_gpu.exe` and `gpu_linear_p1_poc.exe` contain the same bytes.

Each entry uses three runs per executable, alternating order. Timings below
are process wall medians, including input verification and postprocessing.

| Saved workload | Previous | Candidate | Speedup |
|---|---:|---:|---:|
| Four distinct RHS, one nonlinear mesh | 6.701 s | 5.795 s | 1.156x |
| Zero-current nonlinear mesh, with air-gap sampling | 4.056 s | 2.876 s | 1.410x |
| Saved Case 11, nonlinear recovery | 38.177 s | 19.294 s | 1.979x |
| Saved Case 66, nonlinear recovery | 44.691 s | 22.445 s | 1.991x |

All four release requests passed the observable comparison in all 24 process
runs (42 item solves, including seven distinct states). No comparison tolerance
was increased after observing results. Maximum differences: torque `9.83e-12 N m`,
Fx/Fy components `3.88e-10 N`,
circuit flux `3.66e-14 Wb`, air-gap B `7.11e-11 T`. These compare two binaries
on identical inputs, not independently generated meshes.

Local raw evidence is retained under `pcg-performance-20260909/release-normal`,
`release-case11`, and `release-case66`, outside the source repositories. Each
`report.json` is `PASS` and records executable and request hashes. Early
experimental reports are not the release evidence. No full CAD/optimization
sweep or other-GPU performance run was performed. Standalone test fixtures
provide independent analytic/stock-FEMM references; the two hard saved recovery
states were checked against the previous converged GPU executable, not a new
stock-CPU solve. The change does not address mesh-discretization error.

## Reproducing a short comparison

From the repository root, with Python 3.11 or later:

```powershell
python gpu_solver/tests/benchmark_saved_batch.py `
  --baseline C:/solvers/previous/femm_gpu.exe `
  --candidate C:/solvers/current/femm_gpu.exe `
  --request C:/benchmarks/saved-batch.json `
  --output C:/benchmarks/new-comparison-directory --rounds 3
```

The request must reference existing immutable mesh artifacts. The script never
starts CAD, prepares meshes, edits requests, or touches analysis checkpoints.
The output directory must not exist; responses, hashes, per-run profiles,
maximum observable differences and median wall times are retained there. A
report is successful only with `status: PASS`. The parity check keeps status,
hashes and integer counts exact and tests numeric observables with `atol=1e-9`,
`rtol=1e-8`; these are benchmark comparison bounds, not solver tolerances.
Each solve has a 180-second default timeout. Run benchmarks serially on an
otherwise idle GPU. Timing is measured again after a source/build change.

No existing executable, source IPT, mesh, results, checkpoint, cache or lock
was replaced. Changing the executable changes its hash. Keep old results bound
to their original binary; this patch does not bypass caller-side resume checks.

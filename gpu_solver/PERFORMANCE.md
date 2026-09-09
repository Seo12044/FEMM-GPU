# PCG performance work, 2026-09-09

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

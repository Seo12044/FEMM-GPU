# GPU linear P1 PoC

This is an independent CUDA FP64 proof of concept, not a replacement for the
FEMM solvers. It solves one planar, linear magnetic-vector-potential P1 model
with zero/nonzero Dirichlet `A` boundaries, element current density, and
element `B = (dA/dy, -dA/dx)`. The current implementation uses a deterministic
GPU-resident CSR Jacobi-PCG path.

It requires CUDA Toolkit 13.x, CMake, a CUDA-capable NVIDIA GPU, and an MSVC
x64 host toolchain. All field values and solver storage are `double` (FP64).

## Build and test on Windows

In PowerShell, choose an architecture appropriate for the target GPU. If `nvcc`
is already on `PATH`, no environment setup is required. `CUDA_PATH` is an
optional convenience for a conventional CUDA installation:

```powershell
# Optional when nvcc is not already discoverable on PATH.
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
cmake -S . -B build-gpu -G 'Visual Studio 17 2022' -A x64 `
  -DBUILD_GPU_SOLVER=ON -DBUILD_TESTING=ON `
  -DGPU_SOLVER_CUDA_ARCHITECTURES=89
cmake --build build-gpu --config Release `
  --target gpu_linear_p1_poc gpu_bh_curve_poc
ctest --test-dir build-gpu -C Release --output-on-failure
```

For a Visual Studio generator without `CUDA_PATH`, select the installed CUDA
toolset directory explicitly:

```powershell
cmake -S . -B build-gpu -G 'Visual Studio 17 2022' -A x64 `
  -T 'cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' `
  -DBUILD_GPU_SOLVER=ON -DBUILD_TESTING=ON `
  -DGPU_SOLVER_CUDA_ARCHITECTURES=89
```

Alternatively, leave `GPU_SOLVER_CUDA_ARCHITECTURES` empty and set the standard
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` override. No architecture is hard-coded.
The target locally forwards `/utf-8` to the MSVC host compiler through NVCC, so
NVIDIA header C4819 warnings do not alter legacy FEMM target flags.

## Fixture and expected contract

The built-in self-test is a counter-clockwise, four-triangle unit square in SI
units: coordinates and depth are metres, reluctivity is m/H, `J/I` is 1/m^2,
`A` is Wb/m, and `B` is tesla. Nodes 0--3 are zero-`A` boundaries; node 4 is
the centre. With depth 1 m, reluctivity 1 m/H, and current 12 A, it must return:

- `A = [0, 0, 0, 0, 1]` Wb/m;
- triangle `Bx = [2, 0, -2, 0]` T and `By = [0, 2, 0, -2]` T;
- flux linkage `1/3` Wb.

The test also verifies sign reversal at -12 A, residual <= `1e-13`, a forced
iteration-limit failure, 100-run bitwise determinism, and no material GPU-memory
growth.

The second CTest uses the frozen `tests/fixtures/linear_square_v1` CPU FEMM
reference. It independently exercises both paths below:

- reassembly in SI units followed by nodal A, element B, and circuit flux
  comparison against the frozen `.ans` and postprocessor export;
- direct solution of fkn's `.m`/`.dat` algebraic dump, including strict
  duplicate/mirror canonicalization, followed by fkn's centimetre-unit
  conversion `A = 100 mu0 V`.

The fixture is deliberately limited to one planar DC, linear-mu, PM-free,
current-driven region with a zero-A outer boundary. It does not broaden the
solver's supported physics.

## Nonlinear B-H interpolation prerequisite

`gpu_bh_curve_poc` is a separate prerequisite for nonlinear assembly. It
reproduces FEMM's DC natural cubic Hermite B-H preprocessing and evaluation,
including the derivative-root monotonicity check, FEMM's three-point smoothing
fallback, zero-field limit, final-slope extrapolation, and negative-B symmetry.
Host and CUDA evaluations are compared with the frozen repository-local
`35PN230` table and reproducible MATLAB reference samples at every knot and
segment midpoint. This scope is raw DC, `LamType=0`, `LamFill=1`; it does not
claim parity for FEMM's laminated or AC apparent-curve transformations.

This target does not yet connect B-H data to the mesh solver. Newton assembly,
iteration/convergence control, permanent magnets, and nonlinear field/flux
parity remain deferred to separate validated slices.

## Status codes

`0 OK`; `1 INPUT_IO`; `2 INVALID_ARGUMENT`; `3 UNSUPPORTED_FEATURE`;
`4 MESH_INVALID`; `5 BOUNDARY_INVALID`; `6 GPU_UNAVAILABLE`;
`7 GPU_ALLOCATION_FAILED`; `8 ASSEMBLY_FAILED`; `9 LINEAR_SOLVE_NOT_CONVERGED`;
`10 LINEAR_SOLVE_BREAKDOWN`; `11 NUMERICAL_NONFINITE`; `12 OUTPUT_IO`;
`13 INTERNAL_ERROR`.

## Explicit exclusions and gate status

Excluded from the mesh solver: AC/complex solves, nonlinear B-H/Newton iteration, axisymmetry,
air-gap elements, periodic or anti-periodic constraints, circuit unknowns,
remeshing/parameter sweeps, and integration into the MFC/FEMM executable.

The frozen test demonstrates the linear Gate 2 field/flux parity slice. It is
not evidence for nonlinear, permanent-magnet, force/torque, batch, or MATLAB
integration support.

The current deterministic PCG kernel intentionally uses one CUDA thread.
It is an accuracy and contract PoC, not a performance result; parallel sparse
kernels are deferred until FEMM-derived parity passes.

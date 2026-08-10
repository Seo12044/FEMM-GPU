# FEMM-GPU

FEMM-GPU is a CUDA solver for two-dimensional planar magnetostatic problems.
It reads a preprocessed P1 triangle mesh, materials, circuits, and boundary
conditions from a JSON artifact. Results are written as JSON and include field
values, flux linkage, force, torque, and radial air-gap flux density.

The GPU solver is a separate executable under `gpu_solver/`. It does not modify
or replace an installed copy of FEMM. The original FEMM source remains in this
repository for reference and build compatibility.

## Supported model

- Planar DC magnetics
- P1 triangle elements and FP64 arithmetic
- Linear materials and raw-DC nonlinear B-H curves
- GPU-resident nonlinear matrix and right-hand-side assembly
- Permanent-magnet coercivity
- Multiple series-circuit current inputs
- Fixed Dirichlet A boundaries
- Native FEMM periodic air-gap elements for centered sliding-band rotation
- Circuit flux linkage
- Weighted-stress force and torque
- Radial air-gap B sampling
- Single-sample and shared-geometry batch solves
- Deterministic PCG results for identical inputs

The solver does not support AC or complex problems, axisymmetric models,
laminated or AC apparent B-H conversion, general periodic node constraints,
eccentric sliding interfaces, circuit unknowns, or adaptive remeshing. The
standalone preparer accepts `.fem` files only when every feature can be mapped
exactly to this supported subset.

## Requirements

The current Windows build has been tested with:

- Windows x64
- Visual Studio 2022 with Desktop development with C++
- CUDA Toolkit 13.x
- CMake 3.18 or later
- An NVIDIA CUDA-capable GPU

Set the CUDA architecture for the target GPU. The tested RTX 4060 Ti build uses
architecture `89`.

## Build

Run the following commands in PowerShell:

```powershell
git clone https://github.com/Seo12044/FEMM-GPU.git
Set-Location FEMM-GPU

# Only needed when nvcc is not already on PATH.
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
$env:Path = "$env:CUDA_PATH\bin;$env:Path"

cmake -S . -B build-gpu -G 'Visual Studio 17 2022' -A x64 `
  -DBUILD_GPU_SOLVER=ON `
  -DBUILD_TESTING=ON `
  -DGPU_SOLVER_CUDA_ARCHITECTURES=89

cmake --build build-gpu --config Release `
  --target gpu_linear_p1_poc gpu_bh_curve_poc gpu_femm_mesh_noop_solver
```

If `nvcc` is not on `PATH`, the CUDA toolset can instead be selected directly:

```powershell
cmake -S . -B build-gpu -G 'Visual Studio 17 2022' -A x64 `
  -T 'cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' `
  -DBUILD_GPU_SOLVER=ON `
  -DBUILD_TESTING=ON `
  -DGPU_SOLVER_CUDA_ARCHITECTURES=89
```

The executables are written to:

```text
build-gpu\gpu_solver\Release\gpu_linear_p1_poc.exe
build-gpu\gpu_solver\Release\femm_gpu.exe
build-gpu\gpu_solver\Release\gpu_bh_curve_poc.exe
build-gpu\gpu_solver\Release\gpu_femm_mesh_noop_solver.exe
```

`gpu_femm_mesh_noop_solver.exe` is a mesh-preparation helper. A caller can stage
a temporary copy of FEMM and place the helper in that copy as `fkn.exe`. FEMM
then runs its normal Triangle mesh generation, but no CPU magnetic solve is
performed. The Windows build uses the GUI subsystem so repeated mesh generation
does not flash console windows. Keep the helper beside
`gpu_linear_p1_poc.exe`; never copy it into a stock FEMM installation.

`femm_gpu.exe` and `gpu_linear_p1_poc.exe` are byte-identical. The neutral name
is intended for standalone use; the historical name remains available for
existing integrations.

## Test

Run the complete CUDA test set:

```powershell
ctest --test-dir build-gpu -C Release --output-on-failure
```

Run only the built-in solver test:

```powershell
$solver = '.\build-gpu\gpu_solver\Release\gpu_linear_p1_poc.exe'
& $solver --self-test
```

Run the included nonlinear fixture and inspect the response:

```powershell
& $solver --single-sample `
  '.\gpu_solver\tests\fixtures\nonlinear_pm_coil_v1\single_sample_request.json' `
  '.\build-gpu\single_sample_response.json'

Get-Content '.\build-gpu\single_sample_response.json'
```

## Standalone planar DC solve

The standalone interface does not depend on MATLAB, Inventor, or a motor
project. Check the installed solver first:

```powershell
$solver = '.\build-gpu\gpu_solver\Release\femm_gpu.exe'
& $solver --capabilities
```

Validate an immutable mesh artifact and solve it with the Python standard
library frontend:

```powershell
python .\gpu_solver\tools\femm_gpu.py prepare `
  '.\model.fem' '.\model.gpu.json' `
  --femm-root 'C:\femm42' `
  --mesh-noop '.\build-gpu\gpu_solver\Release\gpu_femm_mesh_noop_solver.exe'

python .\gpu_solver\tools\femm_gpu.py --solver $solver `
  validate '.\model.gpu.json'

python .\gpu_solver\tools\femm_gpu.py --solver $solver `
  solve '.\model.gpu.json' '.\response.json' `
  --currents 2 -2 --fields
```

`prepare` copies the FEMM runtime to a temporary directory and replaces
`fkn.exe` only in that copy. It uses stock FEMM's Triangle mesher but performs
no CPU magnetic solve. The input `.fem` and the installed FEMM directory are
read-only inputs. Unsupported model features stop before the artifact is
written.

`--fields` writes nodal magnetic vector potential and per-element Bx/By. Use
`--force-torque --force-group 20 --air-group 30` for weighted-stress force and
torque on a conforming mesh. Native sliding-band artifacts use
`--sliding-band-angle-deg` and do not require motor identity fields.

The request contract is
[`gpu_femm_planar_dc_sample_v1`](gpu_solver/schemas/gpu_femm_planar_dc_sample_v1.schema.json).
Applications in C++, Python, MATLAB, or another language may invoke
`femm_gpu.exe --solve request.json response.json` directly.

FEMM model editing remains a separate step. Unsupported AC, axisymmetric, and
periodic-node models are rejected rather than approximated.

## Run a motor artifact

Check the structure and identity fields of a `gpu_femm_mesh_v1` or
`gpu_femm_mesh_v2` artifact:

```powershell
& $solver --mesh-artifact '.\model.gpu_femm_mesh_v1.json'
```

Run one operating point:

```powershell
& $solver --motor-single-sample '.\request.json' '.\response.json'
```

A single-sample request has the following form. The SHA-256 values, pose, and
circuit count must match the mesh artifact exactly. A v1 artifact represents
one posed conforming mesh. A v2 artifact is a centered zero-degree reference
with a native periodic air-gap element; its request may change
`rotor_angle_deg` while `displacement_mm` remains `[0,0]`.

```json
{
  "protocol": "gpu_femm_motor_sample_v1",
  "mesh_artifact_path": "model.gpu_femm_mesh_v1.json",
  "mesh_artifact_sha256": "<artifact SHA-256>",
  "base_motor_fem_sha256": "<base model SHA-256>",
  "source_fem_sha256": "<posed model SHA-256>",
  "circuit_currents_A": [2.0, -2.0],
  "selected_group_number": 20,
  "air_group_number": 30,
  "airgap_radius_mm": 15.0,
  "airgap_angles_deg": [0.0, 90.0, 180.0, 270.0],
  "rotor_angle_deg": 0.0,
  "displacement_mm": [0.0, 0.0]
}
```

Calculate the artifact hash in PowerShell:

```powershell
(Get-FileHash '.\model.gpu_femm_mesh_v1.json' -Algorithm SHA256).Hash.ToLower()
```

## Run a batch

Use a batch when several current vectors share one geometry artifact. The
solver reuses the decoded artifact and GPU matrix structure. Nonlinear numeric
assembly stays on the GPU between Newton iterations. If device assembly cannot
be initialized, the solver uses the established host assembly path for that
operator and keeps the same request and response format.

```powershell
& $solver --motor-batch '.\batch_request.json' '.\batch_response.json'
```

The batch envelope is:

```json
{
  "protocol": "gpu_femm_motor_batch_v1",
  "max_items_per_chunk": 4,
  "items": [
    {
      "task_id": "operating_point_0001",
      "request": {}
    }
  ]
}
```

Replace the empty `request` object with a complete
`gpu_femm_motor_sample_v1` request. Use the profiling command when stage timing
is needed:

```powershell
& $solver --motor-batch-profile '.\batch_request.json' '.\batch_response.json'
```

A successful response has `status: "PASS"` and process exit code 0. On failure,
check `solve_status`, `error_identifier`, and `error_message`. See
[gpu_solver/README.md](gpu_solver/README.md) for the file formats and status
codes.

For v2 sliding-band artifacts, the air-gap coupling and rotor mapping are
rebuilt for each requested angle. Items with the same verified artifact and
exact mechanical pose can share a CUDA batch. Different poses are partitioned
before the solve, so one angle's air-gap operator is never reused for another.
The device assembly uses a fixed contributor order, so repeated runs of the
same request are deterministic.

## License

The original FEMM source is covered by [license.txt](license.txt).

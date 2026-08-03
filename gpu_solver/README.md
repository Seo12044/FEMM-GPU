# GPU solver CLI and file formats

`gpu_linear_p1_poc` is a CUDA FP64 planar P1 magnetostatic solver. Its command
line interface uses files for both input and output. It does not modify an
installed FEMM directory.

See the repository [README](../README.md) for build instructions and a quick
start.

## Commands

| Command | Purpose |
|---|---|
| `--self-test` | Test the analytic models, parsers, batch path, and determinism |
| `--mesh-artifact <file>` | Validate a `gpu_femm_mesh_v1` or `gpu_femm_mesh_v2` artifact |
| `--motor-single-sample <request> <response>` | Solve one motor operating point |
| `--motor-batch <request> <response>` | Solve a batch of operating points |
| `--motor-batch-profile <request> <response>` | Solve a batch and record stage timings |
| `--femm-reference <stem>` | Check the frozen linear reference |
| `--nonlinear-reference <stem> <curve_dir>` | Check the frozen nonlinear field and flux reference |
| `--postprocess-reference <stem> <curve_dir>` | Check the frozen force, torque, and air-gap reference |
| `--single-sample <request> <response>` | Run the frozen fixture file protocol |

`gpu_bh_curve_poc --fixture <directory>` tests B-H preprocessing and
interpolation separately.

## `gpu_femm_mesh_v1`

Normal motor solves use a preprocessed mesh artifact. The top-level object has
exactly these fields:

```text
schema_version
base_motor_fem_sha256
source_fem_sha256
canonical_identity_sha256
resolved
```

The `resolved` object contains:

```text
source_fem_sha256
base_motor_fem_sha256
model
pose
nodes_mm
triangles
regions
materials
circuits
outer_dirichlet
```

Artifact requirements:

- `model.problem_type` must be `planar` and `frequency_hz` must be `0`.
- Node coordinates and model depth use millimetres.
- Triangle node indices are zero-based and counter-clockwise.
- Every triangle has a valid region and material.
- Materials are isotropic and use `lam_type=0`, `lam_fill=1`.
- Circuit indices, region turns, and PM magnetization angles are explicit.
- Outer Dirichlet nodes and their `A_Wb_per_m` values are explicit.
- Source and base SHA-256 values match at the top level and in `resolved`.

Unknown fields, duplicate JSON keys, invalid indices, and unsupported physics
are rejected.

```powershell
gpu_linear_p1_poc.exe --mesh-artifact model.gpu_femm_mesh_v1.json
```

This command prints the node, triangle, and circuit counts together with the
source and base hashes. It validates the artifact but does not solve it.

## `gpu_femm_mesh_v2`

Version 2 keeps the v1 fields and adds `resolved.air_gap_elements`. Each entry
contains the native FEMM periodic air-gap geometry, sector count, reference
inner/outer shifts, and quadrature node indices and weights. The artifact is a
centered zero-degree reference mesh. At solve time the GPU path applies the
requested rotor angle by cyclically remapping the inner ring and evaluating the
native 10-by-10 air-gap element matrix; the stator mesh remains fixed.

V2 validation is strict: the preserved boundary name, periodicity, center,
radii, arc length, sector count, indices, and weights must be internally
consistent. Unknown fields and identity mismatches are rejected just as in v1.
Eccentric displacement is not supported by this interface.

## `gpu_femm_motor_sample_v1`

A single-sample request contains:

| Field | Description |
|---|---|
| `mesh_artifact_path` | Path to the mesh artifact |
| `mesh_artifact_sha256` | SHA-256 of the complete artifact file |
| `base_motor_fem_sha256` | Base-model hash stored in the artifact |
| `source_fem_sha256` | Posed-model hash stored in the artifact |
| `circuit_currents_A` | Current vector in artifact circuit order |
| `selected_group_number` | Group used for force and torque integration |
| `air_group_number` | Air group used by the weighted-stress mask |
| `airgap_radius_mm` | Air-gap sample radius; may be zero when sampling is disabled |
| `airgap_angles_deg` | Angles for radial B samples |
| `rotor_angle_deg` | V1: angle stored in the pose. V2: runtime sliding-band angle |
| `displacement_mm` | V1: displacement stored in the pose. V2: must be `[0,0]` |

The solve does not start if the artifact file hash, model identity, pose, or
circuit count does not match.

A successful response contains:

- `Fx_N`, `Fy_N`, and `torque_Nm`
- `actual_circuit_currents_A`
- `circuit_flux_linkage_Wb`
- `airgap_sample_angles_deg`
- `airgap_radial_flux_density_T`
- `mesh_element_count`
- Newton iteration count and residual

## `gpu_femm_motor_batch_v1`

A batch request contains `max_items_per_chunk` and `items`. Each item has a
unique `task_id` and one complete `gpu_femm_motor_sample_v1` request.

- Responses preserve input order.
- Each shared artifact is read and validated once.
- CSR symbolic data is reused for matching geometry.
- `max_items_per_chunk` must be between 1 and 4096.
- The effective chunk size also accounts for available VRAM.
- V2 angle-dependent operators currently use an effective chunk size of one.
- Duplicate task IDs and mismatched artifact identities are rejected.

`--motor-batch-profile` returns the same numerical results as `--motor-batch`
and adds timing for artifact loading, assembly, GPU solve, and postprocessing.

## Numerical implementation

- Field values and solver storage use `double`.
- Nonlinear materials use FEMM's DC natural cubic Hermite B-H preprocessing.
- The first Newton iteration uses the cold secant; later iterations use the
  analytic Jacobian.
- Linear systems use a GPU-resident CSR Jacobi-PCG solver.
- Supported GPUs can use cooperative multi-block PCG for small batches on
  large meshes. Other cases use the deterministic fallback kernel.
- V1 force and torque use a default weighted-stress mask; radial air-gap B uses
  smoothed nodal values from the P1 element field.
- V2 force, torque, and radial air-gap B use the native air-gap element Fourier
  reconstruction.

## Status codes

The process returns 0 on success and a nonzero value for input, solve, or
output failure. The JSON `solve_status` uses these stable names:

| Code | Name | Meaning |
|---:|---|---|
| 0 | `OK` | Success |
| 1 | `INPUT_IO` | Input file could not be read |
| 2 | `INVALID_ARGUMENT` | Invalid schema, hash, or argument |
| 3 | `UNSUPPORTED_FEATURE` | Unsupported physics |
| 4 | `MESH_INVALID` | Invalid mesh structure |
| 5 | `BOUNDARY_INVALID` | Invalid boundary conditions |
| 6 | `GPU_UNAVAILABLE` | No usable CUDA device |
| 7 | `GPU_ALLOCATION_FAILED` | GPU allocation failed |
| 8 | `ASSEMBLY_FAILED` | Matrix assembly failed |
| 9 | `LINEAR_SOLVE_NOT_CONVERGED` | PCG iteration limit reached |
| 10 | `LINEAR_SOLVE_BREAKDOWN` | PCG breakdown |
| 11 | `NUMERICAL_NONFINITE` | NaN or Inf encountered |
| 12 | `OUTPUT_IO` | Response file could not be written |
| 13 | `INTERNAL_ERROR` | Internal error |
| 14 | `NONLINEAR_SOLVE_NOT_CONVERGED` | Newton iteration limit reached |
| 15 | `INVALID_MATERIAL` | Invalid material or B-H curve |

## Test fixtures

`tests/fixtures` contains three fixed references:

- `linear_square_v1`: linear field and flux
- `nonlinear_pm_coil_v1`: nonlinear steel, PM, coil, force, torque, and air-gap field
- `35PN230_v1`: B-H preprocessing and interpolation

All CTest cases, including the analytic self-test, execute CUDA kernels.

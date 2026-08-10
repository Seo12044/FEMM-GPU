# GPU solver CLI and file formats

`gpu_linear_p1_poc` is a CUDA FP64 planar and axisymmetric P1 magnetostatic
solver. Its command line interface uses files for both input and output. It
does not modify an installed FEMM directory.

See the repository [README](../README.md) for build instructions and a quick
start.

## Commands

| Command | Purpose |
|---|---|
| `--self-test` | Test the analytic models, parsers, batch path, and determinism |
| `--mesh-artifact <file>` | Validate a neutral or legacy GPU FEMM mesh artifact |
| `--capabilities` | Print the supported physics and neutral protocol as JSON |
| `--solve <request> <response>` | Solve one project-neutral DC magnetostatic problem |
| `--motor-single-sample <request> <response>` | Solve one motor operating point |
| `--motor-batch <request> <response>` | Solve a batch of operating points |
| `--motor-batch-profile <request> <response>` | Solve a batch and record stage timings |
| `--femm-reference <stem>` | Check the frozen linear reference |
| `--nonlinear-reference <stem> <curve_dir>` | Check the frozen nonlinear field and flux reference |
| `--postprocess-reference <stem> <curve_dir>` | Check the frozen force, torque, and air-gap reference |
| `--single-sample <request> <response>` | Run the frozen fixture file protocol |

`gpu_bh_curve_poc --fixture <directory>` tests B-H preprocessing and
interpolation separately.

## `gpu_femm_planar_dc_mesh_v1`

This neutral artifact contains `source_fem_sha256`, model/mesh/material data,
and a canonical content identity. It omits the legacy motor base hash and pose.
The `circuits` array may be empty for a permanent-magnet-only problem.

Create one from a supported FEMM source without MATLAB:

```powershell
python .\gpu_solver\tools\femm_gpu.py prepare model.fem model.gpu.json `
  --femm-root C:\femm42 `
  --mesh-noop .\build-gpu\gpu_solver\Release\gpu_femm_mesh_noop_solver.exe
```

The preparer stages a private FEMM runtime, invokes Triangle with the no-op
solver, validates every resolved material/region/boundary, and writes the JSON
atomically. Output paths inside the stock FEMM installation or equal to the
input model are rejected, even with `--overwrite`. It never replaces the
installed `fkn.exe`. The exact artifact contract is in
[`schemas/gpu_femm_planar_dc_mesh_v1.schema.json`](schemas/gpu_femm_planar_dc_mesh_v1.schema.json).

## `gpu_femm_planar_dc_sample_v1`

This is the project-neutral single-sample protocol used by `--solve`. It binds
the request to the complete artifact SHA-256 and supplies circuit currents in
artifact order. It has no MATLAB, Inventor, motor-model hash, rotor-group, or
posed-file dependency.

```json
{
  "protocol": "gpu_femm_planar_dc_sample_v1",
  "mesh_artifact_path": "model.gpu_femm_mesh_v1.json",
  "mesh_artifact_sha256": "<64 lowercase hex characters>",
  "circuit_currents_A": [2.0, -2.0],
  "force_group_number": -1,
  "stress_air_group_number": -1,
  "airgap_radius_mm": 0.0,
  "airgap_angles_deg": [],
  "sliding_band_angle_deg": 0.0,
  "compute_force_torque": false,
  "include_field_solution": true
}
```

Set both group numbers to `-1` when weighted-stress postprocessing is not
needed. On a conforming v1 mesh, force, torque, or radial-B sampling requires a
valid selected/air group pair. A v2 native air-gap element computes force,
torque, and radial B directly from the air-gap field. Full-field output adds
`node_A_Wb_per_m`, `element_Bx_T`, and `element_By_T`; leave it disabled for
compact scalar responses.

`solve_status` and `postprocess_status` are separate. If optional
postprocessing fails after the field solve succeeds, currents, flux linkage,
mesh counts, and requested full-field arrays remain in the response.
The Python frontend validates the protocol, artifact hash, status fields, and
array sizes in a temporary response before atomically publishing it. A missing
or malformed solver response leaves any existing output file unchanged.

The exact request schema is in
[`schemas/gpu_femm_planar_dc_sample_v1.schema.json`](schemas/gpu_femm_planar_dc_sample_v1.schema.json).

## Generic axisymmetric and periodic artifacts

The preparer emits `gpu_femm_magnetostatic_mesh_v1` when the FEMM source is
axisymmetric or contains ordinary periodic/anti-periodic boundaries. The
artifact adds `resolved.node_constraints`; each entry states
`A(node_b)=A(node_a)` or `A(node_b)=-A(node_a)`. Cycles are checked for sign
contradictions before the artifact is written and again before assembly.

`--solve` automatically selects `gpu_femm_magnetostatic_sample_v1` for this
artifact. The request fields and command line are otherwise unchanged. Generic
responses use:

- `node_potential`, with `node_potential_quantity` and `node_potential_unit`
- `element_B_component_1_T` and `element_B_component_2_T`
- `field_components` equal to `Bx, By` for planar or `Br, Bz` for axisymmetric

For axisymmetric models, the nodal potential is FEMM's poloidal flux function
in Wb. Internally, the solver uses conventional P1 A-phi interpolation and a
seven-point triangle quadrature. Matrix assembly runs on the host; the reduced
FP64 system is solved by the existing CUDA CSR PCG backend. General periodic
constraints use a signed degree-of-freedom map and exact `T^T K T` reduction.

The generic protocol intentionally rejects force/torque, radial air-gap
sampling, and sliding-band rotation. Those postprocessors currently assume
the legacy planar operator. The schemas are
[`schemas/gpu_femm_magnetostatic_mesh_v1.schema.json`](schemas/gpu_femm_magnetostatic_mesh_v1.schema.json)
and
[`schemas/gpu_femm_magnetostatic_sample_v1.schema.json`](schemas/gpu_femm_magnetostatic_sample_v1.schema.json).

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
Eccentric displacement is not supported by this interface. Requested selected
and air groups must exist in the artifact. When radial B samples are requested,
`airgap_radius_mm` must lie inside the native air-gap annulus; FEMM's AGE
Fourier reconstruction reports the field at the annulus mean radius.

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
- V2 items are partitioned by verified artifact and exact mechanical pose.
  Compatible items use the effective chunk width; different poses are solved
  in separate operator groups while response order is preserved.
- Duplicate task IDs and mismatched artifact identities are rejected.

`--motor-batch-profile` returns the same numerical results as `--motor-batch`
and adds timing for artifact loading, host or device assembly, GPU solve, and
postprocessing. `device_assembly_seconds` is zero when the host fallback path
is used; `host_assembly_seconds` is zero when device assembly is active.

## Numerical implementation

- Field values and solver storage use `double`.
- Nonlinear materials use FEMM's DC natural cubic Hermite B-H preprocessing.
- The first Newton iteration uses the cold secant; later iterations use the
  analytic Jacobian.
- Batch nonlinear numeric assembly writes CSR values, the Jacobi diagonal, and
  the right-hand side directly to device memory. Triangle and air-gap
  contributions use a fixed reduction order.
- The established host nonlinear assembly remains available and is selected
  automatically if device-plan setup or execution fails.
- Linear systems use a GPU-resident CSR Jacobi-PCG solver.
- Axisymmetric and general periodic operators assemble on the host and use the
  same GPU-resident PCG solve. Planar models without these constraints retain
  the existing device-assembly path.
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

`tests/fixtures` contains six fixed references:

- `linear_square_v1`: linear field and flux
- `nonlinear_pm_coil_v1`: nonlinear steel, PM, coil, force, torque, and air-gap field
- `35PN230_v1`: B-H preprocessing and interpolation
- `periodic_strip_v1`: general periodic-node reduction against stock FEMM
- `antiperiodic_strip_v1`: signed anti-periodic reduction against stock FEMM
- `axisymmetric_coil_v1`: axisymmetric potential and flux against stock FEMM

The solver and numerical reference tests execute CUDA kernels. The preparer
parser and mesh no-op helper tests do not require a GPU.

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
installed `fkn.exe`. The input is fingerprinted before meshing, parsed from
the staged copy, and checked again immediately before publication. The solver
recomputes the canonical identity for project-neutral artifacts instead of
trusting the stored value. The exact artifact contract is in
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
Direct `gpu_linear_p1_poc.exe --solve` calls also publish through a sibling
temporary file and reject response paths that alias the request, artifact, or
solver executable.

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
seven-point triangle quadrature. Geometry, quadrature, and the signed degree-of-
freedom map are prepared once on the host. For sufficiently large workloads,
the numeric matrix, diagonal, and right-hand side are assembled directly into
the reduced GPU CSR buffers before PCG. General periodic constraints use exact
`T^T K T` signs in deterministic contributor lists. Small models and any
device-plan or kernel failure use the established host assembler.
The preparer binds each generated `.pbc` pair to a matching FEMM boundary
property and checks disconnected boundary-side topology when it is available.
At a rotational-sector apex, FEMM may emit a node paired with itself. A
periodic self-pair is removed as an identity constraint; an anti-periodic
self-pair is reduced exactly to a zero-potential Dirichlet node.

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

## `gpu_femm_motor_sample_v2` and `gpu_femm_motor_batch_v2`

The v2 motor protocols are additive and do not change the strict v1 wire
format. A v2 batch contains only v2 sample requests. Each v2 sample has the v1
fields plus the required `tangent_multi_rhs` member. `null` or an empty array
requests only the established nonlinear sample. An object requests a
first-order response about that nonlinear operating point:

```json
"tangent_multi_rhs": {
  "schema_version": "gpu_femm_tangent_multi_rhs_v1",
  "rhs": [
    {
      "name": "suspension_d",
      "circuit_current_derivative_A_per_A": [1, 0]
    },
    {
      "name": "suspension_q",
      "circuit_current_derivative_A_per_A": [0, 1]
    }
  ]
}
```

The base `circuit_currents_A` is solved nonlinearly once. At its converged
field, the solver reassembles the consistent Newton Jacobian and solves every
named circuit-source derivative in one batched PCG launch. Permanent-magnet
loads and fixed-boundary offsets are not included in derivative RHS vectors.

The response's `tangent_multi_rhs` object records the operating-point
currents, linearization provenance, PCG residual `||J*dA-S||`, and, per RHS:

- `dFx_N_per_A`, `dFy_N_per_A`, and `dtorque_Nm_per_A`
- `circuit_flux_linkage_derivative_Wb_per_A`
- `airgap_radial_flux_density_derivative_T_per_A`
- `linear_convergence`

Force and torque derivatives use the direct bilinear cross-term between the
base and derivative fields. Native AGE stencils are evaluated directly; the
non-AGE weighted-stress mask is built once and reused for all RHS vectors.
Flux linkage and radial air-gap field derivatives are linear evaluations of
`dA`.

If the base nonlinear solve fails, the sample fails normally. If the base
sample passes but tangent processing fails, the sample stays `PASS` and the
nested tangent object is `FAIL` with `GPU_FEMM_TANGENT_*`. This lets a caller
retain the drive result and retry only the derivative with exact nonlinear
central differences.

## Exact 180-degree motor sector (`gpu_femm_motor_batch_v3`)

V3 is an opt-in extension for the isolated-motor performance tangent path. It
requires an actually cropped and conformingly meshed 180-degree
`gpu_femm_mesh_v3` artifact. It never infers sector pairs from a full-circle
mesh and it never approximates a 60-degree Bloch boundary. V1 and v2 remain
strictly unchanged.

The v3 artifact adds `node_constraints`, an empty `air_gap_elements` array,
and SHA-bound `sector` metadata. The metadata contains the positive integer
electrical pole-pair count `p`, suspension spatial order `p+1`, the base
periodic/anti-periodic relation, and an oriented involutive permutation of
circuit partners. `sector_start_angle_deg` binds the actual global start of
the cropped interval, so full-circle Br requests are folded into
`[start,start+180)` rather than assuming a zero-degree cut. The metadata also
carries `apex_self_pair_nodes`, kept separate from physical
`outer_dirichlet`, and the full-model C2 invariance-certificate SHA.
An apex self-pair is zero-potential only for the AP operator; the GPU removes
or adds that condition when switching from `p` to `p+1`. The base relation
must be `(-1)^p`; the GPU derives a second
signed degree-of-freedom map with relation `(-1)^(p+1)` for tangent d/q. Both
operators use the same raw nodes and triangles, so there is no remesh or index
drift.

Each v3 sample requires the following sector object. A drive sample can also
carry the v2 `tangent_multi_rhs` object; a zero/reference sample may omit it.

```json
"sector_performance": {
  "schema_version": "gpu_femm_motor_sector_performance_v1",
  "model_invariance_certificate_sha256": "<64 lowercase hex characters>"
}
```

The weighting-stress mask uses the cut-node pairs with scalar `+1` equality,
independent of the magnetic P/AP sign. The converged base and tangent fields
retain their respective magnetic signs. V3 native AGE artifacts are rejected:
the current release deliberately supports only the conforming posed-mesh WST
route, which avoids an unverified sector-start phase convention.

All returned loads, flux linkages, and radial-B samples are already normalized
to the full machine. Requested angles across 360 degrees are folded into the
stored 180-degree mesh and multiplied by the appropriate base or tangent copy
sign. Base force cancels, base torque doubles, and for the `p`/`p+1` cross
field tangent force doubles while tangent torque cancels. Circuit flux uses
the explicit partner permutation and orientation; it is never blindly scaled.
The `sector_reconstruction` response records the start angle, signs, multipliers,
partner arrays, full-machine normalization, periodic-mask provenance, and the
single-upload tangent-operator memory evidence. It echoes the certificate SHA
and apex node list so the caller can verify active reconstruction provenance.

The tangent PCG path uploads one consistent Newton matrix and its Jacobi
preconditioner, then aliases that operator across all RHS workspaces. Regular
and cooperative PCG launches both use the same physical operator. Legacy
`SolveBatch` keeps its established per-item operator storage.

`--motor-batch-profile` returns the same numerical results as `--motor-batch`
and adds timing for artifact loading, host or device assembly, GPU solve, and
postprocessing. `device_assembly_seconds` is zero when the host fallback path
is used; `host_assembly_seconds` is zero when device assembly is active.

Standalone host/device parity and timing can be checked without changing the
public `--solve` command:

```powershell
gpu_linear_p1_poc.exe --internal-solve-host request.json response-host.json
gpu_linear_p1_poc.exe --internal-solve-device request.json response-device.json
```

## Cyclic native-AGE torque sector (`gpu_femm_motor_batch_v4`)

V4 is a separate, base-field-only path for a genuinely cropped native FEMM
air-gap-element (AGE) sector. It preserves v1/v2/v3 behavior. The mesh is
`gpu_femm_mesh_v4`; its `sector` uses `gpu_femm_motor_sector_v2` and retains
the v1 field names, with `full_machine_sector_count = k` for any integer
`k >= 2` and `angle_deg = 360/k`. Cut-node pairs must be the exact rotation
through that angle. A nonempty AGE record is mandatory.

The circuit partner arrays are a signed generator, not a C2 involution:
applying its index permutation and orientations exactly `k` times must return
every circuit to itself with sign `+1`. The full-machine flux vector is
reconstructed by scatter-summing every signed generator copy.

V4 accepts only zero/drive nonlinear samples. Its request carries
`"tangent_multi_rhs": []` (or `null`) and
`"sector_performance":{"schema_version":"gpu_femm_motor_sector_performance_v2",...}`;
any tangent RHS is rejected. Displacement is rejected by the normal
sliding-band request check. Suspension/tangent work remains on the validated
full-circle route.

AGE torque is integrated on the stored sector and normalized as
`T_full = k*T_sector`; returned `Fx_N` and `Fy_N` are zero by full rotational
cancellation. FEMM constructs a virtual globally sorted 360-degree AGE ring,
even when the physical mesh is cropped. Radial-B Fourier probes therefore use
`angle mod angle_deg`, with P/AP sign from `floor(angle/angle_deg)`. Neither the
physical cut angle nor the stored q0 node angle is subtracted. The v4
`sector_reconstruction` object uses schema
`gpu_femm_motor_sector_reconstruction_v2`, reports
`output_normalization:"full_machine"`,
`load_extraction_method:"native_air_gap_element_signed_sector"`, the scalar
torque multiplier, generator arrays, start/angle/count, and the invariance
certificate SHA.

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
- Planar, axisymmetric, and signed-periodic numeric operators share the same
  deterministic device-assembly plan. Axisymmetric elements use seven stored
  quadrature points; signed constraints assemble directly in reduced CSR.
- Auto mode keeps small one-off workloads on the host and retries the exact
  Newton iteration there if device-plan setup or execution fails.
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

The axisymmetric frozen reference is a linear coil case. Permanent-magnet and
nonlinear axisymmetric models are accepted by the same material path but do
not yet have separate stock-FEMM parity fixtures.

The solver and numerical reference tests execute CUDA kernels. The preparer
parser and mesh no-op helper tests do not require a GPU.

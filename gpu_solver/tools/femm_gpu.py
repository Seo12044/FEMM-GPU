"""Small project-neutral command line frontend for the FEMM GPU solver.

The CUDA executable owns validation and numerical work.  This module only
builds the strict JSON request, fingerprints the immutable mesh artifact, and
starts the executable.  It intentionally has no MATLAB or Inventor dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence


PROTOCOL = "gpu_femm_planar_dc_sample_v1"
GENERIC_PROTOCOL = "gpu_femm_magnetostatic_sample_v1"
LEGACY_ARTIFACT_SCHEMA = "gpu_femm_planar_dc_mesh_v1"
GENERIC_ARTIFACT_SCHEMA = "gpu_femm_magnetostatic_mesh_v1"
SOLVE_STATUSES = {
    "OK",
    "INPUT_IO",
    "INVALID_ARGUMENT",
    "UNSUPPORTED_FEATURE",
    "MESH_INVALID",
    "BOUNDARY_INVALID",
    "GPU_UNAVAILABLE",
    "GPU_ALLOCATION_FAILED",
    "ASSEMBLY_FAILED",
    "LINEAR_SOLVE_NOT_CONVERGED",
    "LINEAR_SOLVE_BREAKDOWN",
    "NUMERICAL_NONFINITE",
    "OUTPUT_IO",
    "INTERNAL_ERROR",
    "NONLINEAR_SOLVE_NOT_CONVERGED",
    "INVALID_MATERIAL",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_request(
    artifact: Path,
    currents: Sequence[float],
    force_group: int = -1,
    air_group: int = -1,
    airgap_radius_mm: float = 0.0,
    airgap_angles_deg: Sequence[float] = (),
    sliding_band_angle_deg: float = 0.0,
    compute_force_torque: bool = False,
    include_field_solution: bool = False,
    protocol: str = PROTOCOL,
) -> dict[str, object]:
    artifact = artifact.resolve(strict=True)
    return {
        "protocol": protocol,
        "mesh_artifact_path": str(artifact),
        "mesh_artifact_sha256": sha256_file(artifact),
        "circuit_currents_A": list(currents),
        "force_group_number": force_group,
        "stress_air_group_number": air_group,
        "airgap_radius_mm": airgap_radius_mm,
        "airgap_angles_deg": list(airgap_angles_deg),
        "sliding_band_angle_deg": sliding_band_angle_deg,
        "compute_force_torque": compute_force_torque,
        "include_field_solution": include_field_solution,
    }


def default_solver() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    candidates = (
        root / "build-gpu" / "gpu_solver" / "Release" / "femm_gpu.exe",
        root / "build-gpu" / "gpu_solver" / "Release" / "gpu_linear_p1_poc.exe",
        root / "build-gpu" / "gpu_solver" / "femm_gpu",
        root / "build-gpu" / "gpu_solver" / "gpu_linear_p1_poc",
    )
    return next((path for path in candidates if path.is_file()), None)


def require_solver(value: str | None) -> Path:
    solver = Path(value).resolve() if value else default_solver()
    if solver is None or not solver.is_file():
        raise FileNotFoundError(
            "GPU solver was not found; pass --solver with gpu_linear_p1_poc.exe"
        )
    return solver


def run_solver(solver: Path, *arguments: str) -> int:
    return subprocess.run([str(solver), *arguments], check=False).returncode


def _atomic_replace(source: Path, destination: Path) -> None:
    """Tolerate short-lived Windows scanner/indexer sharing conflicts."""
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (2 ** attempt))


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def artifact_protocol(path: Path) -> str:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"mesh artifact is not readable JSON: {error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("schema_version"), str):
        raise ValueError("mesh artifact does not declare schema_version")
    schema = document["schema_version"]
    if schema == GENERIC_ARTIFACT_SCHEMA:
        return GENERIC_PROTOCOL
    if schema == LEGACY_ARTIFACT_SCHEMA:
        return PROTOCOL
    raise ValueError(f"unsupported standalone mesh artifact schema: {schema}")


def validate_response(
    path: Path, expected_artifact_sha256: str, returncode: int,
    expected_protocol: str = PROTOCOL,
) -> dict[str, object]:
    """Validate the solver-owned response before publishing it to callers."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            response = json.load(stream, parse_constant=_reject_json_constant)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"GPU solver produced an unreadable response: {error}") from error
    if not isinstance(response, dict):
        raise RuntimeError("GPU solver response must be a JSON object")
    expected_strings = {
        "protocol": expected_protocol,
        "mesh_artifact_sha256": expected_artifact_sha256,
    }
    for name, expected in expected_strings.items():
        if response.get(name) != expected:
            raise RuntimeError(f"GPU solver response has an invalid {name}")
    status = response.get("status")
    if status not in ("PASS", "FAIL"):
        raise RuntimeError("GPU solver response status must be PASS or FAIL")
    for name in ("solve_status", "postprocess_status", "error_identifier", "error_message"):
        if not isinstance(response.get(name), str):
            raise RuntimeError(f"GPU solver response {name} must be a string")
    if response["solve_status"] not in SOLVE_STATUSES:
        raise RuntimeError("GPU solver response has an unknown solve_status")
    if response["postprocess_status"] not in SOLVE_STATUSES | {"NOT_REQUESTED", "NOT_RUN"}:
        raise RuntimeError("GPU solver response has an unknown postprocess_status")
    field_names = (
        ("element_B_component_1_T", "element_B_component_2_T")
        if expected_protocol == GENERIC_PROTOCOL
        else ("element_Bx_T", "element_By_T")
    )
    if expected_protocol == GENERIC_PROTOCOL:
        problem_type = response.get("problem_type")
        components = response.get("field_components")
        if problem_type not in ("planar", "axisymmetric"):
            raise RuntimeError("GPU solver response has an invalid problem_type")
        expected_components = ["Bx", "By"] if problem_type == "planar" else ["Br", "Bz"]
        if components != expected_components:
            raise RuntimeError("GPU solver response has invalid field_components")
        expected_quantity = (
            "magnetic_vector_potential"
            if problem_type == "planar"
            else "poloidal_flux_function"
        )
        expected_unit = "Wb/m" if problem_type == "planar" else "Wb"
        if response.get("node_potential_quantity") != expected_quantity \
                or response.get("node_potential_unit") != expected_unit:
            raise RuntimeError("GPU solver response has invalid node potential metadata")
    node_field_name = (
        "node_potential" if expected_protocol == GENERIC_PROTOCOL else "node_A_Wb_per_m"
    )
    for name in (
        "actual_circuit_currents_A",
        "circuit_flux_linkage_Wb",
        "airgap_sample_angles_deg",
        "airgap_radial_flux_density_T",
        node_field_name,
        *field_names,
    ):
        values = response.get(name)
        if not isinstance(values, list) or not all(_finite_number(value) for value in values):
            raise RuntimeError(f"GPU solver response {name} must contain finite numbers")
    for name in ("Fx_N", "Fy_N", "torque_Nm"):
        value = response.get(name)
        if value is not None and not _finite_number(value):
            raise RuntimeError(f"GPU solver response {name} must be null or finite")
    for name in ("mesh_node_count", "mesh_element_count"):
        value = response.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"GPU solver response {name} must be a non-negative integer")
    if not isinstance(response.get("field_solution_included"), bool):
        raise RuntimeError("GPU solver response field_solution_included must be boolean")
    if len(response["actual_circuit_currents_A"]) != len(response["circuit_flux_linkage_Wb"]):
        raise RuntimeError("GPU solver response circuit result lengths do not match")
    if len(response["airgap_sample_angles_deg"]) != len(response["airgap_radial_flux_density_T"]):
        raise RuntimeError("GPU solver response air-gap result lengths do not match")
    if response["field_solution_included"]:
        field_lengths_match = (
            len(response[node_field_name]) == response["mesh_node_count"]
            and len(response[field_names[0]]) == response["mesh_element_count"]
            and len(response[field_names[1]]) == response["mesh_element_count"]
        )
    else:
        field_lengths_match = not (
            response[node_field_name]
            or response[field_names[0]]
            or response[field_names[1]]
        )
    if not field_lengths_match:
        raise RuntimeError("GPU solver response field result lengths do not match the mesh")
    convergence = response.get("convergence")
    if not isinstance(convergence, dict):
        raise RuntimeError("GPU solver response convergence must be an object")
    iterations = convergence.get("iterations")
    residual = convergence.get("residual_l2")
    if (not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0
            or (residual is not None and not _finite_number(residual))):
        raise RuntimeError("GPU solver response has invalid convergence data")
    if status == "PASS" and returncode != 0:
        raise RuntimeError("GPU solver returned a failure code with a PASS response")
    if status == "FAIL" and returncode == 0:
        raise RuntimeError("GPU solver returned success with a FAIL response")
    if status == "PASS" and response["solve_status"] != "OK":
        raise RuntimeError("GPU solver PASS response does not report a successful solve")
    if status == "PASS" and response["postprocess_status"] not in ("OK", "NOT_REQUESTED"):
        raise RuntimeError("GPU solver PASS response has an unsuccessful postprocess status")
    if status == "PASS" and (response["error_identifier"] or response["error_message"]):
        raise RuntimeError("GPU solver PASS response contains an error")
    if status == "FAIL" and (not response["error_identifier"] or not response["error_message"]):
        raise RuntimeError("GPU solver FAIL response does not identify its error")
    if status == "FAIL":
        failure_status = (
            response["postprocess_status"]
            if response["solve_status"] == "OK"
            else response["solve_status"]
        )
        if failure_status in ("OK", "NOT_REQUESTED", "NOT_RUN"):
            raise RuntimeError("GPU solver FAIL response does not report a failed operation")
        if response["error_identifier"] != f"GPU_FEMM_{failure_status}":
            raise RuntimeError("GPU solver FAIL response has an inconsistent error identifier")
    return response


def command_solve(args: argparse.Namespace) -> int:
    solver = require_solver(args.solver)
    protocol = artifact_protocol(Path(args.artifact).resolve(strict=True))
    if protocol == GENERIC_PROTOCOL and (
        args.force_torque
        or args.force_group != -1
        or args.air_group != -1
        or args.airgap_radius_mm != 0.0
        or args.airgap_angles_deg
        or args.sliding_band_angle_deg != 0.0
    ):
        raise ValueError(
            "generic periodic/axisymmetric artifacts do not support force, air-gap, or sliding-band postprocessing"
        )
    request = make_request(
        Path(args.artifact),
        args.currents,
        args.force_group,
        args.air_group,
        args.airgap_radius_mm,
        args.airgap_angles_deg,
        args.sliding_band_angle_deg,
        args.force_torque,
        args.fields,
        protocol,
    )
    output = Path(args.output).resolve()
    if output in (Path(str(request["mesh_artifact_path"])), solver.resolve()):
        raise ValueError("response path must not overwrite the mesh artifact or GPU solver")
    output.parent.mkdir(parents=True, exist_ok=True)
    request_path: Path | None = None
    response_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        ) as stream:
            json.dump(request, stream, indent=2)
            stream.write("\n")
            request_path = Path(stream.name)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".tmp", dir=output.parent, delete=False
        ) as stream:
            response_path = Path(stream.name)
        returncode = run_solver(solver, "--solve", str(request_path), str(response_path))
        validate_response(
            response_path, str(request["mesh_artifact_sha256"]), returncode, protocol
        )
        _atomic_replace(response_path, output)
        response_path = None
        return returncode
    finally:
        if request_path is not None:
            request_path.unlink(missing_ok=True)
        if response_path is not None:
            response_path.unlink(missing_ok=True)


def command_validate(args: argparse.Namespace) -> int:
    return run_solver(require_solver(args.solver), "--mesh-artifact", args.artifact)


def command_capabilities(args: argparse.Namespace) -> int:
    return run_solver(require_solver(args.solver), "--capabilities")


def command_prepare(args: argparse.Namespace) -> int:
    from femm_gpu_prepare import prepare

    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    prepare(
        Path(args.input_fem),
        Path(args.output_artifact),
        Path(args.femm_root),
        Path(args.mesh_noop),
        args.timeout_s,
        args.overwrite,
    )
    print(f"Wrote immutable GPU FEMM artifact: {Path(args.output_artifact).resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="femm_gpu")
    parser.add_argument("--solver", help="path to the CUDA solver executable")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="compile a supported .fem model into a neutral mesh artifact"
    )
    prepare.add_argument("input_fem")
    prepare.add_argument("output_artifact")
    prepare.add_argument("--femm-root", required=True)
    prepare.add_argument("--mesh-noop", required=True)
    prepare.add_argument("--timeout-s", type=float, default=120.0)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(action=command_prepare)

    solve = commands.add_parser("solve", help="solve one DC magnetostatic operating point")
    solve.add_argument("artifact", help="standalone planar or generic magnetostatic artifact")
    solve.add_argument("output", help="response JSON path")
    solve.add_argument(
        "--currents",
        type=float,
        nargs="*",
        default=(),
        help="currents in artifact order; omit for a zero-circuit PM model",
    )
    solve.add_argument("--force-group", type=int, default=-1)
    solve.add_argument("--air-group", type=int, default=-1)
    solve.add_argument("--force-torque", action="store_true")
    solve.add_argument("--airgap-radius-mm", type=float, default=0.0)
    solve.add_argument("--airgap-angles-deg", type=float, nargs="*", default=())
    solve.add_argument("--sliding-band-angle-deg", type=float, default=0.0)
    solve.add_argument(
        "--fields", action="store_true",
        help="include nodal potential and element field components",
    )
    solve.set_defaults(action=command_solve)

    validate = commands.add_parser("validate", help="validate an immutable mesh artifact")
    validate.add_argument("artifact")
    validate.set_defaults(action=command_validate)

    capabilities = commands.add_parser("capabilities", help="print supported physics")
    capabilities.set_defaults(action=command_capabilities)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return args.action(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"femm_gpu: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

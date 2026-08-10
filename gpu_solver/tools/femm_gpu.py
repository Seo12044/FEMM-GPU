"""Small project-neutral command line frontend for the FEMM GPU solver.

The CUDA executable owns validation and numerical work.  This module only
builds the strict JSON request, fingerprints the immutable mesh artifact, and
starts the executable.  It intentionally has no MATLAB or Inventor dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


PROTOCOL = "gpu_femm_planar_dc_sample_v1"


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
) -> dict[str, object]:
    artifact = artifact.resolve(strict=True)
    return {
        "protocol": PROTOCOL,
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


def command_solve(args: argparse.Namespace) -> int:
    solver = require_solver(args.solver)
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
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    request_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        ) as stream:
            json.dump(request, stream, indent=2)
            stream.write("\n")
            request_path = Path(stream.name)
        return run_solver(solver, "--solve", str(request_path), str(output))
    finally:
        if request_path is not None:
            request_path.unlink(missing_ok=True)


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

    solve = commands.add_parser("solve", help="solve one planar DC operating point")
    solve.add_argument("artifact", help="gpu_femm_mesh_v1/v2 artifact")
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
    solve.add_argument("--fields", action="store_true", help="include nodal A and element B")
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

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "tools" / "femm_gpu.py"
SPEC = importlib.util.spec_from_file_location("femm_gpu", MODULE_PATH)
femm_gpu = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(femm_gpu)
SOLVER = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file()
    else None
)


class FemmGpuCliTests(unittest.TestCase):
    @staticmethod
    def _solve_args(artifact, output):
        return argparse.Namespace(
            solver="unused", artifact=str(artifact), output=str(output), currents=[],
            force_group=-1, air_group=-1, airgap_radius_mm=0.0,
            airgap_angles_deg=[], sliding_band_angle_deg=0.0,
            force_torque=False, fields=False,
        )

    @staticmethod
    def _response(artifact_sha, status="PASS"):
        passed = status == "PASS"
        return {
            "protocol": femm_gpu.PROTOCOL,
            "mesh_artifact_sha256": artifact_sha,
            "status": status,
            "solve_status": "OK" if passed else "NONLINEAR_SOLVE_NOT_CONVERGED",
            "postprocess_status": "NOT_REQUESTED" if passed else "NOT_RUN",
            "error_identifier": "" if passed else "GPU_FEMM_NONLINEAR_SOLVE_NOT_CONVERGED",
            "error_message": "" if passed else "injected failure",
            "Fx_N": None, "Fy_N": None, "torque_Nm": None,
            "actual_circuit_currents_A": [], "circuit_flux_linkage_Wb": [],
            "airgap_sample_angles_deg": [], "airgap_radial_flux_density_T": [],
            "mesh_node_count": 0, "mesh_element_count": 0,
            "field_solution_included": False,
            "node_A_Wb_per_m": [], "element_Bx_T": [], "element_By_T": [],
            "convergence": {"iterations": 0, "residual_l2": None},
        }

    def test_request_is_project_neutral_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "mesh.json"
            artifact.write_bytes(b"immutable artifact\n")
            request = femm_gpu.make_request(
                artifact,
                [1.0, -1.0],
                include_field_solution=True,
            )
        self.assertEqual(request["protocol"], "gpu_femm_planar_dc_sample_v1")
        self.assertEqual(request["circuit_currents_A"], [1.0, -1.0])
        self.assertTrue(request["include_field_solution"])
        self.assertNotIn("motor", json.dumps(request).lower())
        self.assertEqual(
            request["mesh_artifact_sha256"],
            hashlib.sha256(b"immutable artifact\n").hexdigest(),
        )

    def test_zero_circuit_request_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "mesh.json"
            artifact.write_text("{}", encoding="utf-8")
            request = femm_gpu.make_request(artifact, [])
        self.assertEqual(request["circuit_currents_A"], [])

    def test_solve_publishes_only_a_valid_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "mesh.json"
            output = root / "response.json"
            artifact.write_text("{}\n", encoding="utf-8")
            output.write_text("old response\n", encoding="utf-8")
            artifact_sha = femm_gpu.sha256_file(artifact)

            def fake_run(_solver, *arguments):
                Path(arguments[-1]).write_text(
                    json.dumps(self._response(artifact_sha)), encoding="utf-8"
                )
                return 0

            with mock.patch.object(femm_gpu, "require_solver", return_value=Path("fake")), \
                    mock.patch.object(femm_gpu, "run_solver", side_effect=fake_run):
                returncode = femm_gpu.command_solve(self._solve_args(artifact, output))

            self.assertEqual(returncode, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "PASS")
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_solve_keeps_previous_output_when_response_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "mesh.json"
            output = root / "response.json"
            artifact.write_text("{}\n", encoding="utf-8")
            output.write_text("old response\n", encoding="utf-8")

            def fake_run(_solver, *arguments):
                response = self._response("0" * 64)
                Path(arguments[-1]).write_text(json.dumps(response), encoding="utf-8")
                return 0

            with mock.patch.object(femm_gpu, "require_solver", return_value=Path("fake")), \
                    mock.patch.object(femm_gpu, "run_solver", side_effect=fake_run), \
                    self.assertRaisesRegex(RuntimeError, "mesh_artifact_sha256"):
                femm_gpu.command_solve(self._solve_args(artifact, output))

            self.assertEqual(output.read_text(encoding="utf-8"), "old response\n")
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_solve_preserves_a_valid_failure_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "mesh.json"
            output = root / "response.json"
            artifact.write_text("{}\n", encoding="utf-8")
            artifact_sha = femm_gpu.sha256_file(artifact)

            def fake_run(_solver, *arguments):
                Path(arguments[-1]).write_text(
                    json.dumps(self._response(artifact_sha, "FAIL")), encoding="utf-8"
                )
                return 1

            with mock.patch.object(femm_gpu, "require_solver", return_value=Path("fake")), \
                    mock.patch.object(femm_gpu, "run_solver", side_effect=fake_run):
                returncode = femm_gpu.command_solve(self._solve_args(artifact, output))

            self.assertEqual(returncode, 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "FAIL")

    def test_solve_refuses_to_overwrite_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "mesh.json"
            artifact.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(femm_gpu, "require_solver", return_value=Path("fake")), \
                    self.assertRaisesRegex(ValueError, "must not overwrite"):
                femm_gpu.command_solve(self._solve_args(artifact, artifact))

    @unittest.skipIf(SOLVER is None, "solver executable was not supplied")
    def test_capabilities_and_invalid_request_are_machine_readable(self):
        capabilities = subprocess.run(
            [str(SOLVER), "--capabilities"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(capabilities.stdout)["sample_protocol"],
            "gpu_femm_planar_dc_sample_v1",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "request.json"
            response = root / "response.json"
            request.write_text("{}\n", encoding="utf-8")
            run = subprocess.run(
                [str(SOLVER), "--solve", str(request), str(response)],
                check=False,
                capture_output=True,
                text=True,
            )
            document = json.loads(response.read_text(encoding="utf-8"))
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(document["status"], "FAIL")
        self.assertEqual(document["solve_status"], "INVALID_ARGUMENT")
        self.assertTrue(document["error_message"])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])

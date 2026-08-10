import argparse
import hashlib
import importlib.util
import json
import math
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
    def _write_artifact(path, schema=femm_gpu.LEGACY_ARTIFACT_SCHEMA):
        path.write_text(json.dumps({"schema_version": schema}) + "\n", encoding="utf-8")

    @staticmethod
    def _solve_args(artifact, output):
        return argparse.Namespace(
            solver="unused", artifact=str(artifact), output=str(output), currents=[],
            force_group=-1, air_group=-1, airgap_radius_mm=0.0,
            airgap_angles_deg=[], sliding_band_angle_deg=0.0,
            force_torque=False, fields=False,
        )

    @staticmethod
    def _response(artifact_sha, status="PASS", protocol=femm_gpu.PROTOCOL):
        passed = status == "PASS"
        return {
            "protocol": protocol,
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
            self._write_artifact(artifact)
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
            self._write_artifact(artifact)
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
            self._write_artifact(artifact)
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
            self._write_artifact(artifact)
            with mock.patch.object(femm_gpu, "require_solver", return_value=Path("fake")), \
                    self.assertRaisesRegex(ValueError, "must not overwrite"):
                femm_gpu.command_solve(self._solve_args(artifact, artifact))

    def test_generic_response_uses_explicit_cylindrical_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "mesh.json"
            self._write_artifact(artifact, femm_gpu.GENERIC_ARTIFACT_SCHEMA)
            artifact_sha = femm_gpu.sha256_file(artifact)
            response = self._response(
                artifact_sha, protocol=femm_gpu.GENERIC_PROTOCOL
            )
            response.pop("element_Bx_T")
            response.pop("element_By_T")
            response.pop("node_A_Wb_per_m")
            response.update({
                "problem_type": "axisymmetric",
                "field_components": ["Br", "Bz"],
                "node_potential_quantity": "poloidal_flux_function",
                "node_potential_unit": "Wb",
                "node_potential": [],
                "element_B_component_1_T": [],
                "element_B_component_2_T": [],
            })
            path = root / "response.json"
            path.write_text(json.dumps(response), encoding="utf-8")
            validated = femm_gpu.validate_response(
                path, artifact_sha, 0, femm_gpu.GENERIC_PROTOCOL
            )
        self.assertEqual(validated["field_components"], ["Br", "Bz"])

    def test_generic_solve_rejects_all_legacy_postprocess_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "mesh.json"
            output = root / "response.json"
            self._write_artifact(artifact, femm_gpu.GENERIC_ARTIFACT_SCHEMA)
            mutations = (
                ("force_group", 1),
                ("air_group", 2),
                ("airgap_radius_mm", 1.0),
            )
            for field, value in mutations:
                args = self._solve_args(artifact, output)
                setattr(args, field, value)
                with self.subTest(field=field), mock.patch.object(
                        femm_gpu, "require_solver", return_value=Path("fake")), \
                        self.assertRaisesRegex(ValueError, "do not support"):
                    femm_gpu.command_solve(args)

    def _assert_frozen_generic_reference(
            self, fixture_name, problem_type, max_potential_l2, max_flux_relative):
        root = Path(__file__).parent / "fixtures" / f"{fixture_name}_v1"
        artifact = root / f"{fixture_name}.gpu.json"
        answer_lines = (root / f"{fixture_name}.ans").read_text(
            encoding="utf-8").splitlines()
        solution_line = answer_lines.index("[Solution]")
        node_count = int(answer_lines[solution_line + 1])
        cpu_rows = [answer_lines[solution_line + 2 + index].split()
                    for index in range(node_count)]
        cpu_potential = {
            (round(float(row[0]), 12), round(float(row[1]), 12)): float(row[2])
            for row in cpu_rows
        }
        artifact_document = json.loads(artifact.read_text(encoding="utf-8"))
        nodes = artifact_document["resolved"]["nodes_mm"]
        summary = {}
        for line in (root / f"{fixture_name}.summary.txt").read_text(
                encoding="utf-8").splitlines():
            key, value = line.split("=", 1)
            summary[key] = float(value)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "response.json"
            args = self._solve_args(artifact, output)
            args.solver = str(SOLVER)
            args.currents = [summary["circuit_current_A"]]
            args.fields = True
            self.assertEqual(femm_gpu.command_solve(args), 0)
            response = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(response["problem_type"], problem_type)
        self.assertEqual(response["mesh_node_count"], node_count)
        pairs = [
            (cpu_potential[(round(x, 12), round(y, 12))], actual)
            for (x, y), actual in zip(nodes, response["node_potential"])
        ]
        potential_l2 = math.sqrt(
            sum((actual - expected) ** 2 for expected, actual in pairs)
            / sum(expected ** 2 for expected, _ in pairs)
        )
        gpu_flux = response["circuit_flux_linkage_Wb"][0]
        cpu_flux = summary["flux_linkage_Wb"]
        flux_relative = abs(gpu_flux - cpu_flux) / abs(cpu_flux)
        self.assertLessEqual(potential_l2, max_potential_l2)
        self.assertLessEqual(flux_relative, max_flux_relative)
        for constraint in artifact_document["resolved"]["node_constraints"]:
            left = response["node_potential"][constraint["node_a"]]
            right = response["node_potential"][constraint["node_b"]]
            sign = 1.0 if constraint["relation"] == "periodic" else -1.0
            self.assertAlmostEqual(right, sign * left, delta=1e-15)

    @unittest.skipIf(SOLVER is None, "solver executable was not supplied")
    def test_periodic_strip_matches_stock_femm(self):
        self._assert_frozen_generic_reference("periodic_strip", "planar", 1e-10, 1e-10)

    @unittest.skipIf(SOLVER is None, "solver executable was not supplied")
    def test_antiperiodic_strip_matches_stock_femm(self):
        self._assert_frozen_generic_reference(
            "antiperiodic_strip", "planar", 1e-10, 1e-10
        )

    @unittest.skipIf(SOLVER is None, "solver executable was not supplied")
    def test_axisymmetric_coil_matches_stock_femm(self):
        # The host operator uses conventional P1 A_phi interpolation while
        # stock FEMM uses its specialized r^2 basis. This limit freezes the
        # measured discretization difference rather than hiding it.
        self._assert_frozen_generic_reference(
            "axisymmetric_coil", "axisymmetric", 3e-3, 1e-3
        )

    @unittest.skipIf(SOLVER is None, "solver executable was not supplied")
    def test_capabilities_and_invalid_request_are_machine_readable(self):
        capabilities = subprocess.run(
            [str(SOLVER), "--capabilities"],
            check=True,
            capture_output=True,
            text=True,
        )
        document = json.loads(capabilities.stdout)
        self.assertEqual(document["sample_protocol"], "gpu_femm_planar_dc_sample_v1")
        self.assertIn("gpu_femm_magnetostatic_sample_v1", document["sample_protocols"])
        self.assertEqual(document["problem_types"], ["planar", "axisymmetric"])
        self.assertTrue(document["general_periodic_node_constraints"])
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

    @unittest.skipIf(SOLVER is None, "solver executable was not supplied")
    def test_direct_solver_never_overwrites_request_or_artifact(self):
        fixture = (Path(__file__).parent / "fixtures" / "periodic_strip_v1"
                   / "periodic_strip.gpu.json")
        summary = {}
        for line in (fixture.parent / "periodic_strip.summary.txt").read_text(
                encoding="utf-8").splitlines():
            key, value = line.split("=", 1)
            summary[key] = float(value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.json"
            artifact.write_bytes(fixture.read_bytes())
            request = root / "request.json"
            request.write_text(json.dumps(femm_gpu.make_request(
                artifact, [summary["circuit_current_A"]]
            )), encoding="utf-8")
            request_before = request.read_bytes()
            artifact_before = artifact.read_bytes()

            request_collision = subprocess.run(
                [str(SOLVER), "--solve", str(request), str(request)],
                check=False, capture_output=True, text=True,
            )
            artifact_collision = subprocess.run(
                [str(SOLVER), "--solve", str(request), str(artifact)],
                check=False, capture_output=True, text=True,
            )

            self.assertNotEqual(request_collision.returncode, 0)
            self.assertNotEqual(artifact_collision.returncode, 0)
            self.assertEqual(request.read_bytes(), request_before)
            self.assertEqual(artifact.read_bytes(), artifact_before)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])

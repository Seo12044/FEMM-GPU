import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "femm_gpu.py"
SPEC = importlib.util.spec_from_file_location("femm_gpu", MODULE_PATH)
femm_gpu = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(femm_gpu)
SOLVER = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None


class FemmGpuCliTests(unittest.TestCase):
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

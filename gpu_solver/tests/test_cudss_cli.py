"""Optional direct backend: neutral physics, host/device parity and CLI guards."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmark_saved_batch import compare
from test_femm_gpu_cli import femm_gpu

SOLVER = Path(sys.argv[1]).resolve()


def run(*args, env=None):
    return subprocess.run([str(SOLVER), *map(str, args)], capture_output=True, timeout=60, env=env)


class CudssCliTests(unittest.TestCase):
    def test_selection_and_unavailable_build(self):
        result = run("--capabilities")
        self.assertEqual(result.returncode, 0, result.stderr)
        cap = json.loads(result.stdout)
        self.assertEqual(cap["default_linear_solver"], "pcg")
        for args in (("--linear-solver",), ("--linear-solver", "cudss"),
                     ("--linear-solver", "unknown", "--capabilities"),
                     ("--linear-solver", "pcg", "--linear-solver", "pcg", "--capabilities")):
            self.assertEqual(run(*args).returncode, 2)
        if "cudss" not in cap["linear_solvers"]:
            result = run("--linear-solver", "cudss", "--capabilities")
            self.assertEqual(result.returncode, 2)
            self.assertIn(b"not enabled", result.stderr)
        else:
            self.assertEqual(run("--linear-solver", "cudss", "--self-test").returncode, 0)
            for prefix in (("--linear-solver", "cudss", "--preconditioner", "jacobi"),
                           ("--preconditioner", "block-jacobi", "--linear-solver", "cudss")):
                self.assertEqual(run(*prefix, "--capabilities").returncode, 2)

    def test_output_cannot_overwrite_executable(self):
        cap = json.loads(run("--capabilities").stdout)
        if "cudss" not in cap["linear_solvers"]:
            self.skipTest("cuDSS build disabled")
        before = hashlib.sha256(SOLVER.read_bytes()).hexdigest()
        result = run("--linear-solver", "cudss", "--solve", "missing.json", SOLVER)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"must not overwrite solver executable", result.stderr)
        self.assertEqual(hashlib.sha256(SOLVER.read_bytes()).hexdigest(), before)

    def test_unavailable_gpu_does_not_fall_back_to_cpu(self):
        cap = json.loads(run("--capabilities").stdout)
        if "cudss" not in cap["linear_solvers"]:
            self.skipTest("cuDSS build disabled")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = Path(__file__).parent/"fixtures/axisymmetric_coil_v1/axisymmetric_coil.gpu.json"
            request = femm_gpu.make_request(artifact, [12.0], protocol=femm_gpu.GENERIC_PROTOCOL)
            path, output = root/"request.json", root/"response.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            result = run("--linear-solver", "cudss", "--solve", path, output,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["error_identifier"],
                             "GPU_FEMM_GPU_UNAVAILABLE")

    def test_neutral_fixtures_and_default_pcg_unchanged(self):
        cap = json.loads(run("--capabilities").stdout)
        if "cudss" not in cap["linear_solvers"]:
            self.skipTest("cuDSS build disabled")
        fields = ("node_potential", "element_B_component_1_T", "element_B_component_2_T", "circuit_flux_linkage_Wb")
        for name in ("periodic_strip", "antiperiodic_strip", "axisymmetric_coil"):
            with self.subTest(fixture=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifact = Path(__file__).parent/"fixtures"/f"{name}_v1"/f"{name}.gpu.json"
                request = femm_gpu.make_request(artifact, [12.0], include_field_solution=True,
                                                protocol=femm_gpu.GENERIC_PROTOCOL)
                path = root/"request.json"
                path.write_text(json.dumps(request), encoding="utf-8")
                responses = []
                for mode, assembly in ((None, "device"), ("pcg", "device"),
                                       ("cudss", "device"), ("cudss", "host")):
                    output = root/f"{mode}_{assembly}.json"
                    prefix = [] if mode is None else ["--linear-solver", mode]
                    result = run(*prefix, f"--internal-solve-{assembly}", path, output)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    response = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(response["status"], "PASS")
                    responses.append({key: response[key] for key in fields})
                self.assertEqual(responses[0], responses[1])
                compare(responses[0], responses[2])
                compare(responses[2], responses[3])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])

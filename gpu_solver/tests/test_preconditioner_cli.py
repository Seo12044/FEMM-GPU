"""Public preconditioner selection, neutral fixtures, and output protection."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmark_saved_batch import compare
from test_femm_gpu_cli import femm_gpu

SOLVER = Path(sys.argv[1]).resolve()


def run(*arguments):
    return subprocess.run([str(SOLVER), *map(str, arguments)],
                          capture_output=True, timeout=30)


class PreconditionerCliTests(unittest.TestCase):
    def test_capabilities_and_invalid_selection(self):
        result = run("--capabilities")
        self.assertEqual(result.returncode, 0)
        capabilities = json.loads(result.stdout)
        self.assertEqual(capabilities["default_preconditioner"], "jacobi")
        self.assertEqual(capabilities["preconditioners"], ["jacobi", "block-jacobi"])
        for arguments in (("--preconditioner",), ("--preconditioner", "block-jacobi"),
                          ("--preconditioner", "unknown", "--capabilities")):
            self.assertEqual(run(*arguments).returncode, 2)

    def test_prefix_does_not_bypass_executable_output_protection(self):
        before = hashlib.sha256(SOLVER.read_bytes()).hexdigest()
        result = run("--preconditioner", "block-jacobi", "--solve", "missing.json", SOLVER)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"must not overwrite solver executable", result.stderr)
        self.assertEqual(hashlib.sha256(SOLVER.read_bytes()).hexdigest(), before)

    def test_planar_periodic_antiperiodic_axisymmetric_host_device_parity(self):
        fields = ("node_potential", "element_B_component_1_T",
                  "element_B_component_2_T", "circuit_flux_linkage_Wb")
        for name in ("periodic_strip", "antiperiodic_strip", "axisymmetric_coil"):
            with self.subTest(fixture=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifact = Path(__file__).parent / "fixtures" / f"{name}_v1" / f"{name}.gpu.json"
                request = femm_gpu.make_request(artifact, [12.0], include_field_solution=True,
                                                protocol=femm_gpu.GENERIC_PROTOCOL)
                request_path = root / "request.json"
                request_path.write_text(json.dumps(request), encoding="utf-8")
                reference = None
                for mode, backend in ((None, "device"), ("jacobi", "device"),
                                      ("block-jacobi", "device"), ("block-jacobi", "host")):
                    output = root / f"{mode}_{backend}.json"
                    prefix = [] if mode is None else ["--preconditioner", mode]
                    result = run(*prefix, f"--internal-solve-{backend}", request_path, output)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    response = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(response["status"], "PASS")
                    observables = {key: response[key] for key in fields}
                    if reference is None:
                        reference = observables
                    elif mode == "jacobi":
                        self.assertEqual(reference, observables)
                    else:
                        compare(reference, observables)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])

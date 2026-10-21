"""Exercise the benchmark's parity gate without launching CUDA."""
import math
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from benchmark_saved_batch import compare, main


class ParityGateTests(unittest.TestCase):
    def test_timing_and_iteration_diagnostics_are_not_observables(self):
        compare({"timing": {"s": 1}, "cache": {}, "convergence": {"iterations": 10}, "B": [0.1]},
                {"timing": {"s": 2}, "cache": {}, "convergence": {"iterations": 12}, "B": [0.1]})

    def test_roundoff_is_measured(self):
        difference = compare({"torque": 0.1}, {"torque": 0.1 + 1e-12})
        self.assertGreater(difference[".torque"], 0)

    def test_corrupt_observable_fails(self):
        with self.assertRaises(AssertionError):
            compare({"B": [1.0]}, {"B": [1.01]})

    def test_collecting_mismatches_does_not_relax_the_gate(self):
        errors = []
        difference = compare({"B": [1.0, 2.0]}, {"B": [1.01, 2.02]}, violations=errors)
        self.assertEqual(len(errors), 2)
        self.assertAlmostEqual(difference[".B"], 0.02)

    def test_hash_and_status_must_match(self):
        for key in ("status", "mesh_artifact_sha256"):
            with self.assertRaises(AssertionError):
                compare({key: "original"}, {key: "changed"})

    def test_integer_counts_are_exact(self):
        with self.assertRaises(AssertionError):
            compare({"node_count": 1000000000}, {"node_count": 1000000001})

    def test_nonfinite_fails(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(AssertionError):
                compare({"B": 0.0}, {"B": value})

    def test_missing_field_or_array_element_fails(self):
        for value in ({}, {"B": []}):
            with self.assertRaises(AssertionError):
                compare({"B": [1.0]}, value)

    def test_boolean_is_not_a_numeric_result(self):
        for reference in (0.0, 0):
            with self.assertRaises(AssertionError):
                compare({"B": reference}, {"B": False})

    def test_failed_or_interrupted_run_does_not_remain_running(self):
        failures = [
            subprocess.CompletedProcess([], 1, b"", b"solver failed"),
            subprocess.TimeoutExpired("solver", 1),
            KeyboardInterrupt(),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                exe = root / "solver.exe"
                exe.write_bytes(b"not executed")
                request = root / "request.json"
                request.write_text("{}", encoding="utf-8")
                output = root / "evidence"
                argv = ["benchmark", "--baseline", str(exe), "--candidate", str(exe),
                        "--request", str(request), "--output", str(output), "--rounds", "1"]
                with patch("sys.argv", argv), patch("subprocess.run") as run:
                    if isinstance(failure, BaseException):
                        run.side_effect = failure
                    else:
                        run.return_value = failure
                    with self.assertRaises((AssertionError, subprocess.TimeoutExpired, KeyboardInterrupt)):
                        main()
                report = json.loads((output / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "FAILED")
                self.assertTrue(report["error"])
                self.assertEqual(len(report["requests"]), 1)
                self.assertEqual(report["requests"][0]["runs"], [])

    def test_successful_run_records_both_binaries_once(self, modes=False, mismatch=False):
        response = {"status": "PASS", "timing": {}, "items": [{"response": {
            "status": "PASS", "convergence": {"iterations": 2, "residual_l2": 0.0},
            "B": [0.1]}}]}

        def solve(command, **kwargs):
            if modes:
                self.assertEqual(command[1], "--preconditioner")
                self.assertIn(command[2], ("jacobi", "block-jacobi"))
            else:
                self.assertEqual(command[1], "--motor-batch-profile")
            if mismatch and "candidate" in str(command[-1]):
                response["items"][0]["response"]["B"] = [0.2]
            Path(command[-1]).write_text(json.dumps(response), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, b"", b"")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exe = root / "solver.exe"
            exe.write_bytes(b"not executed")
            request = root / "request.json"
            request.write_text("{}", encoding="utf-8")
            output = root / "evidence"
            argv = ["benchmark", "--baseline", str(exe), "--candidate", str(exe),
                    "--request", str(request), "--output", str(output), "--rounds", "1"]
            if modes:
                argv += ["--baseline-preconditioner", "jacobi", "--candidate-preconditioner", "block-jacobi"]
            if mismatch:
                argv += ["--continue-on-parity-failure"]
            with patch("sys.argv", argv), patch("subprocess.run", side_effect=solve):
                if mismatch:
                    with self.assertRaises(AssertionError):
                        main()
                else:
                    main()
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAILED" if mismatch else "PASS")
            self.assertEqual(len(report["requests"]), 1)
            self.assertEqual([r["binary"] for r in report["requests"][0]["runs"]],
                             ["baseline", "candidate"])
            self.assertEqual(report["executables"]["candidate"]["preconditioner"],
                             "block-jacobi" if modes else None)
            self.assertEqual(report["requests"][0]["runs"][-1]["parity"], "FAIL" if mismatch else "PASS")

    def test_same_binary_preconditioner_comparison_records_modes(self):
        self.test_successful_run_records_both_binaries_once(modes=True)

    def test_collected_parity_failure_retains_results_but_fails_report(self):
        self.test_successful_run_records_both_binaries_once(modes=True, mismatch=True)


if __name__ == "__main__":
    unittest.main()

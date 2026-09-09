"""Exercise the benchmark's parity gate without launching CUDA."""
import math
import unittest

from benchmark_saved_batch import compare


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


if __name__ == "__main__":
    unittest.main()

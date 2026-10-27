"""Worker transport, immutable input, item isolation and legacy result parity."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmark_saved_batch import compare, sha256
from benchmark_streaming_batch import run_stream

SOLVER = Path(sys.argv[1]).resolve()
FIXTURE = Path(__file__).parent/"fixtures/stream_worker_v1/mesh.json"


def request(path=FIXTURE):
    sample = dict(protocol="gpu_femm_motor_sample_v1", mesh_artifact_path=str(path),
                  mesh_artifact_sha256=sha256(path), base_motor_fem_sha256="b"*64,
                  source_fem_sha256="a"*64, circuit_currents_A=[2, -2],
                  selected_group_number=20, air_group_number=50, airgap_radius_mm=0,
                  airgap_angles_deg=[], rotor_angle_deg=3, displacement_mm=[0.1, 0])
    return dict(protocol="gpu_femm_motor_batch_v1", max_items_per_chunk=4,
                items=[dict(task_id="same_mesh_current", request=sample)])


def worker(jobs):
    result = subprocess.run([str(SOLVER), "--motor-worker"],
                            input="".join(json.dumps(job)+"\n" for job in jobs),
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    return [json.loads(line) for line in result.stdout.splitlines()]


class MotorWorkerTests(unittest.TestCase):
    def test_prefetched_mask_selection_failure_is_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for index, selected in enumerate((20, 21, 20)):
                value = request()
                value["items"][0]["request"]["selected_group_number"] = selected
                path = root/f"input_{index}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                jobs.append(dict(session="a"*64, sequence=index+1, request_path=str(path),
                                 response_path=str(root/f"response_{index}.json"),
                                 request_sha256=sha256(path)))
            ack = worker(jobs)
            self.assertEqual([a["exit_code"] for a in ack], [0, 1, 0])
            responses = [json.loads((root/f"response_{i}.json").read_text()) for i in range(3)]
            self.assertEqual(responses[1]["status"], "FAIL")
            compare(responses[0], responses[2], strict=True)
            for job in jobs:
                self.assertEqual(sha256(job["request_path"]), job["request_sha256"])

    def test_queue_preserves_all_fields_convergence_and_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"input.json"
            path.write_text(json.dumps(request()), encoding="utf-8")
            baseline = root/"baseline.json"
            result = subprocess.run([str(SOLVER), "--motor-batch-profile", str(path), str(baseline)],
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr+baseline.read_bytes())
            outputs = [root/f"response_{i}.json" for i in range(4)]
            ack = run_stream(SOLVER, "pcg", [path]*4, outputs, 30)
            reference = json.loads(baseline.read_text())
            for path in outputs:
                compare(reference, json.loads(path.read_text()), strict=True)
            self.assertEqual([a["sequence"] for a in ack], [1, 2, 3, 4])

    def test_rejected_job_does_not_discard_following_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"input.json";path.write_text(json.dumps(request()))
            jobs = [dict(session="a"*64, sequence=i+1, request_path=str(path),
                         response_path=str(root/f"response_{i}.json"), request_sha256=sha256(path))
                    for i in range(2)]
            jobs[0]["request_sha256"] = "0"*64
            ack = worker(jobs)
            self.assertEqual([a["exit_code"] for a in ack], [1, 0])
            self.assertFalse((root/"response_0.json").exists())
            self.assertIn("SHA-256 mismatch", ack[0]["error"])

    def test_session_sequence_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"input.json";path.write_text(json.dumps(request()))
            before = path.read_bytes()
            envelope = dict(session="a"*64, sequence=1, request_path=str(path),
                            response_path=str(path), request_sha256=sha256(path))
            self.assertIn("unsafe", worker([envelope])[0]["error"])
            self.assertEqual(path.read_bytes(), before)
            envelope["response_path"] = str(root/"output.json")
            second = {**envelope, "sequence": 1}
            self.assertIn("sequence mismatch", worker([envelope, second])[1]["error"])
            second.update(session="b"*64, sequence=2)
            self.assertIn("session", worker([envelope, second])[1]["error"])

    def test_transaction_memory_bound_and_artifact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = request()
            model["items"] = [dict(task_id=f"item_{i}", request=model["items"][0]["request"])
                              for i in range(33)]
            path = root/"input.json";path.write_text(json.dumps(model))
            envelope = dict(session="a"*64, sequence=1, request_path=str(path),
                            response_path=str(root/"response.json"), request_sha256=sha256(path))
            self.assertIn("exceeds 32", worker([envelope])[0]["error"])
            model = request();model["items"][0]["request"]["mesh_artifact_sha256"] = "0"*64
            path.write_text(json.dumps(model));envelope["request_sha256"] = sha256(path)
            self.assertEqual(worker([envelope])[0]["exit_code"], 1)
            response = json.loads((root/"response.json").read_text())
            self.assertNotEqual(response["items"][0]["response"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])

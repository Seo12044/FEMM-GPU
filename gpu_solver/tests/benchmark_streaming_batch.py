"""A/B saved requests: one-shot processes versus a bounded persistent worker.

No mesh generation. Retains every response, transport acknowledgement and
binary/request digest. Strict parity ignores timing only. Alternates run order.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from benchmark_saved_batch import compare, sha256


def run_stream(executable, solver, requests, outputs, timeout):
    session = sha256(executable)
    with subprocess.Popen([str(executable), "--linear-solver", solver, "--motor-worker"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, encoding="utf-8") as process:
        jobs = [dict(session=session, sequence=i+1, request_path=str(request),
                     response_path=str(output), request_sha256=sha256(request))
                for i, (request, output) in enumerate(zip(requests, outputs))]
        # communicate drains pipes, sends EOF, and enforces one overall deadline.
        # Native input preparation is bounded independently of this wire queue.
        try:
            stdout, stderr = process.communicate(
                "".join(json.dumps(job)+"\n" for job in jobs), timeout=timeout)
        except BaseException:
            process.kill()
            process.communicate()
            raise
        assert process.returncode == 0, stderr
    acknowledgements = [json.loads(line) for line in stdout.splitlines()]
    assert len(acknowledgements) == len(jobs), (stdout, stderr)
    for job, ack in zip(jobs, acknowledgements):
        for key in ("session", "sequence", "request_sha256"):
            assert ack[key] == job[key], (key, ack)
        assert ack["exit_code"] == 0 and not ack["error"], ack
    return acknowledgements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--linear-solver", choices=("pcg", "cudss"), default="pcg")
    parser.add_argument("--request", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    requests = [path.resolve() for path in args.request]
    report = dict(status="RUNNING", runs=[], requests=[dict(path=str(p), sha256=sha256(p))
                  for p in requests], binaries={name: dict(path=str(path), sha256=sha256(path))
                  for name, path in (("baseline", args.baseline), ("candidate", args.candidate))})
    reference = None
    try:
        for repeat in range(args.rounds):
            order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
            for label in order:
                outputs = [args.output/f"{label}_{repeat}_{i:03d}.json" for i in range(len(requests))]
                start = time.perf_counter()
                if label == "candidate":
                    ack = run_stream(args.candidate, args.linear_solver, requests, outputs, args.timeout)
                else:
                    ack = []
                    for request, output in zip(requests, outputs):
                        result = subprocess.run([str(args.baseline), "--linear-solver", args.linear_solver,
                                                 "--motor-batch-profile", str(request), str(output)],
                                                capture_output=True, timeout=args.timeout)
                        assert result.returncode == 0, result.stderr
                elapsed = time.perf_counter()-start
                responses = [json.loads(p.read_text(encoding="utf-8")) for p in outputs]
                for response in responses:
                    assert response["status"] == "PASS", response
                    for item in response["items"]:
                        value = item["response"]["convergence"]["residual_l2"]
                        assert math.isfinite(value) and 0 <= value < 1e-6
                if reference is None:
                    reference = responses
                for before, after in zip(reference, responses):
                    compare(before, after, strict=True)
                report["runs"].append(dict(mode=label, round=repeat, wall_seconds=elapsed,
                                           acknowledgements=ack, strict_parity="PASS"))
                print(f"{label} round {repeat}: {elapsed:.3f} s, strict parity PASS", flush=True)
        for item in report["requests"]:
            assert sha256(item["path"]) == item["sha256"]
        for item in report["binaries"].values():
            assert sha256(item["path"]) == item["sha256"]
        report["median_wall_seconds"] = {label: statistics.median(
            r["wall_seconds"] for r in report["runs"] if r["mode"] == label)
            for label in ("baseline", "candidate")}
        report["speedup"] = report["median_wall_seconds"]["baseline"]/report["median_wall_seconds"]["candidate"]
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        (args.output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

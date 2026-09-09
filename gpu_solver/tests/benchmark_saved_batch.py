"""Compare two binaries on saved batch requests; never prepare or alter a mesh.

The output directory must be new. Each subprocess has a bounded timeout. Runs
alternate A/B order and retain every response plus executable/request hashes.
This is a manual benchmark, not an automatic long-running unit test.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(reference, actual, path="", differences=None):
    if differences is None:
        differences = {}
    if isinstance(reference, dict):
        assert isinstance(actual, dict), f"{path}: expected object"
        assert reference.keys() == actual.keys(), f"{path}: response fields changed"
        for key in reference:
            if key in ("timing", "cache", "convergence"):
                continue
            compare(reference[key], actual[key], f"{path}.{key}", differences)
    elif isinstance(reference, list):
        assert isinstance(actual, list), f"{path}: expected array"
        assert len(reference) == len(actual), f"{path}: array length changed"
        for a, b in zip(reference, actual):
            compare(a, b, path, differences)
    elif isinstance(reference, float):
        assert isinstance(actual, (float, int)) and not isinstance(actual, bool), path
        assert math.isfinite(reference) and math.isfinite(actual), path
        delta = abs(reference - actual)
        differences[path] = max(differences.get(path, 0.0), delta)
        # A numerical comparison gate, never a production convergence setting.
        assert math.isclose(reference, actual, rel_tol=1e-8, abs_tol=1e-9), (
            f"{path}: {reference} != {actual} (absolute difference {delta})"
        )
    else:
        assert type(reference) is type(actual), f"{path}: value type changed"
        assert reference == actual, f"{path}: {reference!r} != {actual!r}"
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--request", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    assert args.rounds > 0 and args.timeout > 0
    args.output.mkdir(parents=True, exist_ok=False)
    executables = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    report = {"status": "RUNNING", "requested_count": len(args.request),
              "executables": {k: {"path": str(v), "sha256": sha256(v)}
                              for k, v in executables.items()}, "requests": []}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for index, request in enumerate(args.request):
        request = request.resolve()
        evidence = {"path": str(request), "sha256": sha256(request), "runs": []}
        reference = None
        maxima = {}
        for repeat in range(args.rounds):
            order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
            for label in order:
                output = args.output / f"request_{index:02d}_{label}_{repeat}.json"
                start = time.perf_counter()
                completed = subprocess.run(
                    [str(executables[label]), "--motor-batch-profile", str(request), str(output)],
                    capture_output=True, timeout=args.timeout,
                )
                elapsed = time.perf_counter() - start
                assert completed.returncode == 0, completed.stderr.decode(errors="replace")
                response = json.loads(output.read_text(encoding="utf-8"))
                assert response["status"] == "PASS"
                for item in response["items"]:
                    value = item["response"]
                    assert value["status"] == "PASS", value
                    residual = value["convergence"]["residual_l2"]
                    assert math.isfinite(residual) and 0 <= residual < 1e-6
                if reference is None:
                    reference = response
                compare(reference, response, differences=maxima)
                record = {"binary": label, "round": repeat, "wall_seconds": elapsed,
                          "timing": response["timing"],
                          "iterations": [x["response"]["convergence"]["iterations"]
                                         for x in response["items"]]}
                evidence["runs"].append(record)
                print(f"{request.name} {label} {repeat}: {elapsed:.3f} s; parity PASS", flush=True)
        assert sha256(request) == evidence["sha256"], "request changed during benchmark"
        evidence["max_absolute_differences"] = maxima
        evidence["median_wall_seconds"] = {
            label: statistics.median(x["wall_seconds"] for x in evidence["runs"] if x["binary"] == label)
            for label in executables}
        evidence["speedup"] = (evidence["median_wall_seconds"]["baseline"]
                               / evidence["median_wall_seconds"]["candidate"])
        report["requests"].append(evidence)
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for label, exe in executables.items():
        assert sha256(exe) == report["executables"][label]["sha256"], "binary changed during benchmark"
    report["status"] = "PASS"
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

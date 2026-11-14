"""Search a synthetic corpus, with optional KS1 Top-K ranking."""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

from topk_reference import INPUT, K, N, OUTPUT, reference

DOCUMENTS = (
    "rmsnorm scales each row using its root mean square",
    "operator fusion saves a residual read from ddr memory",
    "topk chooses the highest scores and returns their positions",
    "an attention mask hides future tokens during inference",
    "bf16 stores each floating point value in two bytes",
    "vector cache holds temporary data close to the npu",
    "a baseline provides a reference for performance comparison",
    "warmup runs prepare the device before latency measurement",
)
EVALUATION = (
    ("rmsnorm root square", {0}),
    ("fusion residual ddr", {1}),
    ("topk highest positions", {2}),
    ("attention mask tokens", {3}),
    ("bf16 bytes", {4}),
    ("vector cache temporary", {5}),
    ("baseline performance", {6}),
    ("warmup latency", {7}),
    ("device", {7}),
    ("astronomy telescope", set()),
)


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


TOKENS = tuple(map(tokenize, DOCUMENTS))


def search(query, device):
    start = perf_counter()
    terms = tokenize(query)
    scores = [len(terms & words) for words in TOKENS]
    raw = INPUT.pack(*scores, *([-1] * (N - len(scores))))
    expected_values, expected_ids = reference(raw)
    if device is None:
        values, indices = expected_values, expected_ids
    else:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "input.bin", Path(tmp) / "output.bin"
            src.write_bytes(raw)
            subprocess.run(
                ["dort", "calbin", "run", "-i", str(device),
                 "-f", str(Path(__file__).resolve().parent / "cal"), "-m", "topk",
                 "--in-tensor", "in_raw", "--out-tensor", "out_raw",
                 "--raw-in", str(src), "--raw-out", str(dst)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            output = OUTPUT.unpack(dst.read_bytes())
        values, indices = output[:K], output[K:]
        if (values, indices) != (expected_values, expected_ids):
            raise ValueError("NPU ranking differs from the CPU reference")
    hits = [(i, int(score), DOCUMENTS[i]) for score, i in zip(values, indices)
            if score > 0 and 0 <= i < len(DOCUMENTS)]
    return hits, (perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    backend = "CPU reference" if args.device is None else "NPU ranking"
    try:
        if args.query is not None:
            hits, elapsed = search(args.query, args.device)
            result = {"backend": backend, "hits": hits, "end_to_end_ms": elapsed}
        else:
            trials = []
            for query, relevant in EVALUATION:
                hits, elapsed = search(query, args.device)
                ids = [hit[0] for hit in hits]
                passed = (bool(ids) and ids[0] in relevant) if relevant else not ids
                trials.append({"query": query, "ids": ids, "passed": passed,
                               "end_to_end_ms": elapsed})
            result = {"backend": backend, "passed": sum(t["passed"] for t in trials),
                      "total": len(trials), "trials": trials}
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stdout or str(error)) from error
    print(json.dumps(result, indent=2))
    return int(args.evaluate and result["passed"] != result["total"])


if __name__ == "__main__":
    raise SystemExit(main())

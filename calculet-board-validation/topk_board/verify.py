"""Generate Top-K inputs and verify the NPU's scores and indices."""
import argparse
import random
import struct
from pathlib import Path

N = 256
K = 16
INPUT = struct.Struct(f"<{N}f")
OUTPUT = struct.Struct(f"<{K}f{K}i")
LOWEST = -float.fromhex("0x1.fffffep127")


def reference(raw):
    scores = INPUT.unpack(raw)
    indices = sorted(range(N), key=lambda i: (-scores[i], i))[:K]
    return tuple(scores[i] for i in indices), tuple(indices)


def check(raw, output):
    expected_values, expected_ids = reference(raw)
    actual = OUTPUT.unpack(output)
    if actual[K:] != expected_ids or actual[:K] != expected_values:
        for rank, (score, index) in enumerate(zip(expected_values, expected_ids)):
            got = (actual[rank], actual[K + rank])
            if got != (score, index):
                raise ValueError(f"rank {rank}: expected {(score, index)}, got {got}")


def fixtures():
    shuffled = list(range(-128, 128))
    random.Random(42).shuffle(shuffled)
    close_scores = [1.0 + (i % 32) * 2**-23 for i in shuffled]
    return {
        "shuffled": shuffled,
        "negative": [-x for x in range(1, N + 1)],
        "ties": [i % 7 - 4 for i in range(N)],
        "zero": [-0.0, 0.0] * (N // 2),
        "minimum": [LOWEST] * N,
        "close_fp32": close_scores,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("generate").add_argument("directory", type=Path)
    verify = sub.add_parser("check")
    verify.add_argument("input", type=Path)
    verify.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.command == "generate":
        args.directory.mkdir(parents=True, exist_ok=True)
        for name, scores in fixtures().items():
            (args.directory / f"{name}.bin").write_bytes(INPUT.pack(*scores))
    else:
        try:
            check(args.input.read_bytes(), args.output.read_bytes())
        except (ValueError, struct.error) as error:
            raise SystemExit(str(error)) from error
        print(f"{args.input.stem}: all {K} scores and indices match")

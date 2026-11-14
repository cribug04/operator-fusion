"""Compare the on-board final RMSNorm and whole-model logits."""
import argparse
import math
import struct
from pathlib import Path

WIDTH = 512
SIZE = WIDTH * 2
INPUT_OFFSET = 24576
OUTPUT_OFFSET = 25600
WEIGHT_OFFSET = 6135808


def read_bf16(data, offset=0):
    words = struct.unpack_from(f"<{WIDTH}H", data, offset)
    return tuple(struct.unpack("<f", struct.pack("<I", word << 16))[0] for word in words)


def compare(name, actual, expected, relative, absolute):
    errors = [abs(a - b) for a, b in zip(actual, expected)]
    count = sum(not (math.isfinite(a) and math.isfinite(b)
                    and abs(a - b) <= absolute + relative * abs(b))
                for a, b in zip(actual, expected))
    print(f"{name}: {count}/{WIDTH} outside tolerance, max absolute error {max(errors):.6g}")
    return count


def check(dump, weights, reference):
    if len(dump) < OUTPUT_OFFSET + SIZE:
        raise ValueError("Need the full output buffer, including both RMSNorm probe slots")
    x = read_bf16(dump, INPUT_OFFSET)
    w = read_bf16(weights, WEIGHT_OFFSET)
    denominator = math.sqrt(math.fsum(v * v for v in x) / WIDTH + 1e-6)
    expected = tuple(v * gain / denominator for v, gain in zip(x, w))
    rms_errors = compare("Final RMSNorm", read_bf16(dump, OUTPUT_OFFSET), expected, 0.035, 0.002)
    logits_errors = compare("Model logits", read_bf16(dump), read_bf16(reference), 0.05, 0.05)
    return rms_errors + logits_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    raise SystemExit(int(bool(check(args.dump.read_bytes(),
                                    (base / "param_blk0.bin").read_bytes(),
                                    (base / "reference_logits.bin").read_bytes()))))

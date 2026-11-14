"""Generate a deterministic BF16 fixture and check raw device output."""
import argparse
import math
from pathlib import Path
import struct

ROWS, HIDDEN = 4, 4096
COUNT = ROWS * HIDDEN
EPS = 1e-6


def bf16(values):
    def bits(value):
        word = struct.unpack("<I", struct.pack("<f", value))[0]
        return (word + 0x7FFF + ((word >> 16) & 1)) >> 16

    return struct.pack(f"<{len(values)}H", *(bits(x) for x in values))


def unpack(data):
    return [struct.unpack("<f", b"\x00\x00" + item)[0]
            for item in (data[i:i + 2] for i in range(0, len(data), 2))]


def reference(raw):
    assert len(raw) == 2 * (2 * COUNT + HIDDEN)
    data = unpack(raw)
    weight = data[2 * COUNT:]
    residual = unpack(bf16([a + b for a, b in zip(data[:COUNT], data[COUNT:2 * COUNT])]))
    output = []
    for offset in range(0, COUNT, HIDDEN):
        row = residual[offset:offset + HIDDEN]
        rms = math.sqrt(math.fsum(s * s for s in row) / HIDDEN + EPS)
        output.extend(s * w / rms for s, w in zip(row, weight))
    return bf16(residual), output


def generate(folder):
    pattern = [(i % 31 - 15) / 16 for i in range(HIDDEN)]
    a = pattern * 2 + [0.0] * HIDDEN + [v / 1024 for v in pattern]
    b = [-v for v in pattern] + [(i % 17 - 8) / 32 for i in range(HIDDEN)] + [0.0] * (2 * HIDDEN)
    w = [(i % 23 - 11) / 8 for i in range(HIDDEN)]
    (folder / "input.bin").write_bytes(bf16(a + b + w))


def check(raw, outputs):
    residual, expected = reference(raw)
    results = []
    for path in outputs:
        data = path.read_bytes()
        if len(data) != 4 * COUNT:
            raise SystemExit(f"{path}: expected {4 * COUNT} bytes, got {len(data)}")
        if data[:2 * COUNT] != residual:
            raise SystemExit(f"{path}: BF16 residual differs from reference")
        actual = unpack(data[2 * COUNT:])
        errors = [abs(x - y) for x, y in zip(actual, expected)]
        failures = sum(not math.isclose(x, y, rel_tol=0.035, abs_tol=0.002)
                       for x, y in zip(actual, expected))
        if failures:
            raise SystemExit(f"{path}: {failures} output elements exceed tolerance")
        print(f"{path}: residual exact; output within tolerance; max abs error {max(errors):.6g}")
        results.append(data)
    if len(results) == 2 and results[0] != results[1]:
        raise SystemExit("Fused and split output bytes differ")
    if len(results) == 2:
        print("Fused and split outputs are byte-identical")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("generate").add_argument("folder", type=Path)
    verify = commands.add_parser("check")
    verify.add_argument("input", type=Path)
    verify.add_argument("outputs", type=Path, nargs="+")
    args = cli.parse_args()
    if args.command == "generate":
        generate(args.folder)
    else:
        check(args.input.read_bytes(), args.outputs)

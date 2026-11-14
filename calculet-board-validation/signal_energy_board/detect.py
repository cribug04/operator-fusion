import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

AMPLITUDES = [0, .125, .25, 1, .125, .25, 0, .125,
              .25, .125, 2, .25, 0, .25, .125, .875]
EXPECTED = [3, 10, 15]


def main():
    parser = argparse.ArgumentParser(description="Detect strong vibration windows")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    start = perf_counter()
    samples = [a * (-1)**i for a in AMPLITUDES for i in range(64)]
    reference = [math.fsum(x*x for x in samples[i:i+64])/64
                 for i in range(0, len(samples), 64)]
    values = reference
    if args.device is not None:
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "input.bin"
            dst = Path(directory) / "output.bin"
            src.write_bytes(struct.pack("<1024f", *samples))
            cal = Path(__file__).resolve().parent / "cal"
            subprocess.run(["dort", "calbin", "run", "-i", str(args.device),
                            "-f", str(cal), "-m", "signal_energy",
                            "--in-tensor", "in_raw", "--out-tensor", "out_raw",
                            "--raw-in", str(src), "--raw-out", str(dst)],
                           check=True, stdout=subprocess.DEVNULL)
            values = struct.unpack("<16f", dst.read_bytes())

    flagged = [i for i, v in enumerate(values) if v > 0.5]
    elapsed = (perf_counter() - start) * 1000
    if flagged != EXPECTED or any(
        not math.isclose(v, ref, rel_tol=1e-5, abs_tol=1e-7)
        for v, ref in zip(values, reference)
    ):
        raise SystemExit(f"Validation failed: flagged windows {flagged}, values {values}")
    print(json.dumps({"backend": "CPU" if args.device is None else "NPU",
                      "mean_square": values, "flagged_windows": flagged,
                      "elapsed_ms": elapsed}, indent=2))


if __name__ == "__main__":
    main()

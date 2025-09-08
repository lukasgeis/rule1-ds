import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("-b", "--binary", required=True, type=Path)
    parser.add_argument("-n", "--num_threads", type=int, default=4)
    return parser.parse_args()


def process_file(args):
    input_file, output_dir, binary_path = args
    output_file = output_dir / input_file.stem

    print(f"Run on {input_file}")
    with open(output_file, "w") as out:
        proc = subprocess.Popen(
            [binary_path, "--file", input_file],
            stderr=out,
        )
        proc.wait()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = list(args.input.iterdir())
    job_args = [(f, args.output, args.binary) for f in files]

    with ProcessPoolExecutor(max_workers=args.num_threads) as executor:
        executor.map(process_file, job_args)


if __name__ == "__main__":
    main()

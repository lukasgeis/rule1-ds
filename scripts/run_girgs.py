import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("-b", "--binary", required=True, type=Path)
    parser.add_argument("-g", "--girgs", required=True, type=Path)
    parser.add_argument("-n", "--num_threads", type=int, default=4)
    return parser.parse_args()


def process_degree(args):
    deg, input_file, output_dir, binary_path, girgs_path = args

    input_file = output_dir / f"temp_deg{deg}.txt"
    output_file = output_dir / f"girgs_deg{deg}"

    print(f"Generating input for Degree={deg}")
    girgs = subprocess.Popen(
        [
            girgs_path,
            "-n", "100000",
            "-deg", str(deg),
            "-a", "1",
            "-t", "0",
            "-edge", "1",
            "-file", input_file,
        ]
    )
    girgs.wait()

    print(f"Running for Degree={deg}")
    with open(output_file, "w") as out:
        proc = subprocess.Popen(
            [binary_path, "--file", input_file, "--girgs"],
            stderr=out,
        )
        proc.wait()

    print(f"Deleting Temp-File for Degree={deg}")
    input_file.unlink(missing_ok=True)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    degrees = range(10, 2001)

    job_args = [(deg, args.output, args.binary, args.girgs) for deg in degrees]

    with ProcessPoolExecutor(max_workers=args.num_threads) as executor:
        executor.map(process_degree, job_args)


if __name__ == "__main__":
    main()

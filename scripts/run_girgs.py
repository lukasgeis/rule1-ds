import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("-b", "--binary", required=True, type=Path)
    parser.add_argument("-g", "--girgs", required=True, type=Path)
    parser.add_argument("-n", "--num_threads", type=int, default=4)
    return parser.parse_args()


def process_degree(args):
    deg, output_dir, binary_path, girgs_path = args

    input_file = output_dir / f"temp_deg{deg}"
    output_file = output_dir / f"girgs_deg{deg}"

    input_file = input_file.resolve()
    girgs_path = girgs_path.resolve()

    girgs_file = input_file.with_suffix(".txt")

    print(f"Generating input for Degree={deg}")
    cmd = [
        str(girgs_path),
        "-n", "100000",
        "-deg", str(deg),
        "-a", "1",
        "-t", "0",
        "-edge", "1",
        "-file", str(input_file),
    ]

    girgs = subprocess.run(
        cmd, check=True, capture_output=True, text=True
    )
    
    print(f"Running for Degree={deg}")
    with open(output_file, "w") as out:
        proc = subprocess.Popen(
            [binary_path, "--file", girgs_file, "--girgs"],
            stderr=out,
        )
        proc.wait()

    print(f"Deleting Temp-File for Degree={deg}")
    girgs_file.unlink(missing_ok=True)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    degrees = range(10, 2000)

    job_args = [(deg, args.output, args.binary, args.girgs) for deg in degrees]

    with ProcessPoolExecutor(max_workers=args.num_threads) as executor:
        executor.map(process_degree, job_args)


if __name__ == "__main__":
    main()

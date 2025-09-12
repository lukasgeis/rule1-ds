#!/usr/bin/env python3
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import random
from multiprocessing import Pool
from tqdm import tqdm
import tempfile
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("-b", "--binary", required=True, type=Path)
    parser.add_argument("-g", "--girgs", required=True, type=Path)
    parser.add_argument("-n", "--num_threads", type=int, default=4)
    return parser.parse_args()


def process_degree(args):
    deg, output_dir, binary_path, girgs_path = args

    output_file = output_dir / f"girgs_deg{deg}.log"

    try:
        # test whether file exists and is complete ("maxrss" is contained in the last line)
        with open(output_file, "r") as existing:
            for line in existing:
                if "maxrss" in line:
                    return
    except:
        pass

    girgs_path = girgs_path.resolve()

    with tempfile.TemporaryDirectory(delete=True, dir=output_dir) as tmpdir:
        tmp_file = Path(tmpdir) / "graph"
        cmd = [
            str(girgs_path),
            "-n", "100000",
            "-deg", str(deg),
            "-a", "1",
            "-t", "0",
            "-edge", "1",
            "-file", str(tmp_file),
        ]

        subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )

        input_file = tmp_file.with_suffix(".txt")
        assert input_file.exists
        
        with open(output_file, "w") as out:
            proc = subprocess.Popen(
                [binary_path, "--file", input_file, "--girgs"],
                stderr=out,
            )
            _, _, rusage = os.wait4(proc.pid, 0)
            out.write(f"maxrss={rusage.ru_maxrss*1024} bytes\n")



def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    degrees = range(10, 2000)

    job_args = [(deg, args.output, args.binary, args.girgs) for deg in degrees]
    random.shuffle(job_args)

    with Pool(args.num_threads) as pool:
        list(tqdm(pool.imap_unordered(process_degree, job_args, chunksize=1), total=len(job_args)))


if __name__ == "__main__":
    main()

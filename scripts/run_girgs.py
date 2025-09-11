#!/usr/bin/env python3
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import random
from multiprocessing import Pool
from tqdm import tqdm
import tempfile

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("-b", "--binary", required=True, type=Path)
    parser.add_argument("-g", "--girgs", required=True, type=Path)
    parser.add_argument("-n", "--num_threads", type=int, default=4)
    parser.add_argument("-s", "--skip-existing", action="store_true")
    return parser.parse_args()


def process_degree(args):
    deg, output_dir, binary_path, girgs_path = args

    output_file = output_dir / f"girgs_deg{deg}.log"

    girgs_path = girgs_path.resolve()

    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        tmp_file.close()
        cmd = [
            str(girgs_path),
            "-n", "100000",
            "-deg", str(deg),
            "-a", "1",
            "-t", "0",
            "-edge", "1",
            "-file", tmp_file.name,
        ]

        subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
        
        with open(output_file, "w") as out:
            proc = subprocess.Popen(
                [binary_path, 
                 "--file", Path(tmp_file.name).with_suffix(".txt"),
                   "--girgs"],
                stderr=out,
            )
            proc.wait()



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

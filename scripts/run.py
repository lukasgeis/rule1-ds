#!/usr/bin/env python3
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("-b", "--binary", required=True, type=Path)
    parser.add_argument("-n", "--num_threads", type=int, default=4)
    parser.add_argument("-s", "--skip_existing", action="store_true")
    return parser.parse_args()


def process_file(args):
    input_file, output_dir, binary_path, skip_existing = args
    name = input_file.stem
    if name.endswith(".gr"):
        name = name[:-3]

    output_file = output_dir / f"{name}.log"
    if skip_existing and output_file.exists():
        return

    with open(output_file, "w") as out:
        proc = subprocess.Popen(
            [binary_path, "--file", input_file],
            stderr=out,
        )
        proc.wait()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = list(args.input.glob("*.gr")) + list(args.input.glob("*.gr.bz2"))
    random.shuffle(files)
    job_args = [(f, args.output, args.binary, args.skip_existing) for f in files]

    with Pool(args.num_threads) as pool:
        list(tqdm(pool.imap_unordered(process_file, job_args, chunksize=1), total=len(job_args)))

if __name__ == "__main__":
    main()

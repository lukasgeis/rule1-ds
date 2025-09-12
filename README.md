# DominatingSet Rule1 Reduction

This repository may be used to reproduce the experiments described in 

  "Revisiting a Successful Reduction Rule for Dominating Set" by Geis, Leonhardt, Meintrup, Meyer, Penschuck and Retschmeier, ALENEX 2026.

It is a based on our more extensive implementation in the [PaceYourself](https://github.com/manpen/pace25/tree/master) solver.
If you want to use the code beyond reproduction of the manuscript, the [solver repository] (https://github.com/manpen/pace25/tree/master) is most likely the better starting point.

## Instructions
### Download dataset(s)
The paper's experiments rely on a large number of graph instances totalling ~80 GB in compressed size.
We provide a "base" set of instances that should suffice to obtain most results (~4000 instances below 150 MB).
Additionally we provide the remaining 120 instances above 150 MB to reproduce the full set of experiments.
   - base dataset (15 GB):             https://ae.cs.uni-frankfurt.de/public_files/raw/alenex26_base.tar   extract to input/base
   - OPTIONAL large instances (65 GB): https://ae.cs.uni-frankfurt.de/public_files/raw/alenex26_large.tar  extract to input/large

HINT: You can simply run `download_base.sh` (and optionally also `download_large.sh`) to download and extract the files. The script requires `wget` and `tar`.

COMMENT TO REVIEWERS: The datasets will be uploaded to Zenodo including the respective attribution and license files (see paper for source).

### Run experiments
 - Adjust the number of threads in the beginning of `run.sh` (see comments there)
 - Execute `run.sh`. 
   After building the docker image two (three if large instances were downloaded) experimental campaign are started successively.
   Each will have their own progress bar.
 - Runtime roughly:
        3h 
    +  50h / NUM_THREADS_BASE
    + 100h / NUM_THREADS_LARGE 
    + 100h / NUM_THREADS_GIRGS

REMARK: A measurement is skipped if the respective log file exists and is complete.
Hence it should be possible to restart `./run.sh` without losing progress or needing to cleanup partial logs.
We still recommend to backup the `output` folder before a restart ;)



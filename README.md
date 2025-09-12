# DominatingSet Rule1 Reduction

This repository may be used to reproduce the experiments described in 

  "Revisiting a Successful Reduction Rule for Dominating Set" by Geis, Leonhardt, Meintrup, Meyer, Penschuck and Retschmeier, ALENEX 2026.

It is a minimalistic fork of our more extensive implementation in the [PaceYourself](https://github.com/manpen/pace25/tree/master) solver.
If you want to use the code the [solver repository] (https://github.com/manpen/pace25/tree/master) is most likely the better starting point.

## Instructions
The experiments in the paper rely on a large number of graph instances totalling ~80 GB in size.
We provide a "base" set of instances to get started (instances below 150 MB) and optional additional instances (above 150 MB) to reproduce the full set of experiments.

 - Before beginning download the following datasets and extract all files directly into the folder `input`;
   the individual instance files should stay bz2 compressed:
   - base dataset (15 GB):             https://ae.cs.uni-frankfurt.de/public_files/raw/alenex26_base.tar   extract to input/base
   - OPTIONAL large instances (65 GB): https://ae.cs.uni-frankfurt.de/public_files/raw/alenex26_large.tar  extract to input/large
 - Adjust the number of threads in the beginning of `run.sh` (see comments there)
 - Execute `run.sh`. Runtime roughly:
    
    + 100h / NUM_THREADS_GIRGS




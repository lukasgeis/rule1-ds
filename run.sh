#!/bin/bash

####################################
# these numbers work on a 32 core machine (excl. hyperthreading) with 128GB RAM
NUM_THREADS_BASE=12     #recommended: min(#cores, RAMSIZE/10GB); set to "4" if the machine has only 64GB RAM
NUM_THREADS_LARGE=4     #recommended: min(#cores, RAMSIZE/32GB); set to "1" if the machine has only 64GB RAM
NUM_THREADS_GIRG=24     #recommended: min(#cores, RAMSIZE/6GB); requires ~1GB*NUM_THREADS_GIRG of disk storage
####################################

GREEN='\033[0;32m'
NC='\033[0m' # No Color

set -x
set -e

# change into root directory of repository
BASE=`cd $(dirname $0); pwd`
cd $BASE

# build docker container
TAG=alenex26-60-ae
docker build -t $TAG .

docker_run() {
    docker run \
        -it --rm \
        -v $BASE/input:/input:ro \
        -v $BASE/output:/output:rw \
        $TAG \
        $@
}

########

echo -e "${GREEN}Running experiments on the base data set ...${NC}"
docker_run /app/scripts/run.py       -o /output/data  -b /app/rule1 -i /input/base -n $NUM_THREADS_BASE

echo -e "${GREEN}Running experiments on the large data set (if existing) ...${NC}"
docker_run /app/scripts/run.py       -o /output/data  -b /app/rule1 -i /input/large -n $NUM_THREADS_LARGE

echo -e "${GREEN}Running experiments on random girg graphs ...${NC}"
docker_run /app/scripts/run_girgs.py -o /output/girgs -b /app/rule1 -g /app/girgs/build/genhrg -n $NUM_THREADS_GIRG

echo -e "${GREEN}Plotting results ...${NC}"
docker_run /app/scripts/plot.py       /output/data  /output/plots/ | tee output/plots/data_stats.log
docker_run /app/scripts/plot_girgs.py /output/girgs /output/plots/ | tee output/plots/girgs_stats.log

echo -e "${GREEN}All done! Results are in the output/plots folder.${NC}"

#!/bin/bash
NUM_THREADS_BASE=24     #recommended: min(#cores, RAMSIZE/)
NUM_THREADS_LARGE=24    #recommended: min(#cores, RAMSIZE/)
NUM_THREADS_GIRG=24     #recommended: min(#cores, RAMSIZE/6GB); requires ~1GB*NUM_THREADS_GIRG of disk storage

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

# run experiments on data set
#docker_run /app/scripts/run.py       -o /output/data  -b /app/rule1 -i /input -n $NUM_THREADS_BASE
#docker_run /app/scripts/run.py       -o /output/data  -b /app/rule1 -i /input -n $NUM_THREADS_BASE

########

docker_run /app/scripts/run_girgs.py -o /output/girgs -b /app/rule1 -g /app/girgs/build/genhrg -n $NUM_THREADS_GIRG
docker_run /app/scripts/plot_girgs.py /output/girgs /output/plots/ | tee output/plots/girgs_stats.log



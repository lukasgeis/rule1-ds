#!/bin/bash
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


# run experiments on data set
docker_run /app/scripts/run.py       -o /output/data  -b /app/rule1 -i /input 
docker_run /app/scripts/run_girgs.py -o /output/girgs -b /app/rule1 -g /app/girgs/build/genhrg -n 32



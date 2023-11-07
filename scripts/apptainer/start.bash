#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

source $SCRIPT_DIR/../setup/envfile

apptainer instance start \
    --env-file $SCRIPT_DIR/../setup/envfile \
    --nv \
    -B /nas/:/mnt:ro \
    $1 \
    inst

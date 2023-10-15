#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

singularity exec \
    --env-file $SCRIPT_DIR/../setup/envfile \
    instance://inst \
    bash \
    $1 \
    ${@:2}

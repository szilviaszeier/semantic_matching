#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

apptainer exec \
    --env-file $SCRIPT_DIR/../setup/envfile \
    instance://inst \
    bash \
    $1 \
    ${@:2}

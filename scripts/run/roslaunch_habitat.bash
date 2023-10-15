#!/bin/bash

source /opt/ros/melodic/setup.bash
source /opt/conda/etc/profile.d/conda.sh
conda activate habitat

echo "SEMANTIC_ROOT: $SEMANTIC_ROOT"
source $SEMANTIC_ROOT/kimera_ws/devel/setup.bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

roslaunch \
    habitat_interface \
    simulation.launch \
    $@

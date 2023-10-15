#!/bin/bash

source /opt/ros/melodic/setup.bash
source /opt/conda/etc/profile.d/conda.sh
conda activate habitat

source $SEMANTIC_ROOT/kimera_ws/devel/setup.bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

roslaunch \
    kimera_interface \
    ucoslam_k4a_semantic.launch \
    $@
#!/bin/bash

source /opt/ros/melodic/setup.bash
source /opt/conda/etc/profile.d/conda.sh
conda activate detectron2

source $SEMANTIC_ROOT/kimera_ws/devel/setup.bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

roslaunch \
    detectron2_ros \
    detectron2_inference.launch \
    $@
#!/bin/bash

# Remove /usr/lib/x86_64-linux-gnu due to bad link with Numba
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH/usr/lib/x86_64-linux-gnu:}

source /opt/ros/melodic/setup.bash
source /opt/conda/etc/profile.d/conda.sh
conda activate superpixel

source $SEMANTIC_ROOT/kimera_ws/devel/setup.bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

roslaunch \
    superpixel_segmentation \
    superpixel_segmentation.launch \
    $@
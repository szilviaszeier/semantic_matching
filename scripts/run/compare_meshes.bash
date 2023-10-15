#!/bin/bash

source /opt/ros/melodic/setup.bash
source /opt/conda/etc/profile.d/conda.sh
conda activate superpixel

source $SEMANTIC_ROOT/kimera_ws/devel/setup.bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

python $SEMANTIC_ROOT/kimera_ws/src/superpixel_segmentation/src/compare_voxel_grids.py $@
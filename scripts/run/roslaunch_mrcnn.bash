#!/bin/bash

source /opt/ros/melodic/setup.bash
source /opt/conda/etc/profile.d/conda.sh
source $SEMANTIC_ROOT/kimera_ws/devel/setup.bash
conda activate semantic_mapping

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

#roslaunch \
#    mask_rcnn_ros \
#    mask_rcnn_node.launch \
#    $@

#NOTE: The -s is really important to not pick up a wrong version of TensorFlow.
TF_FORCE_GPU_ALLOW_GROWTH=true python -s ${SCRIPT_DIR}/../../kimera_ws/src/mask_rcnn_ros/src/mask_rcnn_node.py \
    --visualization False \
    --dataset_name sunrgbd \
    --model_path $SEMANTIC_ROOT/kimera_ws/src/mask_rcnn_ros/models/mask_rcnn_sun.h5\
    --input_rgb_topic $1

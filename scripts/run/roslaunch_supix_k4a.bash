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
    image_topic:=/rgb/image_raw \
    depth_topic:=/depth_to_rgb/image_raw \
    camera_config:=/home/szilvi/semantic_mapping/kimera_ws/src/superpixel_segmentation/config/calib_k4a.yml \
    robot_hostname:=szilvi_msi \
    sensor_frame:=szilvi_msi_camera_base \
    parent_frame:=szilvi_msi_odometry_frame \
    depth_scale:=1000 \
    $@
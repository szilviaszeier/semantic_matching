#!/bin/bash
source /opt/ros/melodic/setup.bash
source $SEMANTIC_ROOT/kimera_ws/devel/setup.bash

roslaunch -p ${ROS_MASTER_URI##*:} $@
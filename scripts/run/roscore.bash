#!/bin/bash

source /opt/ros/melodic/setup.bash
roscore -p ${ROS_MASTER_URI##*:}
#!/bin/bash

source /opt/ros/melodic/setup.bash

rosservice call /kimera_semantics_node/generate_mesh

rosparam get -p /
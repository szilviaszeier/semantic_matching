#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda deactivate

source /opt/ros/melodic/setup.bash

cd $SEMANTIC_ROOT/kimera_ws

catkin build $1
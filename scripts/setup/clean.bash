#!/bin/bash

source /opt/ros/melodic/setup.bash

cd $SEMANTIC_ROOT/kimera_ws

catkin clean --deinit --yes

rm ./src/.rosinstall
rm ./src/.rosinstall.bak
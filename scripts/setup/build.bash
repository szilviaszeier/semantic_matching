#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
ENV_PATH=$(conda info --envs | grep -Po 'semantic_mapping\K.*' | sed 's: ::g')

conda deactivate

source /opt/ros/melodic/setup.bash

cd $SEMANTIC_ROOT/kimera_ws

catkin init
catkin config --source-space $SEMANTIC_ROOT/kimera_ws/src/

(cd $SEMANTIC_ROOT/kimera_ws/src && wstool init \
    && wstool merge kimera_interface/install/kimera_interface.rosinstall \
    && wstool update)

(cd $SEMANTIC_ROOT/kimera_ws/src/vision_opencv \
    && git checkout fc782bb \
    && git apply $SEMANTIC_ROOT/kimera_ws/src/vision_opencv_fc782bb.patch)

catkin config --extend /opt/ros/melodic
catkin config --merge-devel
catkin config --cmake-args \
    -DCMAKE_CXX_STANDARD=14 \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$ENV_PATH/bin/python3 \
    -DPYTHON_INCLUDE_DIR=$ENV_PATH/include/python3.7m \
    -DPYTHON_LIBRARY=$ENV_PATH/lib/libpython3.7m.so

# blacklist python2 packages
catkin config -a --blacklist \
    minkindr_python \
    numpy_eigen \
    iclcv_catkin \
    iclcv_segmentation

# blacklist ZED, Kinect, and RealSense packages because no install
catkin config -a --blacklist \
    azure_kinect_ros_driver \
    zed_nodelets \
    zed_wrapper \
    zed_interfaces \
    realsense2_camera

catkin build opencv3_catkin

catkin build


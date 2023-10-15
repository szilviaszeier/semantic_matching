#!/bin/bash

source /opt/ros/melodic/setup.bash
rosbag reindex $1
rosbag play $1
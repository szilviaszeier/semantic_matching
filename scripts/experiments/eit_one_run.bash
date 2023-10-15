#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# first   ($1): segmentation topic and method 
#    (/habitat/semantics/image_raw /semantics/semantic_image /semantics/superpixel)
# second ($2): rosbag to play
# if supix:
#    third ($3): baseline, camera, 3d_supix, 3d_neighbor, dbscan
#    fourth  ($4): upper view mesh path
# if detectron: 
#    third ($3): model to use (Mask_R_CNN_7 Transfiner_7)
# fifth ($5): result mesh save path relative to kimera_ws/src/kimera_interface/mesh_results

$SCRIPT_DIR/../singularity/start.bash /home/szilvi/image_sandbox
screen -dmS roscore $SCRIPT_DIR/../singularity/exec.bash $SCRIPT_DIR/../run/roscore.bash

method=$1
bag=$2
shift 2

screen -dmS kimera $SCRIPT_DIR/../singularity/exec.bash $SCRIPT_DIR/../run/roslaunch_kimera_k4a.bash left_cam_segmentation_topic:=$method bag_file:=$bag play_bag:=True should_use_sim_time:=True metric_semantic_reconstruction:=true robot_hostname:=szilvi_msi launch_rviz:=false semantic_label_file_path:=/home/szilvi/semantic_mapping/kimera_ws/src/kimera_interface/cfg/categories4detectron2_20.csv

case $method in
    /semantics/superpixel)
        echo "Using supix, launching node..."
        screen -dmS supix $SCRIPT_DIR/../singularity/exec.bash $SCRIPT_DIR/../run/roslaunch_supix_k4a.bash --wait method:=$1 mesh_path:=$2
        shift 2
        ;;
    /semantics/semantic_image)
        echo "Using detectron2, launching node..."
        screen -dmS detectron2 $SCRIPT_DIR/../singularity/exec.bash $SCRIPT_DIR/../run/roslaunch_detectron2.bash --wait model:=$1
        shift 1
        ;;
    *)
        echo "Method not implemented"
        ;;
esac

sleep 5s

$SCRIPT_DIR/../singularity/exec.bash $SCRIPT_DIR/../run/save_mesh.bash

echo "Moving saved mesh to $1"
mv $SCRIPT_DIR/../../kimera_ws/src/kimera_interface/mesh_results/semantic_mesh.ply $1

$SCRIPT_DIR/../singularity/stop.bash
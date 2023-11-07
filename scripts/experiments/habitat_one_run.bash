#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# first   ($1): segmentation topic and method 
#    (/habitat/semantics/image_raw /semantics/semantic_image /semantics/superpixel)
# if supix:
#    second ($2): baseline (True) or cameraplane (False)
#    third  ($3): upper view mesh path
# if detectron: 
#    second ($2): model to use (Mask_R_CNN_7 Transfiner_7)
# fourth  ($4): habitat mesh path
# fifth   ($5): sensor height
# sixth   ($6): corresponding trajectory path
# seventh ($7): depth noise
# eigth   ($8): result mesh save path relative to kimera_ws/src/kimera_interface/mesh_results

$SCRIPT_DIR/../apptainer/start.bash /home/szilvi/image_sandbox
screen -dmS roscore $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roscore.bash

method=$1
shift 1

if [ $method = rgb ]
then
    screen -dmS kimera $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roslaunch_kimera_habitat.bash --wait play_bag:=False launch_rviz:=False metric_semantic_reconstruction:=False
else
    screen -dmS kimera $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roslaunch_kimera_habitat.bash --wait left_cam_segmentation_topic:=$method play_bag:=False launch_rviz:=False semantic_label_file_path:=/home/szilvi/semantic_mapping/kimera_ws/src/kimera_interface/cfg/categories4detectron2_20.csv

    case $method in
        /semantics/superpixel)
            echo "Using supix, launching node..."
            screen -dmS supix $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roslaunch_supix.bash --wait method:=$1 mesh_path:=$2
            shift 2
            ;;
        /semantics/semantic_image)
            echo "Using detectron2, launching node..."
            screen -dmS detectron2 $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roslaunch_detectron2.bash --wait model:=$1
            shift 1
            ;;
        /habitat/semantics/image_raw)
            echo "Using gt."
            ;;
        *)
            echo "Method not implemented"
            ;;
    esac
fi
sleep 5s

$SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roslaunch_habitat.bash --wait mesh_path:=$1 sensor_height:=$2 output_agent_pose_name:=$3 depth_noise_multiplier:=$4 replay_mode:=True

shift 4

$SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/save_mesh.bash

echo "Moving saved mesh to $1"
mv $SCRIPT_DIR/../../kimera_ws/src/kimera_interface/mesh_results/semantic_mesh.ply $1

$SCRIPT_DIR/../apptainer/stop.bash
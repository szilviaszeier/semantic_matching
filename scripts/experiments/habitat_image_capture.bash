#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

#HABITAT_DATA_PATH=$SCRIPT_DIR/../../kimera_ws/src/habitat_interface/data/states_lower
HABITAT_DATA_PATH=/mnt/home/baffy-1000022/work/semantic_mapping/nipg30/kimera_ws/src/habitat_interface/data
#MESH_RESULTS_PATH=$SCRIPT_DIR/../../kimera_ws/src/kimera_interface/mesh_results/habitat_height
MESH_RESULTS_PATH=$SCRIPT_DIR/../../kimera_ws/src/kimera_interface/mesh_results/eit

ROOMS=(room_2) # room_2 office_3 frl_apartment_4
TRAJECTORIES=(0 1) #  1
HEIGHTS=(0.2) # 0.2 0.8 1.2 1.8 0.25 0.3
method=3d_neighbor
model=Transfiner_16

for room in "${ROOMS[@]}"
do
    for trajectory in "${TRAJECTORIES[@]}"
    do
        for height in "${HEIGHTS[@]}"
        do
            echo "Room: $room -- method: $method -- height: $height -- trajectory: $trajectory"

            $SCRIPT_DIR/../apptainer/start.bash /home/szilvi/image_sandbox

            screen -dmS roscore $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roscore.bash

            sleep 3s

            # screen -dmS image_capture $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/image_capture_node.bash\
            #     --save_folder /home/szilvi/semantic_mapping/kimera_ws/src/superpixel_segmentation/assets/eit/lower\
            #     --image_topic /rgb/image_raw\
            #     --depth_topic /depth_to_rgb/image_raw\
            #     --semantics_topic /rgb/image_raw\
            #     --camera_config /home/szilvi/semantic_mapping/kimera_ws/src/superpixel_segmentation/config/calib_k4a.yml
                #/mnt/project_data/semantic_mapping/lower_training/${room}/2/${height}
            screen -dmS image_capture $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/image_capture_node.bash --save_folder /home/szilvi/semantic_mapping/kimera_ws/src/superpixel_segmentation/assets/${room}/${trajectory}

            echo "Using supix, launching node..."
            screen -dmS supix $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roslaunch_supix.bash --wait method:=$method mesh_path:=/home/szilvi/semantic_mapping/kimera_ws/src/kimera_interface/mesh_results/habitat_height/${room}_1.8_${trajectory}_gt.ply
            #screen -dmS supix $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roslaunch_supix_k4a.bash --wait method:=$method mesh_path:=$MESH_RESULTS_PATH/transf16/semantic_mesh_human_k4a_eit_clc_rec_1_transf16.ply

            echo "Using detectron2, launching node..."
            screen -dmS detectron2 $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roslaunch_detectron2.bash --wait model:=$model
            #screen -dmS detectron2 $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roslaunch_detectron2.bash --wait model:=$model image_topic:=/rgb/image_raw

            sleep 5s

            $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roslaunch_habitat.bash --wait mesh_path:=/mnt/database/replica/$room/habitat/mesh_semantic.ply sensor_height:=$height output_agent_pose_name:=$HABITAT_DATA_PATH/$room/low_traj.npy depth_noise_multiplier:=0 replay_mode:=True target_fps:=2
            #$SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/roslaunch_kimera_k4a.bash --wait play_bag:=True should_use_sim_time:=True bag_file:=/mnt/home/szilvi-1000038/hmi-rehab/data/rosbags/eit_clc/eit_clc_record_1.bag metric_semantic_reconstruction:=true robot_hostname:=szilvi_msi launch_rviz:=false semantic_label_file_path:=/home/szilvi/semantic_mapping/kimera_ws/src/kimera_interface/cfg/categories4detectron2_20.csv

            $SCRIPT_DIR/../apptainer/stop.bash
        done
    done
done

#./scripts/apptainer/exec.bash ./scripts/run/image_capture_node.bash --save_folder /home/szilvi/semantic_mapping/kimera_ws/src/superpixel_segmentation/assets/eit/lower --image_topic /rgb/image_raw --depth_topic /depth_to_rgb/image_raw --semantics_topic /rgb/image_raw --camera_config /home/szilvi/semantic_mapping/kimera_ws/src/superpixel_segmentation/config/calib_k4a.yml
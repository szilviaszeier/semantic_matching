#!/bin/bash

# Script to generate the 3D semantic reconstructions at different heights with different 2D semantic segmentation
# methods that can be later used to verify lower heights result in worse semantic reconstructions

# The trajectories can be found in /nas/project_data/semantic_mapping/height_evaluation_trajectories.zip
# Download the zip and extract it under semantic_mapping/kimera_ws/src/habitat_interface/data/

# Adjust MESH_RESULTS_PATH or make sure the given path exists

# Make sure to adjust the container path in relevan files (habitat_one_run.bash) on line starting with:
# $SCRIPT_DIR/../apptainer/start.bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

HABITAT_DATA_PATH=/home/szilvi/semantic_mapping_old/kimera_ws/src/habitat_interface/data/states_lower # states_lower contains trajectories, downloadable from the groups server
MESH_RESULTS_PATH=$SCRIPT_DIR/../../kimera_ws/src/kimera_interface/mesh_results/habitat_height

ROOMS=(frl_apartment_4 room_2 office_3) #
HEIGHTS=(0.2) # 0.2 0.8 1.2 1.8
METHODS=(transfiner habitat) # habitat mrcnn transfiner
TRAJECTORIES=(0 1)
SUPIX=(baseline camera 3d_supix 3d_neighbor dbscan)

for room in "${ROOMS[@]}"
do
    for trajectory in "${TRAJECTORIES[@]}"
    do
        # SEGMENTATION
        for height in "${HEIGHTS[@]}"
        do
            for method in "${METHODS[@]}"
            do
                echo "Room: $room -- method: $method -- height: $height -- trajectory: $trajectory"
                case $method in
                    habitat)
                        path=$MESH_RESULTS_PATH/${room}_${height}_${trajectory}_gt.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /habitat/semantics/image_raw\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $height\
                                $HABITAT_DATA_PATH/${room}_floor_a_0.2_${trajectory}.npy\
                                0.0\
                                $path
                        fi
                        ;;
                    mrcnn)
                        path=$MESH_RESULTS_PATH/${room}_${height}_${trajectory}_mrcnn.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                Mask_R_CNN_16\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $height\
                                $HABITAT_DATA_PATH/${room}_floor_a_0.2_${trajectory}.npy\
                                0.0\
                                $path
                        fi
                        ;;
                    transfiner)
                        path=$MESH_RESULTS_PATH/${room}_${height}_${trajectory}_transfiner.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                Transfiner_16\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $height\
                                $HABITAT_DATA_PATH/${room}_floor_a_0.2_${trajectory}.npy\
                                0.0\
                                $path
                        fi
                        ;;
                esac

            done # METHODS

            for supix in "${SUPIX[@]}"
            do
                echo "Room: $room -- method: $method -- height: $height -- trajectory: $trajectory -- supix: $supix"
                path=$MESH_RESULTS_PATH/${room}_${height}_${supix}_${trajectory}_transfiner.ply
                if [ ! -f $path ] 
                then
                    $SCRIPT_DIR/habitat_one_run.bash\
                        /semantics/superpixel\
                        $supix\
                        $MESH_RESULTS_PATH/${room}_1.8_${trajectory}_transfiner.ply\
                        /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                        $height\
                        $HABITAT_DATA_PATH/${room}_floor_a_0.2_${trajectory}.npy\
                        0\
                        $path
                fi
            done # SUPIX
        done # HEIGHTS
    done # TRAJECTORIES
done # ROOMS
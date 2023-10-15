#!/bin/bash

# CLC
High (rec1) {mrcnn, transfiner}
High (rec2) {mrcnn, transfiner}
Low {rec3, rec4, rec5} {mrcnn, transfiner}
{rec3, rec4, rec5} {mrcnn, transfiner} supix 



SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

EIT_RECORDINGS_PATH=/mnt/home/szilvi-1000038/hmi-rehab/data/rosbags/
MESH_RESULTS_PATH=$SCRIPT_DIR/../../kimera_ws/src/kimera_interface/mesh_results/eit

ROOMS=(eit_clc eit_meeting)
HIGH_TRAJECTORIES=(1 2) 
LOW_TRAJECTORIES=(3) # 4 5
METHODS=(mrcnn transfiner)
SUPIX=(3d_supix 3d_neighbor dbscan) # baseline camera


for room in "${ROOMS[@]}"
do
    for high_trajectory in "${HIGH_TRAJECTORIES[@]}"
    do

        # SEGMENTATION

            for method in "${METHODS[@]}"
            do
                echo "Room: $room -- method: $method -- high"
                case $method in
                    mrcnn)
                        path=$MESH_RESULTS_PATH/${room}/high_${method}_${high_trajectory}.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/eit_one_run.bash\
                                /semantics/semantic_image\
                                $EIT_RECORDINGS_PATH/$room/${room}_record_${high_trajectory}.bag\
                                Mask_R_CNN_16\
                                $path
                        fi
                        ;;
                    transfiner)
                        path=$MESH_RESULTS_PATH/${room}/high_${method}_${high_trajectory}.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/eit_one_run.bash\
                                /semantics/semantic_image\
                                $EIT_RECORDINGS_PATH/$room/${room}_record_${high_trajectory}.bag\
                                Transfiner_16\
                                $path
                        fi
                        ;;
                esac
            done # METHODS
        
        # SUPIX
        for baseline in "${SUPIX[@]}"
        do
            for low_trajectory in "${LOW_TRAJECTORIES[@]}"
            do
                for method in "${METHODS[@]}"
                do
                    echo "Room: $room -- supix: $baseline -- method: $method -- low_traj: $low_trajectory"
                    case $method in
                        mrcnn)
                            path=$MESH_RESULTS_PATH/${room}/low_supix_mrcnn_${high_trajectory}_${baseline}_${low_trajectory}.ply
                            if [ ! -f $path ] 
                            then
                                $SCRIPT_DIR/eit_one_run.bash\
                                    /semantics/superpixel\
                                    $EIT_RECORDINGS_PATH/${room}_record_${low_trajectory}.bag\
                                    $baseline\
                                    $MESH_RESULTS_PATH/${room}/high_mrcnn_${high_trajectory}.ply\
                                    $path
                            fi
                            ;;
                        transfiner)
                            path=$MESH_RESULTS_PATH/${room}/low_supix_transfiner_${high_trajectory}_${baseline}_${low_trajectory}.ply
                            if [ ! -f $path ] 
                            then
                                $SCRIPT_DIR/eit_one_run.bash\
                                    /semantics/superpixel\
                                    $EIT_RECORDINGS_PATH/${room}_record_${low_trajectory}.bag\
                                    $baseline\
                                    $MESH_RESULTS_PATH/${room}/high_transfiner_${high_trajectory}.ply\
                                    $path
                            fi
                            ;;
                        esac
                done # METHODS (SUPIX)
            done # LOW_TRAJECTORY
        done # SUPIX
    done # HIGH_TRAJECTORIES
done # ROOMS

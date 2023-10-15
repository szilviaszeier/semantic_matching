#!/bin/bash


# High Habitat
# High MRCNN
# High Transfiner
# Low Habitat
# Low MRCNN
# Low Transfiner
# Habitat Vertical Supix
# Habitat Camera Supix
# MRCNN Vertical Supix
# MRCNN Camera Supix
# Transfiner Vertical Supix
# Transfiner Camera Supix

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

HABITAT_DATA_PATH=/mnt/home/baffy-1000022/work/semantic_mapping/nipg30/kimera_ws/src/habitat_interface/data
MESH_RESULTS_PATH=$SCRIPT_DIR/../../kimera_ws/src/kimera_interface/mesh_results/habitat

ROOMS=(frl_apartment_4 room_2 office_3) # frl_apartment_4 room_2 office_3 office_4
NOISES=(0.0) # 0.0 1.0
HEIGHTS=(low) # high low
METHODS=(3d_neighbor_using_GT_mesh_fast_slic_100_10) # rgb habitat mrcnn transfiner trained_model
# 3d_neighbor_using_GT_mesh
# 3d_neighbor_using_GT_mesh_fast_slic_100_10
SUPIX=() # baseline camera 3d_neighbor dbscan 3d_supix 3d_slic


for room in "${ROOMS[@]}"
do
    for noise in "${NOISES[@]}"
    do
        noise_name=${noise%.*}

        # SEGMENTATION
        for height in "${HEIGHTS[@]}"
        do
            if [ $height = high ]
            then
                camera_level=1.8
            else
                camera_level=0.2
            fi
            for method in "${METHODS[@]}"
            do
                echo "Room: $room -- method: $method -- height: $height ($camera_level) -- noise: $noise"
                case $method in
                    habitat)
                        path=$MESH_RESULTS_PATH/${room}/${height}_gt_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /habitat/semantics/image_raw\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    mrcnn)
                        path=$MESH_RESULTS_PATH/${room}/${height}_mrcnn_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                Mask_R_CNN_16\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    transfiner)
                        path=$MESH_RESULTS_PATH/${room}/${height}_transfiner_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                Transfiner_16\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    trained_model)
                        path=$MESH_RESULTS_PATH/${room}/${height}_trained_model_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                trained_model\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    transf_trained_model)
                        path=$MESH_RESULTS_PATH/${room}/${height}_transf_trained_model_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                transf_trained_model\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    transf_trained_model)
                        path=$MESH_RESULTS_PATH/${room}/${height}_transf_trained_model_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                transf_trained_model\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    supix_trained_model)
                        path=$MESH_RESULTS_PATH/${room}/${height}_supix_trained_model_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                supix_trained_model\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;                        
                    rgb)
                        path=$MESH_RESULTS_PATH/${room}/${height}_rgb_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                rgb\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    3d_nb_model_final)
                        path=$MESH_RESULTS_PATH/${room}/${height}_3d_nb_model_final_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                3d_nb_model_final\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    fast_slic_model_final)
                        path=$MESH_RESULTS_PATH/${room}/${height}_fast_slic_model_final_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                fast_slic_model_final\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    gt_fast_slic)
                        path=$MESH_RESULTS_PATH/${room}/${height}_gt_fast_slic_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                gt_fast_slic\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    mrcnn_fast_slic_tested)
                        path=$MESH_RESULTS_PATH/${room}/${height}_mrcnn_fast_slic_tested_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                mrcnn_fast_slic_tested\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    mrcnn_model_final)
                        path=$MESH_RESULTS_PATH/${room}/${height}_mrcnn_model_final_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                mrcnn_model_final\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    3d_neighbor_using_GT_mesh_fast_slic_100_10)
                        path=$MESH_RESULTS_PATH/${room}/${height}_3d_neighbor_using_GT_mesh_fast_slic_100_10_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/semantic_image\
                                3d_neighbor_using_GT_mesh_fast_slic_100_10\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                $camera_level\
                                $HABITAT_DATA_PATH/$room/${height}_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;                    
                esac
                
            done # METHODS
        done # HEIGHTS
        
        # SUPIX
        for baseline in "${SUPIX[@]}"
        do
            for method in "${METHODS[@]}"
            do
                echo "Room: $room -- supix: $baseline -- method: $method -- noise: $noise"
                case $method in
                    habitat)
                        path=$MESH_RESULTS_PATH/${room}/low_supix_gt_${baseline}_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/superpixel\
                                $baseline\
                                $MESH_RESULTS_PATH/${room}/high_gt_${noise_name}_noise.ply\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                0.2\
                                $HABITAT_DATA_PATH/$room/low_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    mrcnn)
                        path=$MESH_RESULTS_PATH/${room}/low_supix_mrcnn_${baseline}_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/superpixel\
                                $baseline\
                                $MESH_RESULTS_PATH/${room}/high_mrcnn_${noise_name}_noise.ply\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                0.2\
                                $HABITAT_DATA_PATH/$room/low_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    transfiner)
                        path=$MESH_RESULTS_PATH/${room}/low_supix_transfiner_${baseline}_${noise_name}_noise.ply
                        if [ ! -f $path ] 
                        then
                            $SCRIPT_DIR/habitat_one_run.bash\
                                /semantics/superpixel\
                                $baseline\
                                $MESH_RESULTS_PATH/${room}/high_transfiner_${noise_name}_noise.ply\
                                /mnt/database/replica/$room/habitat/mesh_semantic.ply\
                                0.2\
                                $HABITAT_DATA_PATH/$room/low_traj.npy\
                                $noise\
                                $path
                        fi
                        ;;
                    esac
            done # METHODS (SUPIX)
        done # SUPIX_BASLINES
    done # NOISES
done # ROOMS

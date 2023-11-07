#!/bin/bash

# High Habitat - High MRCNN
# High Habitat - High Transfiner
# Low Habitat  - Low MRCNN
# Low Habitat  - Low Transfiner
# Low Habitat  - Habitat Vertical Supix
# Low Habitat  - Habitat Camera Supix
# Low Habitat  - MRCNN Vertical Supix
# Low Habitat  - MRCNN Camera Supix
# Low Habitat  - Transfiner Vertical Supix
# Low Habitat  - Transfiner Camera Supix

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

MESH_RESULTS_PATH=$SCRIPT_DIR/../../kimera_ws/src/kimera_interface/mesh_results/habitat

ROOMS=(frl_apartment_4 room_2 office_3) # room_2 office_3 office_4)
NOISES=(0.0) #1.0
HEIGHTS=(low)
METHODS=(3d_neighbor_using_GT_mesh_fast_slic_100_10) # trained_model gt mrcnn transfiner
# 3d_neighbor_using_GT_mesh
# 3d_neighbor_using_GT_mesh_fast_slic_100_10
SUPIX=() # baseline camera 3d_supix 3d_neighbor dbscan 3d_slic 

$SCRIPT_DIR/../apptainer/start.bash /home/szilvi/image_sandbox

for room in "${ROOMS[@]}"
do
    for noise in "${NOISES[@]}"
    do
        HIGH_EVALUATIONS=()
        LOW_EVALUATIONS=()

        noise_name=${noise%.*}

        for method in "${METHODS[@]}"
        do

            if [ $method != gt ]
            then
                #HIGH_EVALUATIONS+=($MESH_RESULTS_PATH/${room}/high_${method}_${noise_name}_noise.ply)
                LOW_EVALUATIONS+=($MESH_RESULTS_PATH/${room}/low_${method}_${noise_name}_noise.ply)
            fi

            for baseline in "${SUPIX[@]}" 
            do
                LOW_EVALUATIONS+=($MESH_RESULTS_PATH/${room}/low_supix_${method}_${baseline}_${noise_name}_noise.ply)
            done # SUPIX_BASLINES
        done # METHODS

        # compare to high_gt
        #$SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/compare_meshes.bash\
        #    $MESH_RESULTS_PATH/${room}/high_gt_${noise_name}_noise.ply\
        #    "${HIGH_EVALUATIONS[@]}"

        # compare to low_gt
        $SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/compare_meshes.bash\
            $MESH_RESULTS_PATH/${room}/low_gt_${noise_name}_noise.ply\
            "${LOW_EVALUATIONS[@]}"

    done # NOISES
done # ROOMS

$SCRIPT_DIR/../apptainer/stop.bash
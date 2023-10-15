#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

HABITAT_DATA_PATH=$SCRIPT_DIR/../../kimera_ws/src/habitat_interface/data/states_lower # states_lower contains trajectories, downloadable from the groups server
MESH_RESULTS_PATH=$SCRIPT_DIR/../../kimera_ws/src/kimera_interface/mesh_results/habitat_height

ROOMS=(frl_apartment_4 room_2 office_3) #
HEIGHTS=(0.2) # 0.2 0.8 1.2 1.8
METHODS=(transfiner) # habitat mrcnn transfiner
TRAJECTORIES=(0 1)
SUPIX=(baseline camera 3d_supix 3d_neighbor dbscan)

$SCRIPT_DIR/../singularity/start.bash /home/szilvi/image_sandbox/

for room in "${ROOMS[@]}"
do
    for trajectory in "${TRAJECTORIES[@]}"
    do
        HIGH_EVALUATIONS=()
        LOW_EVALUATIONS=()

        for method in "${METHODS[@]}"
        do
            if [ $method != gt ]
            then
                HIGH_EVALUATIONS+=($MESH_RESULTS_PATH/${room}_1.8_${trajectory}_${method}.ply)
                LOW_EVALUATIONS+=($MESH_RESULTS_PATH/${room}_0.2_${trajectory}_${method}.ply)
            fi

            for supix in "${SUPIX[@]}" 
            do
                LOW_EVALUATIONS+=($MESH_RESULTS_PATH/${room}_0.2_${supix}_${trajectory}_${method}.ply)
            done # SUPIX_BASLINES
        done # METHODS

        # compare to high_gt
        $SCRIPT_DIR/../singularity/exec.bash $SCRIPT_DIR/../run/compare_meshes.bash\
            $MESH_RESULTS_PATH/${room}_1.8_${trajectory}_gt.ply\
            "${HIGH_EVALUATIONS[@]}"

        # compare to low_gt
        $SCRIPT_DIR/../singularity/exec.bash $SCRIPT_DIR/../run/compare_meshes.bash\
            $MESH_RESULTS_PATH/${room}_0.2_${trajectory}_gt.ply\
            "${LOW_EVALUATIONS[@]}"
    done # TRAJECTORIES
done # ROOMS

$SCRIPT_DIR/../singularity/stop.bash
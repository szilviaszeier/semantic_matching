#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

MESH_RESULTS_PATH=$SCRIPT_DIR/../../kimera_ws/src/kimera_interface/mesh_results/

SEM_SEG=(mrcnn16 transf16)
HIGH_RECS=(1) #2
LOW_RECS=(3) 
SUPIX=(vertical camera 3d_supix 3d_neighbor dbscan 3d_slic) # 3d_slic 3d_neighbor 3d_supix camera dbscan vertical

$SCRIPT_DIR/../apptainer/start.bash /home/szilvi/image_sandbox/

HIGH_EVALUATIONS=()
LOW_EVALUATIONS=()

for sem_seg in "${SEM_SEG[@]}" 
do 
    for high_recs in "${HIGH_RECS[@]}" 
    do
        HIGH_EVALUATIONS+=($MESH_RESULTS_PATH/semantic_mesh_human_k4a_eit_meeting_rec_${high_recs}_${sem_seg}.ply)
    done # HIGH_RECS
done # SEM_SEG

for sem_seg in "${SEM_SEG[@]}" 
do
    for low_recs in "${LOW_RECS[@]}" 
    do
        LOW_EVALUATIONS+=($MESH_RESULTS_PATH/semantic_mesh_low_k4a_eit_meeting_rec_${low_recs}_${sem_seg}.ply)
        for supix in "${SUPIX[@]}"
        do 
            for high_recs in "${HIGH_RECS[@]}" 
            do
                LOW_EVALUATIONS+=($MESH_RESULTS_PATH/semantic_mesh_low_k4a_eit_meeting_rec_${low_recs}_${sem_seg}_rec_${high_recs}_${supix}.ply)
            done # HIGH_RECS
        done # SUPIX
    done # LOW_RECS
done # SEM_SEG

# HIGH EVALUATIONS
$SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/compare_meshes.bash\
    $MESH_RESULTS_PATH/semantic_mesh_human_k4a_eit_meeting_rec_1.ply\
    "${HIGH_EVALUATIONS[@]}"

# LOW EVALUATIONS
$SCRIPT_DIR/../apptainer/exec.bash $SCRIPT_DIR/../run/compare_meshes.bash\
    $MESH_RESULTS_PATH/semantic_mesh_low_k4a_eit_meeting_rec_3.ply\
    "${LOW_EVALUATIONS[@]}"


$SCRIPT_DIR/../apptainer/stop.bash
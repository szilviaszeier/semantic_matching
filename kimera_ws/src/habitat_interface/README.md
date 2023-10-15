# AiHabitat simulated experiments

Testing our pipeline on a simulation with ground truth dataset and virtual sensors.
We make use of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) for Rendering sensor measurement of the Replica dataset and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) for generated noise models.

## Installation

As in the reconstruction pipeline we provide a conda environment, where all components are installed for convenience.

To install the conda env:
```
cd $SEMANTIC_ROOT/config
conda env create -f habitat_conda.yml
```

We are going to use the same catkin workspace for communication with ROS that we have previously created in kimera_ws. 
Here the Python path is modified to use Python 3. 

## Downloading the Replica Dataset

1. Download the Replica dataset as described [here](https://github.com/facebookresearch/Replica-Dataset). 
   To work with the Replica dataset, you need a file called `sorted_faces.bin` for each model. 
   Such files (1 file per model), along with a convenient setup script can be downloaded from here: [sorted_faces.zip](http://dl.fbaipublicfiles.com/habitat/sorted_faces.zip). You need:

- Download the zip file from the above link - *Note: Expect aroung 80Gb of Data.... :(*
- Unzip it to any location with given path and locate the *sorted_faces* folder.
- Here run  `source copy_to_folders <Replica root directory. e.g ~/models/replica/>` which copies semantic description to the scene folders

## Running the experiment

Start roscore in opened terminal
`roscore`

Setup path and conda env:

```
cd $SEMANTIC_ROOT/scripts/local
source init_workspace.sh
cd $SEMANTIC_ROOT/kimera_ws/src/habitat_interface
conda activate habitat
```

Finally run the python interface node with

```
python traj_sim.py 
```

## Navigation

To control the virtual agent the window named `Arrow Keys` needs to be focused and one of the follow key combinations can be used to manipulate the movement.

- Move forward: Up
- Move backward: Down
- Rotate left: Left
- Rotate right: Right
- Exit: Escape
- Stop/Stay: Space
- Look up: Control+Up
- Look down: Control+Down
- Move up: Shift+Up
- Move down: Shift+Down

## Parameters to set

All argument are passed through roslaunch and are listed in the launch/simulation.launch file.

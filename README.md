# Cross-Viewpoint Semantic Mapping: Integrating Human and Robot Perspectives for Improved 3D Semantic Reconstruction

Official repository for "Cross-Viewpoint Semantic Mapping: Integrating Human and Robot Perspectives for Improved 3D Semantic Reconstruction" and the supplementary material for the Learning to Generate 3D Shapes and Scenes ECCV 2022 workshop submission, titled "3D Semantic Label Transfer and Matching in Human-Robot Collaboration".

Source code and database will be uploaded upon publication acceptance.
apptainer build

Code of semantic 3D reconstruction in simulated and real environments and edge-assisted semantic label transfer, based on the [Kimera](https://github.com/MIT-SPARK/Kimera) framework. The first published version of David Rozenberszki and Gabor Soros is at [github](https://github.com/RozDavid/semantic_mapping).

<hr />  
The workspace consists of the following components:

 - [kimera_interface](kimera_ws/src/kimera_interface/): connection between Kimera and the rest of our nodes (SLAM, semantic segmentation, etc)
 - [habitat_interface](kimera_ws/src/habitat_interface/): connection between ROS and Habitat
 - [semantic_segmentation](kimera_ws/src/semantic_segmentation/): ESANet RGBD segmentation
 - [superpixel_segmentation](kimera_ws/src/superpixel_segmentation/): superpixel-based RGB segmentation
 - [detectron2_ros](kimera_ws/src/detectron2_ros): contains the inference code for Transfiner or any model trained using detectron2
 - [openbot_navigation](kimera_ws/src/openbot_navigation/): commands to control our OpenBot robot
 - [rendering](kimera_ws/src/rendering/): PyRender-based mesh renderer for semantic label transfer to arbitrary viewpoints
 - [synthetic_data_postprocessing](synthetic_data_postprocessing/): contains the necessary post-processing procedures for training from lower viewpoint
 - [instance_segmentation](instance_segmentation/): contains post-processing instance segmentation

<hr />
 Supporting scripts for installation and execution can be found in the `scripts` folder:

 - The `setup` folder contains scripts for installations and builds.
 - The `apptainer` folder contains scripts interacting with apptainer.
 - The `run` folder contains scripts for launching nodes and running modules.
 - The `experiments` folder contains scripts automatically launching nodes for a particular experiment

## Installation with Apptainer containers
You will need a linux machine running apptainer for this type of installation.

### Clone the repository with **SSH**:
```bash
git clone --recurse-submodules git@github.com/szilviaszeier/semantic_matching.git
```
### Install Apptainer:
You can freely choose to install the Apptainer container runtime on your local machine or on a server. Building containers locally requires sudo access, but it is possible to build containers in the cloud if you do not have root access on your host. We describe both cases below.

#### Dependencies
First, you will also need to install some dependencies:.

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    pkg-config
```

#### Build and install Apptainer
To install Apptainer, follow the steps provided [here](https://github.com/apptainer/apptainer).

### Building the image and building the code

Note that the container requires about 3 GB and building the source requires about 10 GB disk space.

Go to the root of this repository and run the following commands:

1. Build the Apptainer container. Replace `<path/to/container.sif>` with the desired output path for the container.

    Build the container locally:  
    In development stage, add the `--sandbox` parameter to be able to modify the container later.

    ```bash
    apptainer build  <path/to/container.sif> install_base.def
    ```

2. Adapt the following environment variables to match your current directory structure:
    ```bash
    ./scripts/setup/envfile
    ```

3. Start an apptainer instance:
    ```bash
    ./scripts/apptainer/start.bash <path/to/container.sif>
    ```

4. Create Conda environments:
    ```bash
    ./scripts/apptainer/exec.bash ./scripts/setup/conda.bash
    ```

5. Initialize and build all the code in the catkin workspace:
    ```bash
    ./scripts/apptainer/exec.bash ./scripts/setup/build.bash
    ```
## Running nodes

After starting an instance with

```bash
./scripts/apptainer/start.bash <path/to/image.sif>
```

You can start any node with the following command:

```bash
./scripts/apptainer/exec.bash ./scripts/run/<node-to-start.bash> <further-args>
```

You can specify further arguments to the node after the pervious command with the following syntax:
```
<arg-name>:=<value>
```

(Except for the Mask R-CNN node, because that is not yet implemented under roslaunch.)

## Example with Habitat to RVIZ rendering

Terminal 1
```bash
cd $SEMANTIC_ROOT
./scripts/apptainer/start.bash semantic_mapping.sif  # replace with your SIF name
./scripts/apptainer/exec.bash ./scripts/run/roscore.bash
```

Terminal 2
```bash
cd $SEMANTIC_ROOT
./scripts/apptainer/exec.bash ./scripts/run/roslaunch_habitat.bash mesh_path:=/home/esoptron/data/replica/apartment_0/habitat/mesh_semantic.ply
```

Terminal 3
```bash
cd $SEMANTIC_ROOT
./scripts/apptainer/exec.bash ./scripts/run/rviz.bash
```

## Example using the Apptainer shell
This example shows how to start the habitat rendering via the Apptainer shell. You can view the rendered images (that get published on the ROS topics) in RVIZ.
```bash
cd $SEMANTIC_ROOT
./scripts/apptainer/start.bash semantic_mapping.sif  # replace with your SIF name
apptainer shell instance://inst --nv --env-file <enfile_path>
source /opt/ros/melodic/setup.bash
source /opt/conda/etc/profile.d/conda.sh
conda activate habitat
source $SEMANTIC_ROOT/kimera_ws/devel/setup.bash
roslaunch habitat_interface simulation.launch mesh_path:=$SEMANTIC_DATA_DIR/replica/apartment_0/habitat/mesh_semantic.ply
```

## Alternative Installation
Legacy installation steps without Apptainer can be found [here](legacy_installation.md)

 ## Citing
```text
@article{kopacsi2023cross,
  title={Cross-Viewpoint Semantic Mapping: Integrating Human and Robot Perspectives for Improved 3D Semantic Reconstruction},
  author={Kopacsi, Laszlo and Baffy, Benjamin and Baranyi, Gabor and Skaf, Joul and Soros, Gabor and Szeier, Szilvia and Lorincz, Andras and Sonntag, Daniel},
  journal={Sensors},
  volume={23},
  number={11},
  pages={5126},
  year={2023},
  publisher={MDPI}
}
```
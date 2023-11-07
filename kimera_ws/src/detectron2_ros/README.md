
# Transfiner Inferencing
The repository contains only the inference code for Transfiner or any model trained using detectron2

## Installing the required libraries using conda
```commandline
conda create -n transfiner python=3.9 -y
source activate transfiner
 
conda install pytorch torchvision cudatoolkit -c pytorch

# ninja yacs (*optional)
pip install ninja yacs cython matplotlib tqdm 
pip install opencv-python 
pip install scikit-image 
pip install kornia

pip install 'git+https://github.com/facebookresearch/detectron2.git'

```

There is also a singularity script attached. However, it is recommended to install using conda, to avoid future compatibility issues.

## Running the code
just execute "run.sh" file as is, or modify depending on the used server.

## Pretrained Models
The models trained and evaluated can befound under [pretrained_models](pretrained_models/).

## Additional information
- The code supports multiple GPUs
- It is optimized to gain the highest performance, meaning the prediction phases takes the majority of the time not the code excecution itself (~7 FPS on Nvidia GeForce GTX 1660 Ti)
- The classes and their colors are dynamic and subject to change depending on the provided model
- Colors are saved on a json file once they are generated, their order is based on the provided categories

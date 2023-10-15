#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh

conda env create -f $SEMANTIC_ROOT/config/semantic_mapping_conda.yaml
conda run -n semantic_mapping pip install cython
conda run -n semantic_mapping pip install eigency 

conda env update -f $SEMANTIC_ROOT/config/habitat_conda.yml
conda env create -f $SEMANTIC_ROOT/config/superpixel_conda.yml
conda env create -f $SEMANTIC_ROOT/config/detectron2_conda.yml
conda run -n detectron2 python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
# Create conda environment
conda create -n transfiner python=3.9 -y
source activate transfiner

# Install PyTorch dependencies
conda install pytorch torchvision cudatoolkit -c pytorch -y

# Install additional required libraries
pip install ninja yacs cython matplotlib tqdm
pip install opencv-python 
pip install scikit-image
pip install kornia
pip install shapely plotly


# Install detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'




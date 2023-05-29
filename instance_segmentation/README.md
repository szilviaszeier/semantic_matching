## Indoor instance segmentation using pytorch and detectron2 

### requirements
1) OS: ubuntu =< 18.x 
2) Conda: Miniconda/Anaconda
3) CUDA, CPU can be also used by modifying few parameters.

### installation
``` commandline
# Create conda environment
conda create -n transfiner python=3.9 -y
source activate transfiner

# Install PyTorch dependencies
conda install pytorch torchvision cudatoolkit -c pytorch -y

# Install additional required libraries
pip install ninja yacs cython matplotlib tqdm  -y
pip install opencv-python 
pip install scikit-image
pip install kornia
pip install shapely plotly


# Install detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
or simply run "installation.sh"


### Data format:
- Images should have an appropriate format (png, jpg/jpeg), with 3 channels.
- Annotations should be of coco format otherwise a mapping function should be specified.

### Folder structure:
- Datasets, directory containing the required datasets, multiple ones can be specified.
- Output, directory containing the training output, including the checkpoints and the final model. In order to resume training the existence of these checkpoints is required.
- Models, directory containing the pre-trained models and their required libraries.

### Running the code
Specify the required arguments or pass them in the main function.
``` commandline
python main.py [optional arguments]
```
```
usage: main.py [-h] [-dp DATA_PATH] [-dc DICTIONARIES_PATH] [-ld LOAD_DICTIONARIES] [-aug USE_DATA_AUGMENTATION] [-prn PRE_TRAINED_MODEL] [-pbb PRE_TRAINED_BACKBONE]
               [-wn WORKERS_NUMBER] [-ug USE_GPU] [-ipb IMAGES_PER_BATCH] [-lr BASE_LR] [-it MAX_ITERATIONS] [-bs BATCH_SIZE] [-out OUTPUT_DIR] [-th SCORE_THRESHOLD]
               [-mn MODEL_NAME] [-aj ANNOTATIONS_JSON]

optional arguments:
  -h, --help            show this help message and exit
  -dp DATA_PATH, --data-path DATA_PATH
                        Main datasets directory
  -dc DICTIONARIES_PATH, --dictionaries-path DICTIONARIES_PATH
                        JSON coco format annotations path
  -ld LOAD_DICTIONARIES, --load-dictionaries LOAD_DICTIONARIES
                        Indicates whether to load saved data dictionaries or not
  -aug USE_DATA_AUGMENTATION, --use-data-augmentation USE_DATA_AUGMENTATION
                        Indicates whether to augment the training data or not.
  -prn PRE_TRAINED_MODEL, --pre-trained-model PRE_TRAINED_MODEL
                        Pretrained model name
  -pbb PRE_TRAINED_BACKBONE, --pre-trained-backbone PRE_TRAINED_BACKBONE
                        Pretrained model backbone
  -wn WORKERS_NUMBER, --workers-number WORKERS_NUMBER
                        Number of workers
  -ug USE_GPU, --use-gpu USE_GPU
                        Indicate whether to use gpu or cpu
  -ipb IMAGES_PER_BATCH, --images-per-batch IMAGES_PER_BATCH
                        Number of images per batch across all machines This is also the number of training images per step
  -lr BASE_LR, --base-lr BASE_LR
                        Learning rate
  -it MAX_ITERATIONS, --max-iterations MAX_ITERATIONS
                        Number of iterations
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Number of regions per image used to train RPN
  -out OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Saved model output dir.
  -th SCORE_THRESHOLD, --score-threshold SCORE_THRESHOLD
                        Prediction score minimum threshold
  -mn MODEL_NAME, --model-name MODEL_NAME
                        Saved model name
  -aj ANNOTATIONS_JSON, --annotations-json ANNOTATIONS_JSON
                        Initial annotations file name


```
The dataset path, and classes should be specified in main.py as well.

### Parameter-tuning 
- The training parameters are already tuned to a near optimal degree, further tuning is possible but does not guarantee a significant improvement in the results.
- Learning rate decay is enabled by default as part of the tuning process.

### Training the lower perspective dataset:
- Carefully check the main.py file, and select the appropriate annotations file (annotations-json) from there options (20, 10, 5, 1), where each represent the number of frames that were skipped.
- Select the appropriate method from used_method = ["seman"], ["supix"], or [transf]:
- Change the number of iterations and batch_size according to this formular (epochs = (iterations * batch_size) / number of images in the training data)
- The model could start overfitting after a couple of epochs

## Replica dataset

[paper](https://arxiv.org/abs/1906.05797)

[github](https://github.com/facebookresearch/Replica-Dataset)

- Download the Replica dataset as described [here](https://github.com/facebookresearch/Replica-Dataset).
   To work with the Replica dataset, you need a file called `sorted_faces.bin` for each model.
   Such files (1 file per model), along with a convenient setup script can be downloaded from here: [sorted_faces.zip](http://dl.fbaipublicfiles.com/habitat/sorted_faces.zip). You need:

- Download the zip file from the above link - *Note: Expect aroung 80Gb of Data.... :(*
- Unzip it to any location with given path and locate the *sorted_faces* folder.
- Here run  `source copy_to_folders <Replica root directory. e.g ~/models/replica/>` which copies semantic description to the scene folders


- Categories in decreasing number of appearence:

```python
[
    "book",
    "wall",
    "lamp",
    "wall-plug",
    "blinds",
    "chair",
    "table",
    "door",
    "cushion",
    "bowl",
    "window",
    "switch",
    "anonymize_text",
    "bottle",
    "anonymize_picture",
    "indoor-plant",
    "cup",
    "box",
    "vent",
    "ceiling",
    "pillow",
    "panel",
    "vase",
    "handrail",
    "plate",
    "floor",
    "clothing",
    "pot",
    "basket",
    "non-plane",
    "rug",
    "shoe",
    "plant-stand",
    "pillar",
    "cabinet",
    "bin",
    "rack",
    "camera",
    "shelf",
    "tv-screen",
    "picture",
    "blanket",
    "sofa",
    "nightstand",
    "small-appliance",
    "mat",
    "countertop",
    "beanbag",
    "bike",
    "sink",
    "base-cabinet",
    "faucet",
    "kitchen-utensil",
    "wall-cabinet",
    "tissue-paper",
    "chopping-board",
    "curtain",
    "tablet",
    "major-appliance",
    "remote-control",
    "scarf",
    "sculpture",
    "bed",
    "stair",
    "tv-stand",
    "refrigerator",
    "stool",
    "comforter",
    "umbrella",
    "plane",
    "knife-block",
    "handbag",
    "pan",
    "clock",
    "shower-stall",
    "towel",
    "toilet",
    "desk",
    "pipe",
    "bench",
    "cloth",
    "candle",
    "desk-organizer",
    "utensil-holder",
    "coaster",
    "bathtub",
    "cooktop",
    "monitor"
]
```



## Matterport3D dataset

[link](https://niessner.github.io/Matterport/)

[paper](https://arxiv.org/abs/1709.06158)

[github](https://github.com/niessner/Matterport)

(You need to register and accept the MP3D terms and conditions first!)
Use the following script to download the Matterport3D data: [download_mp.py](http://kaldir.vc.in.tum.de/matterport/download_mp.py).
Scan data is named by a house hash id. The list of house hash ids is at [scans.txt](http://kaldir.vc.in.tum.de/matterport/v1/scans.txt).

Script usage:
- To download the entire Matterport3D release (1.3TB): download-mp.py -o [directory in which to download]
- To download a specific scan (e.g., 17DRP5sb8fy): download-mp.py -o [directory in which to download] --id 17DRP5sb8fy
- To download a specific file type (e.g., *.sens, valid file suffixes listed here): download-mp.py -o [directory in which to download] --type .sens
- *.sens files can be read using the sens-File reader (it's a bit easier to handle than a larger number of separate image files)
Example:
```
python download-mp.py -o $HOME/data/mp3d --id 17DRP5sb8fy
```
WARNING: the `download_mp.py` script is written for python2. If you have python3, you need to replace these lines:
```
raw_input() --> input()
print msg --> print(msg)
```
Next, download the extra files for the habitat task (http://kaldir.vc.in.tum.de/matterport/v1/tasks//mp3d_habitat.zip):
```
python download-mp.py -o $HOME/data/mp3d --id 17DRP5sb8fy --task_data habitat
```


## 3RScan dataset

[link](http://vmnavab26.in.tum.de/3RScan/)

[paper](https://arxiv.org/pdf/1908.06109.pdf)

[github](https://github.com/WaldJohannaU/3RScan)

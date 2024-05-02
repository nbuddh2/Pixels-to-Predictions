# Object Tracking using YOLO, Deep Sort and Tensorflow
This repository utilizes YOLO along with Deep SORT to facilitate real-time object tracking. YOLO leverages deep convolutional neural networks for object detection. The detections from YOLO are then integrated with Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric), enabling the development of a real-time object tracking system.

## Getting started
Create a venv or conda environment with the following dependencies.
dependencies:
  - python==3.7
  - pip
  - matplotlib
  - opencv

#### Pip
```
pip install -r requirements.txt
```
# yolo
Download yolo model (https://drive.google.com/drive/folders/1h77gvTMMIlmoN7Wn4_7URY92-eAcdMHP?usp=drive_link)
Download the weights for the project using this link (https://drive.google.com/drive/folders/1VZomK53Dqaq7iHbRNKx4_Ddt0sHaueH_?usp=drive_link)

## Running the Object Tracker
Now you can run the object tracker for the model.
```
python object_tracker.py
```
## Github Link
(https://github.com/nbuddh2/Pixels-to-Predictions)

## Acknowledgments
* [Deep SORT Repository](https://github.com/nwojke/deep_sort)

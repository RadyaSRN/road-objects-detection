[![Open In nbviewer](https://img.shields.io/badge/Jupyter-nbviewer-orange?logo=jupyter)](
https://nbviewer.org/github/RadyaSRN/road-objects-detection/blob/main/notebooks/road_objects_detection.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](
https://www.kaggle.com/kernels/welcome?src=https://github.com/RadyaSRN/road-objects-detection/blob/main/notebooks/road_objects_detection.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/RadyaSRN/road-objects-detection/blob/main/notebooks/road_objects_detection.ipynb)
[![W&B Report](https://img.shields.io/badge/Weights%20&%20Biases-Report-orange?logo=weightsandbiases)](
https://wandb.ai/radyasrn-mipt/CV-spring-2025/reports/CV-spring-2025-road-objects-detection--VmlldzoxNDIxMDgxMg)

# Road objects detection
Training detection models (**RetinaNet and SSD**) to detect road objects, then applying a model to **real world traffic video from Moscow**.

![road-objects-detection-20fps](images/road-objects-detection-20fps.gif)

### Detection dataset
For model training the [Traffic Road Object Detection Polish 12k](https://www.kaggle.com/datasets/mikoajkoek/traffic-road-object-detection-polish-12k) was used.

### Usage
* The first option is to open and run the notebook `/notebooks/road_objects_detection.ipynb` with comments and visualizations in Kaggle or Google Colab.

* The second option is cloning the repo, installing the needed requirements, and working locally:
```
git clone https://github.com/RadyaSRN/road-objects-detection.git
cd road-objects-detection
conda create -n roadobj python=3.10
conda activate roadobj
pip install -r requirements.txt
```

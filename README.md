# Pascal VOC Weakly Supervised Semantic Segmentation
This is a repository for the final project of IIT4204, SW Project, Yonsei University

## Prerequisites
- Python3( >= 3.6)
- PyTorch( >= 1.1)
- OpenCV
- tqdm
- tensorboardX

## Data preparation
- PascalVOC [download link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

## Execution

#### train
```
# You must check where the results are saved.
https://github.com/halbielee/SW_Project.git
cd SW_Project
bash script/train.sh
```

#### evaluation for segmentation
```
# You must specify the GT, PRED segmentation map directory.
python evaluate_pred.py
```
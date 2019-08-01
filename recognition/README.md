# InsightFace_recognition

### How to use
- clone

`git clone https://github.com/Fei-Wang/insightface.git`

### Data Preparation
- Prepare Facebank
    - Including train set, validation set and test set
    - Guarantee it have a structure like following:

```
data/facebank/
    id1/
        id1_1.jpg
        id1_2.jpg
    id2/
        id2_1.jpg
        id2_2.jpg
        id2_3.jpg
```

### Training
- Set config file

    `vim ./recognition/configs/config.yaml`

- Training

```
cd ./recognition
export PYTHONPATH=../:$PYTHONPATH
python train.py
```

### Evaluate model
`python predict.py`
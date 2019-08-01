# Insightface-tf

### Intro
- This repositories is a reimplementation of Insightface([github](https://github.com/deepinsight/insightface))
- Training code is included, you can use your own data to train models
- We are trying to use new features and Hight Level APIs in TesnsorFlow, such as Keras, Eager Execution, tf.data and so on. The code can even run successful under TF2.0 with a little changes
- The recognition([github-recognition](https://github.com/Fei-Wang/insightface/tree/master/recognition)) with tf-1.14 has been reimplemented and the RetinaFace-tf will be upload soon

### TODO List
- recognition
    - Backbones
        - _ResNet_v1 [done]_
    - Losses
        - _Arcface loss [done]_
        - _Cosface loss [done]_
        - _Sphereface loss [done]_
        - _Triplet loss [done]_
        - _Center loss [done]_
    - _Training code [done]_
    - _Evaluate [done]_
    - Freeze to pb model [todo]
- RetinaFace [todo]

### Running Environment
- TensorFlow1.14
- python 3.7
- numpy, pyyaml, matplotlib (Anaconda 3 recommended)
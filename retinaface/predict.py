from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys

import tensorflow as tf
import yaml

from retinaface.backbones.resnet_v1_fpn import ResNet_v1_50_FPN
from retinaface.models.models import RetinaFace
from retinaface.utils.anchor import generate_anchors

tf.enable_eager_execution()


def predict(model, images, anchors):
    cls, box, lmk = model(images, training=False)  # shape=[(N, 160, 160, 32),(1/2),(1/4),(1/8),(1/16)]
    # 1. 根据anchor和坐标位置映射为原图位置 shape=(N, H*W*2， 16)

    # 2. 根据阈值进行排除 shape=(N, X, 16)
    # 3. 根据NMS进行排除 shape=(N, Y, 16)
    return cls, box, lmk


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--config_path', type=str, help='path to config path', default='configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    args = parse_args(sys.argv[1:])
    # logger.info(args)
    from retinaface.data.generate_data import GenerateData

    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data = gd.get_train_data()
    model = RetinaFace(ResNet_v1_50_FPN)
    anchors = generate_anchors(config)

    for img, _ in train_data.take(1):
        cls, box, lmk = predict(model, img, anchors)

        print(img.shape, img[0].shape)
        for i in box:
            print(i.shape)


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

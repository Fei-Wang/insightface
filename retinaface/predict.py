from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys

import tensorflow as tf
import yaml

from retinaface.backbones.resnet_v1 import ResNet_v1_50
from retinaface.models.models import MyModel

tf.enable_eager_execution()


def predict(model, images):
    pre = model(images, training=False)  # shape=(N, H, W, 32)
    # 1. 根据anchor和坐标位置映射为原图位置 shape=(N, H*W*2， 16)
    # 2. 根据阈值进行排除 shape=(N, X, 16)
    # 3. 根据NMS进行排除 shape=(N, Y, 16)
    return pre


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
    model = MyModel(ResNet_v1_50)

    ckpt_dir = os.path.expanduser(config['ckpt_dir'])
    ckpt = tf.train.Checkpoint(backbone=model.backbone)
    ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
    print("Restored from {}".format(tf.train.latest_checkpoint(ckpt_dir)))
    # for layer in tf.train.list_variables(tf.train.latest_checkpoint(ckpt_dir)):
    #     print(layer)

    for img, label in train_data.take(1):
        value = predict(model, img)
        # print(value.shape)
        # print(label.bounding_shape())


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys

import tensorflow as tf

from logger import logger

tf.enable_eager_execution()


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train face network')

    args = parser.parse_args(argv)

    return args


def train_net(args=None):
    # DenseNet121(...)
    # DenseNet169(...)
    # DenseNet201(...)
    # InceptionResNetV2(...)
    # InceptionV3(...)
    # MobileNet(...)
    # MobileNetV2(...)
    # NASNetLarge(...)
    # NASNetMobile(...)
    # ResNet50(...)
    # VGG16(...)
    # VGG19(...)
    # Xception(...)
    model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
    model.summary()
    # n = 0
    #  for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
    #      train_step(image_x, image_y)
    #      if n % 10 == 0:
    #          print('.', end='')
    #      n += 1
    pass


def main():
    args = parse_args(sys.argv[1:])
    logger.info(args)
    train_net(args)


if __name__ == '__main__':
    logger.info("hello, insightface/recognition")
    main()

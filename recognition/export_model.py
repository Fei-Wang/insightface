from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys

import tensorflow as tf
import yaml

from recognition.backbones.resnet_v1 import ResNet_v1_50
from recognition.data.generate_data import GenerateData
from recognition.models.models import MyModel

tf.enable_eager_execution()


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--config_path', type=str, help='path to config path', default='configs/config.yaml')

    args = parser.parse_args(argv)

    return args


args = parse_args(sys.argv[1:])

with open(args.config_path) as cfg:
    config = yaml.load(cfg, Loader=yaml.FullLoader)
gd = GenerateData(config)
train_data, _ = gd.get_train_data()
model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'])

ckpt_dir = os.path.expanduser(config['ckpt_dir'])
ckpt = tf.train.Checkpoint(backbone=model.backbone)
ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
print("Restored from {}".format(tf.train.latest_checkpoint(ckpt_dir)))

# print(tf.executing_eagerly())
for img, _ in train_data.take(1):
    _ = model(img)

    pb_dir = os.path.join(ckpt_dir, 'pb')
    tf.saved_model.save(model, pb_dir)
    loaded = tf.compat.v2.saved_model.load(pb_dir)
    test, _, _ = loaded.call(img)
    print(test)

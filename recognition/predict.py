from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys

import tensorflow as tf
import yaml

from backbones.resnet_v1 import ResNet_v1_50
from data.generate_data import GenerateData
from models.models import MyModel

tf.enable_eager_execution()


def get_embeddings(model, images):
    prelogits, _, _ = model(images, training=False)
    embeddings = tf.nn.l2_normalize(prelogits, axis=-1)

    return embeddings


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--config_path', type=str, help='path to config path', default='configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    args = parse_args(sys.argv[1:])
    # logger.info(args)

    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data, _ = gd.get_train_data()
    model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'])

    ckpt_dir = os.path.expanduser(config['ckpt_dir'])
    ckpt = tf.train.Checkpoint(backbone=model.backbone)
    ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
    print("Restored from {}".format(tf.train.latest_checkpoint(ckpt_dir)))
    # for layer in tf.train.list_variables(tf.train.latest_checkpoint(ckpt_dir)):
    #     print(layer)

    for img, _ in train_data.take(1):
        embs = get_embeddings(model, img)
        for i in range(embs.shape[0]):
            for j in range(embs.shape[0]):
                val = 0
                for k in range(512):
                    val += embs[i][k] * embs[j][k]
                print(i, j, val)


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

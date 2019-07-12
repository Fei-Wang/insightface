from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from logger import logger

tf.enable_eager_execution()


def softmax_loss(dense, labels):
    logits = tf.keras.layers.Softmax()(dense)
    # ce or softmax_with_ce
    print(dense, logits, labels)
    return 0


class SphereFace(tf.keras.layers.Layer):

    def __init__(self, classes=1000):
        super(SphereFace, self).__init__()
        self.classes = classes

    def build(self, input_shape):
        self.w = self.add_weight(name='norm_dense_w', shape=(input_shape[-1], self.classes),
                                 initializer='random_normal', trainable=True)

    def call(self, inputs):
        norm_w = tf.nn.l2_normalize(self.w, axis=0)
        x = tf.matmul(inputs, norm_w)
        # x2 = tf.matmul(inputs, self.w)
        # norm1 = tf.norm(self.w, axis=0, keepdims=True)
        #
        # # print(self.w, norm)
        # x3 = x2 / norm1
        return x


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description='define losses.')
    parser.add_argument('--config_path', type=str, help='path to config path', default='../configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    import sys
    args = parse_args(sys.argv[1:])
    logger.info(args)
    sys.path.append("..")
    from data.generate_data import GenerateData
    from backbones.resnet_v1 import ResNet_v1_50
    from models.models import MyModel
    import yaml
    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data = gd.get_train_data()

    model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'], classes=3)

    for img, label in train_data.take(1):
        prelogits, dense, norm_dense = model(img, training=False)
        sm_loss = softmax_loss(dense, label)
        # embeddings = tf.nn.l2_normalize(prelogits, axis=1)
        print(sm_loss)


if __name__ == '__main__':
    logger.info("hello, insightface/recognition")
    main()

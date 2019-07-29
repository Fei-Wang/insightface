from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from retinaface.backbones.resnet_v1 import ResNet_v1_18, ResNet_v1_34, ResNet_v1_50, ResNet_v1_101, ResNet_v1_152

tf.enable_eager_execution()


class FPN(tf.keras.Model):
    """Feature Pyramid Network - https://arxiv.org/abs/1612.03144"""

    def __init__(self, backbone=ResNet_v1_50):
        super(FPN, self).__init__()
        self.backbone = backbone()
        self.bottom_up = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')
        self.lateral_5 = tf.keras.layers.Conv2D(256, (1, 1), padding='same')
        self.lateral_4 = tf.keras.layers.Conv2D(256, (1, 1), padding='same')
        self.lateral_3 = tf.keras.layers.Conv2D(256, (1, 1), padding='same')
        self.lateral_2 = tf.keras.layers.Conv2D(256, (1, 1), padding='same')
        self.anti_aliasing4 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')
        self.anti_aliasing3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')
        self.anti_aliasing2 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')
        self.top_down = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, inputs, training=False, mask=None):
        c2, c3, c4, c5 = self.backbone(inputs, training=training)
        p6 = self.bottom_up(c5)
        p5 = self.lateral_5(c5)
        p4 = self.lateral_4(c4)
        p4 += self.top_down(p5)
        p4 = self.anti_aliasing4(p4)
        p3 = self.lateral_3(c3)
        p3 += self.top_down(p4)
        p3 = self.anti_aliasing3(p3)
        p2 = self.lateral_2(c2)
        p2 += self.top_down(p3)
        p2 = self.anti_aliasing2(p2)
        # print(p2.shape)
        # print(p3.shape)
        # print(p4.shape)
        # print(p5.shape)
        # print(p6.shape)

        return p2, p3, p4, p5, p6


class ResNet_v1_18_FPN(FPN):
    def __init__(self):
        super(ResNet_v1_18_FPN, self).__init__(backbone=ResNet_v1_18)


class ResNet_v1_34_FPN(FPN):
    def __init__(self):
        super(ResNet_v1_34_FPN, self).__init__(backbone=ResNet_v1_34)


class ResNet_v1_50_FPN(FPN):
    def __init__(self):
        super(ResNet_v1_50_FPN, self).__init__(backbone=ResNet_v1_50)


class ResNet_v1_101_FPN(FPN):
    def __init__(self):
        super(ResNet_v1_101_FPN, self).__init__(backbone=ResNet_v1_101)


class ResNet_v1_152_FPN(FPN):
    def __init__(self):
        super(ResNet_v1_152_FPN, self).__init__(backbone=ResNet_v1_152)


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Resnet v1 model.')
    parser.add_argument('--config_path', type=str, help='path to config path', default='../configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    import sys
    args = parse_args(sys.argv[1:])
    # logger.info(args)
    from retinaface.data.generate_data import GenerateData
    import yaml
    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data = gd.get_train_data()
    # model = ResNet_v1_18_FPN()
    # model.build((None, 640, 640, 3))
    # model = ResNet_v1_34_FPN()
    # model.build((None, 640, 640, 3))
    model = ResNet_v1_50_FPN()
    model.build((None, 640, 640, 3))
    # model = ResNet_v1_101_FPN()
    # model.build((None, 640, 640, 3))
    # model = ResNet_v1_152_FPN()
    # model.build((None, 640, 640, 3))
    model.summary()


if __name__ == '__main__':
    main()

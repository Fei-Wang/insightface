from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()


class RetinaFace(tf.keras.Model):
    """RetinaFace - https://arxiv.org/abs/1905.00641"""

    def __init__(self, fpn):
        super(RetinaFace, self).__init__()
        self.fpn = fpn()
        # self.conv = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same')

    def call(self, inputs, training=False, mask=None):
        x = self.fpn(inputs, training=training)
        # x = self.conv(x)
        return x


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description='design model.')
    parser.add_argument('--config_path', type=str, help='path to config path', default='../configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    import sys
    args = parse_args(sys.argv[1:])
    from retinaface.data.generate_data import GenerateData
    from retinaface.backbones.resnet_v1_fpn import ResNet_v1_50_FPN
    import yaml
    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data = gd.get_train_data()

    model = RetinaFace(ResNet_v1_50_FPN)
    model.build((None, 640, 640, 3))
    model.summary()
    # for img, _ in train_data.take(1):
    #     y = model(img, training=False)
    #     print(img.shape, img[0].shape, y.shape, y)


if __name__ == '__main__':
    main()

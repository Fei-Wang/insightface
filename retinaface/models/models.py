from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()


class MyModel(tf.keras.Model):
    def __init__(self, backbone):
        super(MyModel, self).__init__()
        self.backbone = backbone(include_top=False)
        self.conv = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same')

    def call(self, inputs, training=False, mask=None):
        x = self.backbone(inputs, training=training)
        x = self.conv(x)
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
    sys.path.insert(1, "..")
    from data.generate_data import GenerateData
    from backbones.resnet_v1 import ResNet_v1_50
    import yaml
    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data = gd.get_train_data()

    # model = ResNet_v1_50(embedding_size=config['embedding_size'])
    model = MyModel(ResNet_v1_50)
    # model.build((None, 112, 112, 3))
    # model.summary()
    for img, _ in train_data.take(1):
        y = model(img, training=False)
        print(img.shape, img[0].shape, y.shape, y)


if __name__ == '__main__':
    main()

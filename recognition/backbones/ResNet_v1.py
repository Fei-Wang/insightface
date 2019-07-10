from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from logger import logger

tf.enable_eager_execution()


class MyLayer(tf.keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ResNet_v1(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(ResNet_v1, self).__init__()
        self.num_classes = num_classes

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_classes)
        self.dense2 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.dense(x)
        x = self.dense2(x)

        return x


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Resnet v1 model.')
    parser.add_argument('--config_path', type=str, help='path to config path', default='../configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    import sys
    args = parse_args(sys.argv[1:])
    logger.info(args)
    # sys.path.append("..")
    # from data.generate_data import GenerateData
    # import yaml
    # with open(args.config_path) as cfg:
    #     config = yaml.load(cfg, Loader=yaml.FullLoader)
    # gd = GenerateData(config)
    # train_data = gd.get_train_data()

    model = ResNet_v1(num_classes=10)

    # inputs = tf.keras.Input(shape=(112, 112))
    # outputs = ResNet_v1(num_classes=10)(inputs)
    # model = tf.keras.Model(inputs, outputs)
    # model.summary()
    # tf.keras.utils.plot_model(model, 'my_first_model.png', show_shapes=True)

    # for img, _ in train_data.take(1):
    #     y = model(img)
    #     print(img.shape, img[0].shape, y.shape)
    # model.summary()


if __name__ == '__main__':
    logger.info("hello, insightface/recognition")
    main()

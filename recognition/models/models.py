from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()


class NormDense(tf.keras.layers.Layer):

    def __init__(self, classes=1000):
        super(NormDense, self).__init__()
        self.classes = classes

    def build(self, input_shape):
        self.w = self.add_weight(name='norm_dense_w', shape=(input_shape[-1], self.classes),
                                 initializer='random_normal', trainable=True)

    def call(self, inputs, **kwargs):
        norm_w = tf.nn.l2_normalize(self.w, axis=0)
        x = tf.matmul(inputs, norm_w)

        return x


class MyModel(tf.keras.Model):
    def __init__(self, backbone, embedding_size=512, classes=1000):
        super(MyModel, self).__init__()
        self.backbone = backbone(include_top=True, embedding_size=embedding_size)
        self.dense = tf.keras.layers.Dense(classes)
        self.norm_dense = NormDense(classes)

    @tf.function
    def call(self, inputs, training=False, mask=None):
        prelogits = self.backbone(inputs, training=training)
        dense = self.dense(prelogits)
        norm_dense = self.norm_dense(prelogits)
        return prelogits, dense, norm_dense


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description='design model.')
    parser.add_argument('--config_path', type=str, help='path to config path', default='../configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    import sys
    args = parse_args(sys.argv[1:])
    # logger.info(args)
    sys.path.append("..")
    from data.generate_data import GenerateData
    from backbones.resnet_v1 import ResNet_v1_50
    import yaml
    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data, classes = gd.get_train_data()

    # model = ResNet_v1_50(embedding_size=config['embedding_size'])
    model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'], classes=classes)
    model.build((None, 112, 112, 3))
    model.summary()
    # for img, _ in train_data.take(1):
    #     y = model(img, training=False)
    #     # print(img.shape, img[0].shape, y.shape, y)
    #     print(y)


if __name__ == '__main__':
    # log_cfg_path = '../../logging.yaml'
    # with open(log_cfg_path, 'r') as f:
    #     dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    # logging.config.dictConfig(dict_cfg)
    # logger = logging.getLogger("mylogger")
    # logger.info("hello, insightface/recognition")
    main()

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filters=64, strides=(1, 1)):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=strides)
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        if x.shape == inputs.shape:
            res = inputs
        else:
            res = self.conv3(inputs)
            res = self.bn3(res, training=training)
        x += res
        x = self.relu(x)
        return x


class Bottleneck(tf.keras.layers.Layer):

    def __init__(self, filters=64, strides=(1, 1)):
        super(Bottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding='same', strides=strides)
        self.bn4 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        if x.shape == inputs.shape:
            res = inputs
        else:
            res = self.conv4(inputs)
            res = self.bn4(res, training=training)
        x += res
        x = self.relu(x)
        return x


class ResNet_v1(tf.keras.Model):
    def __init__(self, Block=Bottleneck, layers=(3, 4, 6, 3)):
        super(ResNet_v1, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')
        self.blocks1 = tf.keras.Sequential([Block(filters=64, strides=(1, 1)) for _ in range(layers[0])])
        self.blocks2 = tf.keras.Sequential(
            [Block(filters=128, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[1])])
        self.blocks3 = tf.keras.Sequential(
            [Block(filters=256, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[2])])
        self.blocks4 = tf.keras.Sequential(
            [Block(filters=512, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[3])])
        # self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
        # self.dense = None
        # if include_top:
        #     self.dense = tf.keras.layers.Dense(embedding_size)

    def call(self, inputs, training=False, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.blocks1(x, training=training)
        c3 = self.blocks2(c2, training=training)
        c4 = self.blocks3(c3, training=training)
        c5 = self.blocks4(c4, training=training)
        # x = self.globalpool(x)
        # if self.dense is not None:
        #     x = self.dense(x)

        return c2, c3, c4, c5


class ResNet_v1_18(ResNet_v1):
    def __init__(self):
        super(ResNet_v1_18, self).__init__(Block=BasicBlock, layers=(2, 2, 2, 2))


class ResNet_v1_34(ResNet_v1):
    def __init__(self):
        super(ResNet_v1_34, self).__init__(Block=BasicBlock, layers=(3, 4, 6, 3))


class ResNet_v1_50(ResNet_v1):
    def __init__(self):
        super(ResNet_v1_50, self).__init__(Block=Bottleneck, layers=(3, 4, 6, 3))


class ResNet_v1_101(ResNet_v1):
    def __init__(self):
        super(ResNet_v1_101, self).__init__(Block=Bottleneck, layers=(3, 4, 23, 3))


class ResNet_v1_152(ResNet_v1):
    def __init__(self):
        super(ResNet_v1_152, self).__init__(Block=Bottleneck, layers=(3, 8, 36, 3))


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

    model = ResNet_v1_18()
    model.build((None, 640, 640, 3))
    model.summary()
    model = ResNet_v1_34()
    model.build((None, 640, 640, 3))
    model.summary()
    model = ResNet_v1_50()
    model.build((None, 640, 640, 3))
    model.summary()
    model = ResNet_v1_101()
    model.build((None, 640, 640, 3))
    model.summary()
    model = ResNet_v1_152()

    model.build((None, 640, 640, 3))
    model.summary()
    # model = tf.keras.applications.ResNet50(input_shape=(112, 112, 3), include_top=False)
    # model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')
    # model = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3))
    # model.summary()
    # inputs = tf.keras.Input(shape=(112, 112, 3))
    # outputs = ResNet_v1_50(embedding_size=512)(inputs, training=False)
    # model = tf.keras.Model(inputs, outputs)
    # model.summary()
    # tf.keras.utils.plot_model(model, 'my_first_model.png', show_shapes=True)

    # for img, _ in train_data.take(1):
    #     y = model(img, training=False)
    #     print(img.shape, img[0].shape, y.shape, y)


if __name__ == '__main__':
    # log_cfg_path = '../../logging.yaml'
    # with open(log_cfg_path, 'r') as f:
    #     dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    # logging.config.dictConfig(dict_cfg)
    # logger = logging.getLogger("mylogger")
    # logger.info("hello, insightface/recognition")
    main()

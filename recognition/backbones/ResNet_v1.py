from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from logger import logger

tf.enable_eager_execution()


# class ResNetBlock_A(tf.keras.layers.Layer):
#
#     def __init__(self, filters=64, strides=(1, 1)):
#         super(ResNetBlock_A, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=strides)
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.relu1 = tf.keras.layers.ReLU()
#         self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         self.relu2 = tf.keras.layers.ReLU()
#         self.conv3 = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=strides)
#         self.bn3 = tf.keras.layers.BatchNormalization()
#
#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         # print('A2', x.shape, inputs.shape, x.shape == inputs.shape)
#         if x.shape == inputs.shape:
#             res = inputs
#         else:
#             res = self.conv3(inputs)
#             res = self.bn3(res)
#         x += res
#         x = self.relu2(x)
#         return x


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

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # print('A2', x.shape, inputs.shape, x.shape == inputs.shape)
        if x.shape == inputs.shape:
            res = inputs
        else:
            res = self.conv3(inputs)
            res = self.bn3(res)
        x += res
        x = self.relu(x)
        return x


#
# class ResNetBlock_B(tf.keras.layers.Layer):
#
#     def __init__(self, filters=64, strides=(1, 1)):
#         super(ResNetBlock_B, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=strides)
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.relu1 = tf.keras.layers.ReLU()
#         self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         self.relu2 = tf.keras.layers.ReLU()
#         self.conv3 = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding='same')
#         self.bn3 = tf.keras.layers.BatchNormalization()
#         self.relu3 = tf.keras.layers.ReLU()
#         self.conv4 = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding='same', strides=strides)
#         self.bn4 = tf.keras.layers.BatchNormalization()
#
#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         if x.shape == inputs.shape:
#             res = inputs
#         else:
#             res = self.conv4(inputs)
#             res = self.bn4(res)
#         x += res
#         x = self.relu3(x)
#         return x


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

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if x.shape == inputs.shape:
            res = inputs
        else:
            res = self.conv4(inputs)
            res = self.bn4(res)
        x += res
        x = self.relu(x)
        return x


# class ResNetBlock_B2(tf.keras.layers.Layer):
#
#     def __init__(self, filters=64):
#         super(ResNetBlock_B2, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=(2, 2))
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.relu1 = tf.keras.layers.ReLU()
#         self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         self.relu2 = tf.keras.layers.ReLU()
#         self.conv3 = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding='same')
#         self.bn3 = tf.keras.layers.BatchNormalization()
#         self.relu3 = tf.keras.layers.ReLU()
#         self.conv4 = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding='same', strides=(2, 2))
#         self.bn4 = tf.keras.layers.BatchNormalization()
#
#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         print(x.shape, inputs.shape, x.shape == inputs.shape)
#         res = self.conv4(inputs)
#         res = self.bn4(res)
#         x += res
#         x = self.relu3(x)
#         return x

#
# class ResNet_v1_18(tf.keras.Model):
#     def __init__(self, num_classes=10):
#         super(ResNet_v1_18, self).__init__()
#         # self.num_classes = num_classes
#         self.conv = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')
#         self.bn = tf.keras.layers.BatchNormalization()
#         self.relu = tf.keras.layers.ReLU()
#         self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')
#         self.blocks1 = [ResNetBlock_A1(filters=64)] + [ResNetBlock_A1(filters=64) for _ in range(1)]
#         self.blocks2 = [ResNetBlock_A2(filters=128)] + [ResNetBlock_A1(filters=128) for _ in range(1)]
#         self.blocks3 = [ResNetBlock_A2(filters=256)] + [ResNetBlock_A1(filters=256) for _ in range(1)]
#         self.blocks4 = [ResNetBlock_A2(filters=512)] + [ResNetBlock_A1(filters=512) for _ in range(1)]
#         self.blocks = self.blocks1 + self.blocks2 + self.blocks3 + self.blocks4
#         self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
#         self.dense = tf.keras.layers.Dense(num_classes)
#         self.softmax = tf.keras.layers.Softmax()
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.conv(inputs)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         for block in self.blocks:
#             x = block(x)
#         x = self.globalpool(x)
#         x = self.dense(x)
#         x = self.softmax(x)
#
#         return x
#
#
# class ResNet_v1_34(tf.keras.Model):
#     def __init__(self, num_classes=10):
#         super(ResNet_v1_34, self).__init__()
#         # self.num_classes = num_classes
#         self.conv = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')
#         self.bn = tf.keras.layers.BatchNormalization()
#         self.relu = tf.keras.layers.ReLU()
#         self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')
#         self.blocks1 = [ResNetBlock_A1(filters=64)] + [ResNetBlock_A1(filters=64) for _ in range(2)]
#         self.blocks2 = [ResNetBlock_A2(filters=128)] + [ResNetBlock_A1(filters=128) for _ in range(3)]
#         self.blocks3 = [ResNetBlock_A2(filters=256)] + [ResNetBlock_A1(filters=256) for _ in range(5)]
#         self.blocks4 = [ResNetBlock_A2(filters=512)] + [ResNetBlock_A1(filters=512) for _ in range(2)]
#         self.blocks = self.blocks1 + self.blocks2 + self.blocks3 + self.blocks4
#         self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
#         self.dense = tf.keras.layers.Dense(num_classes)
#         self.softmax = tf.keras.layers.Softmax()
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.conv(inputs)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         for block in self.blocks:
#             x = block(x)
#         x = self.globalpool(x)
#         x = self.dense(x)
#         x = self.softmax(x)
#
#         return x


class ResNet_v1(tf.keras.Model):
    def __init__(self, Block=Bottleneck, layers=(3, 4, 6, 3), num_classes=10):
        super(ResNet_v1, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')
        self.blocks1 = [Block(filters=64, strides=(1, 1)) for _ in range(layers[0])]
        self.blocks2 = [Block(filters=128, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[1])]
        self.blocks3 = [Block(filters=256, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[2])]
        self.blocks4 = [Block(filters=512, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[3])]
        self.blocks = self.blocks1 + self.blocks2 + self.blocks3 + self.blocks4
        self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(num_classes)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for block in self.blocks:
            x = block(x)
        print(x.shape)
        x = self.globalpool(x)
        x = self.dense(x)
        x = self.softmax(x)

        return x


class ResNet_v1_18(ResNet_v1):
    def __init__(self, num_classes=10):
        super(ResNet_v1_18, self).__init__(Block=BasicBlock, layers=(2, 2, 2, 2), num_classes=num_classes)


class ResNet_v1_34(ResNet_v1):
    def __init__(self, num_classes=10):
        super(ResNet_v1_34, self).__init__(Block=BasicBlock, layers=(3, 4, 6, 3), num_classes=num_classes)


class ResNet_v1_50(ResNet_v1):
    def __init__(self, num_classes=10):
        super(ResNet_v1_50, self).__init__(Block=Bottleneck, layers=(3, 4, 6, 3), num_classes=num_classes)


class ResNet_v1_101(ResNet_v1):
    def __init__(self, num_classes=10):
        super(ResNet_v1_101, self).__init__(Block=Bottleneck, layers=(3, 4, 23, 3), num_classes=num_classes)


class ResNet_v1_152(ResNet_v1):
    def __init__(self, num_classes=10):
        super(ResNet_v1_152, self).__init__(Block=Bottleneck, layers=(3, 8, 36, 3), num_classes=num_classes)


#
# class ResNet_v1_50(tf.keras.Model):
#     def __init__(self, num_classes=10):
#         super(ResNet_v1_50, self).__init__()
#         # self.num_classes = num_classes
#         self.conv = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')
#         self.bn = tf.keras.layers.BatchNormalization()
#         self.relu = tf.keras.layers.ReLU()
#         self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')
#         self.blocks1 = [ResNetBlock_B(filters=64, strides=(1, 1)) for _ in range(3)]
#         self.blocks2 = [ResNetBlock_B(filters=128, strides=(2, 2))] + [ResNetBlock_B(filters=128, strides=(1, 1)) for _
#                                                                        in range(3)]
#         self.blocks3 = [ResNetBlock_B(filters=256, strides=(2, 2))] + [ResNetBlock_B(filters=256, strides=(1, 1)) for _
#                                                                        in range(5)]
#         self.blocks4 = [ResNetBlock_B(filters=512, strides=(2, 2))] + [ResNetBlock_B(filters=512, strides=(1, 1)) for _
#                                                                        in range(2)]
#         self.blocks = self.blocks1 + self.blocks2 + self.blocks3 + self.blocks4
#         self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
#         self.dense = tf.keras.layers.Dense(num_classes)
#         self.softmax = tf.keras.layers.Softmax()
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.conv(inputs)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         for block in self.blocks:
#             x = block(x)
#         x = self.globalpool(x)
#         x = self.dense(x)
#         x = self.softmax(x)
#
#         return x
#
#
# class ResNet_v1_101(tf.keras.Model):
#     def __init__(self, num_classes=10):
#         super(ResNet_v1_101, self).__init__()
#         # self.num_classes = num_classes
#         self.conv = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')
#         self.bn = tf.keras.layers.BatchNormalization()
#         self.relu = tf.keras.layers.ReLU()
#         self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')
#         self.blocks1 = [ResNetBlock_B1(filters=64)] + [ResNetBlock_B1(filters=64) for _ in range(2)]
#         self.blocks2 = [ResNetBlock_B2(filters=128)] + [ResNetBlock_B1(filters=128) for _ in range(3)]
#         self.blocks3 = [ResNetBlock_B2(filters=256)] + [ResNetBlock_B1(filters=256) for _ in range(22)]
#         self.blocks4 = [ResNetBlock_B2(filters=512)] + [ResNetBlock_B1(filters=512) for _ in range(2)]
#         self.blocks = self.blocks1 + self.blocks2 + self.blocks3 + self.blocks4
#         self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
#         self.dense = tf.keras.layers.Dense(num_classes)
#         self.softmax = tf.keras.layers.Softmax()
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.conv(inputs)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         for block in self.blocks:
#             x = block(x)
#         x = self.globalpool(x)
#         x = self.dense(x)
#         x = self.softmax(x)
#
#         return x
#
#
# class ResNet_v1_152(tf.keras.Model):
#     def __init__(self, num_classes=10):
#         super(ResNet_v1_152, self).__init__()
#         # self.num_classes = num_classes
#         self.conv = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')
#         self.bn = tf.keras.layers.BatchNormalization()
#         self.relu = tf.keras.layers.ReLU()
#         self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')
#         self.blocks1 = [ResNetBlock_B1(filters=64)] + [ResNetBlock_B1(filters=64) for _ in range(2)]
#         self.blocks2 = [ResNetBlock_B2(filters=128)] + [ResNetBlock_B1(filters=128) for _ in range(7)]
#         self.blocks3 = [ResNetBlock_B2(filters=256)] + [ResNetBlock_B1(filters=256) for _ in range(35)]
#         self.blocks4 = [ResNetBlock_B2(filters=512)] + [ResNetBlock_B1(filters=512) for _ in range(2)]
#         self.blocks = self.blocks1 + self.blocks2 + self.blocks3 + self.blocks4
#         self.globalpool = tf.keras.layers.GlobalAveragePooling2D()
#         self.dense = tf.keras.layers.Dense(num_classes)
#         self.softmax = tf.keras.layers.Softmax()
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.conv(inputs)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         for block in self.blocks:
#             x = block(x)
#         x = self.globalpool(x)
#         x = self.dense(x)
#         x = self.softmax(x)
#
#         return x


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
    sys.path.append("..")
    # from data.generate_data import GenerateData
    # import yaml
    # with open(args.config_path) as cfg:
    #     config = yaml.load(cfg, Loader=yaml.FullLoader)
    # gd = GenerateData(config)
    # train_data = gd.get_train_data()

    # model = ResNet_v1_50(num_classes=10)
    # model = tf.keras.applications.ResNet50(input_shape=(112, 112, 3), include_top=False)
    # model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')
    # model = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3))
    # model.summary()
    inputs = tf.keras.Input(shape=(112, 112, 3))
    outputs = ResNet_v1_50(num_classes=10)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    # tf.keras.utils.plot_model(model, 'my_first_model.png', show_shapes=True)

    # for img, _ in train_data.take(1):
    #     y = model(img)
    #     print(img.shape, img[0].shape, y.shape, y)


if __name__ == '__main__':
    logger.info("hello, insightface/recognition")
    main()

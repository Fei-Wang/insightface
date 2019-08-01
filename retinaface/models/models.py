from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()


class ContextModule(tf.keras.Model):
    # TODO: not sure exactly how to construct the module
    def __init__(self):
        super(ContextModule, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=False, mask=None):
        x = self.conv1(inputs)
        p1 = self.conv2(x)
        p2 = self.conv3(p1)
        p3 = self.conv4(p2)
        p = tf.concat([p1, p2, p3], axis=-1)
        return p


class RetinaFace(tf.keras.Model):
    """RetinaFace - https://arxiv.org/abs/1905.00641"""

    def __init__(self, fpn, num_class=2, anchor_per_scale=3):
        super(RetinaFace, self).__init__()
        self.num_class = num_class
        self.fpn = fpn()
        self.cm = [ContextModule() for _ in range(5)]
        self.cls_conv = [tf.keras.layers.Conv2D(num_class * anchor_per_scale, (3, 3), padding='same') for _ in range(5)]
        self.box_conv = [tf.keras.layers.Conv2D(4 * anchor_per_scale, (3, 3), padding='same') for _ in range(5)]
        self.lmk_conv = [tf.keras.layers.Conv2D(10 * anchor_per_scale, (3, 3), padding='same') for _ in range(5)]
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=False, mask=None):
        features = self.fpn(inputs, training=training)
        x = [self.cm[i](features[i]) for i in range(len(features))]
        cls = [self.cls_conv[i](x[i]) for i in range(len(features))]
        box = [self.box_conv[i](x[i]) for i in range(len(features))]
        lmk = [self.lmk_conv[i](x[i]) for i in range(len(features))]

        # no param part, for calc convenience
        cls = [tf.reshape(cls[i], (cls[i].shape[0], cls[i].shape[1], cls[i].shape[2], -1, self.num_class)) for i in
               range(len(features))]
        cls = [self.softmax(cls[i]) for i in range(len(features))]
        box = [tf.reshape(box[i], (box[i].shape[0], box[i].shape[1], box[i].shape[2], -1, 4)) for i in
               range(len(features))]
        lmk = [tf.reshape(lmk[i], (lmk[i].shape[0], lmk[i].shape[1], lmk[i].shape[2], -1, 10)) for i in
               range(len(features))]
        # pred = [tf.concat([cls[i], box[i], lmk[i]], axis=-1) for i in range(len(features))]

        return cls, box, lmk


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
    # model.build((None, 640, 640, 3))
    # model.summary()
    for img, _ in train_data.take(1):
        cls, box, lmk = model(img, training=False)
        print(img.shape, img[0].shape)
        for i in box:
            print(i.shape)


if __name__ == '__main__':
    main()

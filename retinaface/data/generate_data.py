from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf

tf.enable_eager_execution()


class GenerateData:

    def __init__(self, config=None):
        self.config = config
        self._trian_paths, self._trian_labels = self._get_path_label(self.config['train_dir'])
        self._valid_paths, self._valid_labels = self._get_path_label(self.config['valid_dir'])

    @staticmethod
    def _get_path_label(train_dir):
        train_dir = os.path.expanduser(train_dir)
        image_dir = os.path.join(train_dir, 'images')
        label_file = os.path.join(train_dir, 'label.txt')
        with open(label_file, 'r') as f:
            lines = f.readlines()
        paths = []
        labels = []
        path_label_dict = {}
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                path = os.path.join(image_dir, line[1:].strip())
                path_label_dict[path] = []
                continue

            components = line.split(' ')
            path_label_dict[path].append(components)

        for path, label in path_label_dict.items():
            paths.append(path)
            labels.append(label)

        return paths, labels

    def _preprocess(self, image_path, trianing=True):
        image_raw = tf.io.read_file(image_path)
        # image = tf.image.decode_image(image_raw)
        image = tf.image.decode_png(image_raw)
        image = tf.cast(image, tf.float32)
        image = image / 255
        image = tf.image.resize(image, (self.config['image_size'], self.config['image_size']))

        # image = tf.image.resize(image, (224, 224))
        # image = tf.image.random_crop(image, size=[112, 112, 3])
        # image = tf.image.random_flip_left_right(image)

        # image = image[None, ...]
        return image

    def _preprocess_train(self, image_path, label):
        image = self._preprocess(image_path, trianing=True)

        return image, label

    def get_train_data(self):
        paths, labels = self._trian_paths, self._trian_labels
        assert (len(paths) == len(labels))
        total = len(paths)
        labels = tf.ragged.constant(labels)
        train_dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        train_dataset = train_dataset.map(self._preprocess_train,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
            total).batch(self.config['batch_size'])

        return train_dataset


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Generate Data.')
    parser.add_argument('--config_path', type=str, help='path to config path', default='../configs/config.yaml')
    args = parser.parse_args(argv)

    return args


def main():
    import sys
    args = parse_args(sys.argv[1:])
    import yaml
    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)

    # dataset = tf.data.Dataset.range(100)

    # dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
    # for i in dataset:
    #     print(i)
    gd = GenerateData(config)
    train_data = gd.get_train_data()
    import matplotlib.pyplot as plt
    for img, label in train_data.take(1):
        print(img.shape)
        print(label.shape)
        plt.imshow(img[0])
        plt.show()
        pass


if __name__ == '__main__':
    main()

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
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

        max = 0
        idx = -1
        for i, l in enumerate(labels):
            if max < len(l):
                max = len(l)
                idx = i
        print(idx)
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

    def _preprocess_train_triplet(self, image_path1, image_path2, image_path3):
        image1 = self._preprocess(image_path1, trianing=True)
        image2 = self._preprocess(image_path2, trianing=True)
        image3 = self._preprocess(image_path3, trianing=True)

        return image1, image2, image3

    def _preprocess_val(self, image_path1, image_path2, label):
        image1 = self._preprocess(image_path1, trianing=False)
        image2 = self._preprocess(image_path2, trianing=False)

        return image1, image2, label

    def get_train_data(self):
        paths, labels = self._trian_paths, self._trian_labels
        assert (len(paths) == len(labels))
        total = len(paths)
        train_dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        train_dataset = train_dataset.map(self._preprocess_train,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
            total).batch(self.config['batch_size'])

        return train_dataset

    def get_val_data(self, num):
        paths = self._valid_paths
        paths1 = []
        paths2 = []
        labels = []
        oo = 0
        while oo < num / 2:
            num_cls = np.random.randint(0, len(paths))
            cls = paths[num_cls]
            if len(cls) > 0:
                im_no1 = np.random.randint(0, len(cls))
                im_no2 = np.random.randint(0, len(cls))
                if im_no1 != im_no2:
                    paths1.append(cls[im_no1])
                    paths2.append(cls[im_no2])
                    labels.append(True)
                    oo = oo + 1

        nn = 0
        while nn < num / 2:
            num_cls1 = np.random.randint(0, len(paths))
            num_cls2 = np.random.randint(0, len(paths))
            if num_cls1 != num_cls2:
                cls1 = paths[num_cls1]
                cls2 = paths[num_cls2]
                if len(cls1) > 0 and len(cls2) > 0:
                    im_no1 = np.random.randint(0, len(cls1))
                    im_no2 = np.random.randint(0, len(cls2))
                    paths1.append(cls1[im_no1])
                    paths2.append(cls2[im_no2])
                    labels.append(False)
                    nn = nn + 1

        val_dataset = tf.data.Dataset.from_tensor_slices((paths1, paths2, labels))
        val_dataset = val_dataset.map(self._preprocess_val,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(num).batch(
            self.config['valid_batch_size'])

        return val_dataset


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

    dataset = tf.data.Dataset.range(100)


    dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
    for i in dataset:
        print(i)
    # gd = GenerateData(config)
    # train_data = gd.get_train_data()
    # import matplotlib.pyplot as plt
    # for img, label in train_data.take(1):
    #     plt.imshow(img[0])
    #     plt.show()
    pass

if __name__ == '__main__':
    main()

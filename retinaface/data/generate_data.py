from __future__ import absolute_import, division, print_function, unicode_literals

import os

import cv2
import tensorflow as tf

tf.enable_eager_execution()


class GenerateData:

    def __init__(self, config=None):
        self.config = config
        self._train_paths, self._train_labels = self._get_path_label(self.config['train_dir'],
                                                                     self.config['image_size'])
        # self._valid_paths, self._valid_labels = self._get_path_label(self.config['valid_dir'],
        #                                                              self.config['image_size'])

    @staticmethod
    def _get_path_label(train_dir, img_size):
        train_dir = os.path.expanduser(train_dir)
        image_dir = os.path.join(train_dir, 'images')
        label_file = os.path.join(train_dir, 'label.txt')
        with open(label_file, 'r') as f:
            lines = f.readlines()
        paths = []
        labels = []
        path_label_dict = {}
        idx_x = [0, 2, 4, 7, 10, 13, 16]
        idx_y = [idx_x[i] + 1 for i in range(len(idx_x))]
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                components = line.split(' ')
                path = os.path.join(image_dir, components[1])
                path_label_dict[path] = []
                # ori_shape = cv2.imread(path).shape
                # h = ori_shape[0]
                # w = ori_shape[1]
                w = int(components[2])
                h = int(components[3])
                continue

            components = line.split(' ')

            label = []
            for i in range(len(idx_x)):
                if i == 1:
                    # transform w, h to x, y
                    x_ = float(components[idx_x[i]]) * img_size / max(w, h) + label[0]
                    y_ = float(components[idx_y[i]]) * img_size / max(w, h) + label[1]
                else:
                    x_ = (float(components[idx_x[i]]) + max((h - w) / 2, 0)) * img_size / max(w, h)
                    y_ = (float(components[idx_y[i]]) + max((w - h) / 2, 0)) * img_size / max(w, h)
                label.append(x_)
                label.append(y_)
            # components = [float(components[i]) for i in range(len(components))]
            path_label_dict[path].append(label)

        for path, label in path_label_dict.items():
            paths.append(path)
            labels.append(label)

        return paths, labels

    def _preprocess(self, image_path, training=True):
        image_raw = tf.io.read_file(image_path)
        # image = tf.image.decode_image(image_raw)
        image = tf.image.decode_jpeg(image_raw)
        image = tf.cast(image, tf.float32)
        image = image / 255
        # # image = tf.image.resize(image, (self.config['image_size'], self.config['image_size']))
        image = tf.compat.v1.image.resize_image_with_pad(image, self.config['image_size'], self.config['image_size'])
        # image = tf.image.random_crop(image, size=[112, 112, 3])
        # image = tf.image.random_flip_left_right(image)

        # image = image[None, ...]
        return image

    def _preprocess_train(self, image_path, label):
        image = self._preprocess(image_path, training=True)

        return image, label, image_path

    def get_train_data(self):
        paths, labels = self._train_paths, self._train_labels
        assert (len(paths) == len(labels))
        total = len(paths)
        labels = tf.ragged.constant(labels)

        train_dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(total)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.map(self._preprocess_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(self.config['batch_size'])

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

    gd = GenerateData(config)
    train_data = gd.get_train_data()
    for img, label, path in train_data.take(1):
        print(img.shape)
        print(label.bounding_shape())
        print(path)

        img = cv2.cvtColor(img[0].numpy(), cv2.COLOR_BGR2RGB)
        boxes = label[0]
        for box in boxes:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            for i in range(5):
                cv2.circle(img, (int(box[4 + 2 * i]), int(box[4 + 2 * i + 1])), 2, (0, 255, 0), -1)

        cv2.imshow('img', img)
        cv2.waitKey()


if __name__ == '__main__':
    main()

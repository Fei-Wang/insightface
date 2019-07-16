from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


class GenerateData:

    def __init__(self, config=None):
        self.config = config

    @staticmethod
    def __get_path_label(image_dir):
        image_dir = os.path.expanduser(image_dir)
        ids = list(os.listdir(image_dir))
        ids.sort()
        cat_num = len(ids)
        # logger.info("the total people number is {}".format(cat_num))
        id_dict = dict(zip(ids, list(range(cat_num))))
        paths = []
        labels = []
        for i in ids:
            cur_dir = os.path.join(image_dir, i)
            fns = os.listdir(cur_dir)
            paths.append([os.path.join(cur_dir, fn) for fn in fns])
            labels.append([id_dict[i]] * len(fns))
        return paths, labels

    def __preprocess(self, image_path, trianing=True):
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

    def __preprocess_train(self, image_path, label):
        image = self.__preprocess(image_path, trianing=True)

        return image, label

    def __preprocess_val(self, image_path1, image_path2, label):
        image1 = self.__preprocess(image_path1, trianing=False)
        image2 = self.__preprocess(image_path2, trianing=False)

        return image1, image2, label

    def get_train_data(self):
        paths, labels = self.__get_path_label(self.config['train_dir'])
        cat_num = len(paths)
        paths = [path for cls in paths for path in cls]
        labels = [label for cls in labels for label in cls]
        assert (len(paths) == len(labels))
        total = len(paths)
        # logger.info("the total pic number is {}".format(total))
        # tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
        # tfrec.write(image_ds)
        # filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
        # dataset = tf.data.TFRecordDataset(filenames)
        train_dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        train_dataset = train_dataset.map(self.__preprocess_train,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
            total).batch(self.config['batch_size'])

        return train_dataset, cat_num

    def get_val_data(self, num):
        paths, _ = self.__get_path_label(self.config['valid_dir'])
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
        val_dataset = val_dataset.map(self.__preprocess_val,
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
    # logger.info(args)
    import yaml
    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data, classes = gd.get_train_data()
    import matplotlib.pyplot as plt
    # for img, _ in train_data.take(1):
    #     plt.imshow(img[0])
    #     plt.show()

    val_data = gd.get_val_data(3)
    for img1, img2, label in val_data:
        print(label)
        plt.imshow(img1[0])
        plt.show()

        plt.imshow(img2[0])
        plt.show()


if __name__ == '__main__':
    # log_cfg_path = '../../logging.yaml'
    # with open(log_cfg_path, 'r') as f:
    #     dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    # logging.config.dictConfig(dict_cfg)
    # logger = logging.getLogger("mylogger")
    # logger.info("hello, insightface/recognition")
    main()

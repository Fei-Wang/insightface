from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from backbones.resnet_v1 import ResNet_v1_50
from data.generate_data import GenerateData
from models.models import MyModel
from predict import get_embeddings

tf.enable_eager_execution()


class Valid_Data:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    @staticmethod
    def __cal_cos_sim(emb1, emb2):
        return tf.reduce_sum(emb1 * emb2, axis=-1)

    def __get_sim_label(self):
        sims = None
        labels = None
        for image1, image2, label in self.data:
            emb1 = get_embeddings(self.model, image1)
            emb2 = get_embeddings(self.model, image2)
            sim = self.__cal_cos_sim(emb1, emb2)
            if sims is None:
                sims = sim
            else:
                sims = tf.concat([sims, sim], axis=0)

            if labels is None:
                labels = label
            else:
                labels = tf.concat([labels, label], axis=0)

        return sims, labels

    @staticmethod
    def __cal_metric(sim, label, thresh):
        tp = tn = fp = fn = 0
        predict = tf.greater_equal(sim, thresh)
        for i in range(len(predict)):
            if predict[i] and label[i]:
                tp += 1
            elif predict[i] and not label[i]:
                fp += 1
            elif not predict[i] and label[i]:
                fn += 1
            else:
                tn += 1
        # 防止分母为零，影响不大
        acc = (tp + tn) / len(predict)
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        fpr = fp / (fp + tn + 1e-6)
        return acc, p, r, fpr

    def get_metric(self, thresh):
        sim, label = self.__get_sim_label()
        return self.__cal_metric(sim, label, thresh)

    def draw_curve(self):
        P = []
        R = []
        TPR = []
        FPR = []
        sim, label = self.__get_sim_label()
        for thresh in np.arange(0, 1, 0.01):
            acc, p, r, fpr = self.__cal_metric(sim, label, thresh)
            P.append(p)
            R.append(r)
            TPR.append(r)
            FPR.append(fpr)

        plt.axis([0, 1, 0, 1])
        plt.xlabel("R")
        plt.ylabel("P")
        plt.plot(R, P, color="r", linestyle="--", marker="*", linewidth=1.0)
        plt.show()

        plt.axis([0, 1, 0, 1])
        plt.xlabel("FRP")
        plt.ylabel("TPR")
        plt.plot(FPR, TPR, color="r", linestyle="--", marker="*", linewidth=1.0)
        plt.show()


def parse_args(argv):
    parser = argparse.ArgumentParser(description='valid model')
    parser.add_argument('--config_path', type=str, help='path to config path', default='configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    args = parse_args(sys.argv[1:])
    # logger.info(args)

    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    valid_data = gd.get_val_data(config['valid_num'])
    model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'], classes=12)
    vd = Valid_Data(model, valid_data)
    acc, p, r, fpr = vd.get_metric(0.995)
    # print(p, r, tpr, fpr)
    # vd.draw_curve()


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

from retinaface.backbones.resnet_v1_fpn import ResNet_v1_50_FPN
from retinaface.models.models import RetinaFace
from retinaface.predict import predict
from retinaface.utils.anchor import AnchorUtil
from retinaface.utils.box import cal_iou, box_filter


class ValidData:
    def __init__(self, model, data, au):
        self.model = model
        self.data = data
        self.au = au

    def get_metric(self, conf_thresh, iou_thresh, top_k):
        tps = 0
        fps = 0
        fns = 0
        for imgs, labels, _ in self.data:  # for each batch
            preds = predict(self.model, imgs, self.au)
            # print(preds.shape)
            dets = box_filter(preds, conf_thresh, iou_thresh, top_k)
            for i, det in enumerate(dets):  # for each image
                # label = labels[i]
                label = np.array(labels[i].to_list())
                num_truth = label.shape[0]  # truth num
                num_pred_truth = det.shape[0]  # pred truth num
                if num_truth == 0:
                    tp = 0
                    fp = num_pred_truth
                    fn = 0
                else:
                    match = set()
                    for box in det:
                        # print(box.shape, label.shape)
                        iou = cal_iou(box[2:], label)
                        idx = np.argmax(iou)
                        if iou[idx] >= 0.5:
                            match.add(idx)
                    tp = len(match)
                    fp = num_pred_truth - tp
                    fn = num_truth - tp

                tps += tp
                fps += fp
                fns += fn

        p = tps / (tps + fps) if tps + fps != 0 else 0
        r = tps / (tps + fns) if tps + fns != 0 else 0
        # print(p, r)
        return p, r

    def draw_curve(self, iou_thresh, top_k):
        P = []
        R = []
        for thresh in np.linspace(0, 1, 100):
            p, r = self.get_metric(thresh, iou_thresh, top_k)
            P.append(p)
            R.append(r)

        plt.axis([0, 1, 0, 1])
        plt.xlabel("R")
        plt.ylabel("P")
        plt.plot(R, P, color="r", linestyle="--", marker="*", linewidth=1.0)
        plt.show()


def parse_args(argv):
    parser = argparse.ArgumentParser(description='valid model')
    parser.add_argument('--config_path', type=str, help='path to config path', default='configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    args = parse_args(sys.argv[1:])
    # logger.info(args)
    from retinaface.data.generate_data import GenerateData

    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data = gd.get_train_data()
    model = RetinaFace(ResNet_v1_50_FPN, num_class=2, anchor_per_scale=6)
    au = AnchorUtil(config)

    # import os
    # ckpt_dir = os.path.expanduser(config['ckpt_dir'])
    # ckpt = tf.train.Checkpoint(backbone=model.backbone)
    # ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
    # print("Restored from {}".format(tf.train.latest_checkpoint(ckpt_dir)))

    vd = ValidData(model, train_data, au)
    p, r = vd.get_metric(0.6, 0.2, 100)
    print(p, r)
    vd.draw_curve(0.2, 100)


if __name__ == '__main__':
    main()

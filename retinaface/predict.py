from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys

import numpy as np
import yaml

from retinaface.backbones.resnet_v1_fpn import ResNet_v1_50_FPN
from retinaface.models.models import RetinaFace
from retinaface.utils.anchor import AnchorUtil
from retinaface.utils.nms import nms


def predict(model, images, au, conf_thresh, iou_thresh, top_k):
    classes, boxes, lmks = model(images, training=False)  # shape=[(N, 160, 160, 12),(1/2),(1/4),(1/8),(1/16)],[],[]
    # 1. according anchor update boxes and lmks
    boxes = au.decode_box(boxes)
    lmks = au.decode_lmk(lmks)

    preds = None
    for i, cls in enumerate(classes):
        # score = np.reshape(score, (score.shape[0], score.shape[1], score.shape[2], -1, 2))
        box = boxes[i]
        lmk = lmks[i]
        pred = np.concatenate((cls, box, lmk), axis=-1)
        pred = np.reshape(pred, (pred.shape[0], -1, pred.shape[-1]))
        preds = np.concatenate((preds, pred), axis=1) if preds is not None else pred

    dets = []
    for i in range(preds.shape[0]):
        # for each img of batch
        pred = preds[i, :, :]
        # 2. according thresh to exclude
        idx = np.where(pred[:, 0] >= conf_thresh)
        pred = pred[idx]
        # 3. according nms to exclude
        idx = nms(pred, iou_thresh)[:top_k]
        pred = pred[idx]
        dets.append(pred)
    return dets


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train face network')
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

    for img, _ in train_data.take(1):
        dets = predict(model, img, au, 0.6, 0.2, 100)


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

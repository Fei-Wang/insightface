from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys

import yaml

from retinaface.backbones.resnet_v1_fpn import ResNet_v1_50_FPN
from retinaface.models.models import RetinaFace
from retinaface.utils.anchor import AnchorUtil
import numpy as np


def predict(model, images, au, conf_thresh, nms_thresh):
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
    print(preds.shape, type(preds))
    # 2. according thresh to exclude
    # idx = np.where(preds[:, :, 0] > conf_thresh)

    # 3. according nms to exclude
    return preds


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
    conf_thresh = config['conf_thresh']
    nms_thresh = config['nms_thresh']
    for img, _ in train_data.take(1):
        preds = predict(model, img, au, conf_thresh, nms_thresh)


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

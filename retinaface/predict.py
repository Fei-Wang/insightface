from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys

import numpy as np
import yaml

from retinaface.backbones.resnet_v1_fpn import ResNet_v1_50_FPN
from retinaface.models.models import RetinaFace
from retinaface.utils.anchor import AnchorUtil
from retinaface.utils.box import box_filter


def predict(model, images, au):
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
    import cv2
    for img, label, path in train_data.take(1):
        preds = predict(model, img, au)
        dets = box_filter(preds, 0.6, 0.2, 100)

        print(img.shape)
        print(label.bounding_shape())
        print(path)
        print(dets[0].shape)

        path = str(path[0].numpy(), encoding='utf-8')
        ori_img = cv2.imread(path)
        ori_h = ori_img.shape[0]
        ori_w = ori_img.shape[1]
        img = img[0].numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = label[0]
        # boxes = dets[0][2:, :]
        for box in boxes:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            for i in range(5):
                cv2.circle(img, (int(box[4 + 2 * i]), int(box[4 + 2 * i + 1])), 2, (0, 255, 0), -1)

            ori_box = []
            for i in range(7):
                x_ = box[2 * i] * max(ori_h, ori_w) / config['image_size'] - max((ori_h - ori_w) / 2, 0)
                y_ = box[2 * i + 1] * max(ori_h, ori_w) / config['image_size'] - max((ori_w - ori_h) / 2, 0)
                ori_box.append(x_)
                ori_box.append(y_)

            cv2.rectangle(ori_img, (int(ori_box[0]), int(ori_box[1])), (int(ori_box[2]), int(ori_box[3])), (0, 0, 255),
                          2)
            for i in range(5):
                cv2.circle(ori_img, (int(ori_box[4 + 2 * i]), int(ori_box[4 + 2 * i + 1])), 2, (0, 255, 0), -1)

        # fig = plt.figure()
        # plt.subplot(211)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.subplot(212)
        # plt.axis('off')
        # plt.imshow(ori_img)
        # plt.show()
        cv2.imshow('img', img)
        cv2.waitKey()
        cv2.imshow('ori_img', ori_img)
        cv2.waitKey()


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

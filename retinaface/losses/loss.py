from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from retinaface.utils.box import cal_iou

tf.enable_eager_execution()

epsilon = 1e-6  # in case log0


def _match_gt_anchor(gt, anchor):
    # find global optimizer
    idx = np.zeros(gt.shape[0]) - 1  # -1 represent not match, in case gt > anchor
    ious = np.zeros((gt.shape[0], anchor.shape[0]))
    for i in range(gt.shape[0]):
        ious[i] = cal_iou(gt[i], anchor[:, 2:])
    num = 0
    while num < min(gt.shape[0], anchor.shape[0]):
        loc_gt, loc_anchor = tf.unravel_index(np.argmax(ious), ious.shape)
        idx[loc_gt] = loc_anchor
        ious[loc_gt, :] = -1
        ious[:, loc_anchor] = -1
        # print(loc_gt, loc_anchor)
        num += 1
    return idx


def _smooth_l1(x):
    if abs(x) < 1:
        return 0.5 * x ** 2
    else:
        return abs(x) - 0.5


def _cal_loc_smooth_l1_loss(label, pred):
    loss = 0
    for i in range(label.shape[0]):
        loss += _smooth_l1(label[i] - pred[i])
    return loss


def _cal_pos_anchor_loss(gt, anchor, img_size=640, lambda1=0.25, lambda2=0.1, lambda3=0.01):
    cls_loss = -tf.math.log(anchor[0])
    norm_gt = gt / img_size
    norm_anchor = anchor[2:] / img_size
    # the diff between loss(x,y,w,h) and loss(x1,y1,x2,y2) is small, and change from one to another is easy
    # here use (x1,y1,x2,y2)
    box_loss = _cal_loc_smooth_l1_loss(norm_gt[:4], norm_anchor[:4])
    lmk_loss = _cal_loc_smooth_l1_loss(norm_gt[4:], norm_anchor[4:])
    pixel_loss = 0
    loss = cls_loss + lambda1 * box_loss + lambda2 * lmk_loss + lambda3 * pixel_loss
    return loss


def _cal_loss_per_image_per_scale(pred, labels, stride, img_size=640, lambda1=0.25, lambda2=0.1, lambda3=0.01):
    losses = -tf.math.log(pred[..., 1])
    label_dict = {}
    for label in labels:
        # for each gt
        # label = label.numpy()
        t_x = int((label[0] + label[2]) / 2 / stride)
        t_y = int((label[1] + label[3]) / 2 / stride)

        label = tf.reshape(label, (1, -1))
        label_dict[(t_x, t_y)] = tf.concat((label_dict[(t_x, t_y)], label), axis=0) \
            if (t_x, t_y) in label_dict else label

    # for loc, label in label_dict.items():
    #     idxes = _match_gt_anchor(label, pred[loc[0], loc[1]])
    #     for i, idx in enumerate(idxes):
    #         if idx >= 0:
    #             # change anchor (loc[0], loc[1], idx) loss value
    #             losses[loc[0], loc[1], int(idx)] = _cal_pos_anchor_loss(label[i], pred[loc[0], loc[1], int(idx)],
    #                                                                     img_size=img_size, lambda1=lambda1,
    #                                                                     lambda2=lambda2, lambda3=lambda3)

    loss = tf.reduce_mean(losses)
    # print(pred.shape, labels.shape, labels[0].shape)
    return loss


def _cal_loss_per_scale(pred, label, stride, img_size=640, lambda1=0.25, lambda2=0.1, lambda3=0.01):
    losses = 0
    for i in range(label.shape[0]):
        # for each image of each scale
        loss = _cal_loss_per_image_per_scale(pred[i], label[i], stride, img_size=img_size, lambda1=lambda1,
                                             lambda2=lambda2, lambda3=lambda3)
        losses += loss

    losses /= int(label.shape[0])
    return losses


def cal_loss(classes, boxes, lmks, label, strides, img_size=640, lambda1=0.25, lambda2=0.1, lambda3=0.01):
    losses = 0
    for i, cls in enumerate(classes):
        # for each scale
        box = boxes[i]
        lmk = lmks[i]
        # pred = tf.concat((cls, box, lmk), axis=-1)
        pred = tf.concat((cls, box, lmk), axis=-1)
        stride = strides[i]
        loss = _cal_loss_per_scale(pred, label, stride, img_size=img_size, lambda1=lambda1, lambda2=lambda2,
                                   lambda3=lambda3)
        losses += loss

    # losses /= len(classes)
    return losses


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description='define losses.')
    parser.add_argument('--config_path', type=str, help='path to config path', default='../configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    import sys
    args = parse_args(sys.argv[1:])
    # logger.info(args)
    from retinaface.data.generate_data import GenerateData
    from retinaface.backbones.resnet_v1_fpn import ResNet_v1_50_FPN
    from retinaface.models.models import RetinaFace
    from retinaface.utils.anchor import AnchorUtil

    import yaml
    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data = gd.get_train_data()
    model = RetinaFace(ResNet_v1_50_FPN, num_class=2, anchor_per_scale=6)
    au = AnchorUtil(config)

    for img, label, _ in train_data.take(1):
        classes, boxes, lmks = model(img, training=True)  # shape=[(N, 160, 160, 12),(1/2),(1/4),(1/8),(1/16)],[],[]
        boxes = au.decode_box(boxes)
        lmks = au.decode_lmk(lmks)

        loss = cal_loss(classes, boxes, lmks, label, config['feat_strides'])
        print(loss)


if __name__ == '__main__':
    main()

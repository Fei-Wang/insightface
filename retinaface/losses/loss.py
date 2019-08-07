from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from retinaface.utils.box import cal_iou

tf.enable_eager_execution()


class LossUtil:
    def __init__(self, config):
        self.strides = config['feat_strides']
        self.img_size = config['image_size']
        self.lambda1 = config['lambda1']
        self.lambda2 = config['lambda2']
        self.lambda3 = config['lambda3']

    @staticmethod
    def _match_gt_anchor(gt, anchor):
        # find global optimizer
        idx = np.zeros(gt.shape[0]) - 1  # -1 represent not match, in case gt > anchor
        ious = np.zeros((gt.shape[0], anchor.shape[0]))
        for i in range(gt.shape[0]):
            ious[i] = cal_iou(gt[i], anchor[:, 2:])
        num = 0
        while num < min(gt.shape[0], anchor.shape[0]):
            loc_gt, loc_anchor = np.unravel_index(np.argmax(ious), ious.shape)
            idx[loc_gt] = loc_anchor
            ious[loc_gt, :] = -1
            ious[:, loc_anchor] = -1
            num += 1
        return idx

    def _decode_label(self, preds, all_labels):
        gts = []
        for k, pred in enumerate(preds):
            gt = np.zeros(pred.shape, dtype=np.float32)
            gt[..., 1] = 1  # set to false
            stride = self.strides[k]
            label_dict = {}
            for i, labels in enumerate(all_labels):
                # for each image
                for label in labels:
                    # for each label
                    t_x = int((label[0] + label[2]) / 2 / stride)
                    t_y = int((label[1] + label[3]) / 2 / stride)

                    label = np.reshape(label, (1, -1))
                    label_dict[(i, t_x, t_y)] = np.concatenate((label_dict[(i, t_x, t_y)], label), axis=0) \
                        if (i, t_x, t_y) in label_dict else label

            for loc, label in label_dict.items():
                idxes = self._match_gt_anchor(label, pred[loc[0], loc[1], loc[2]])
                for i, idx in enumerate(idxes):
                    if idx >= 0:
                        # change anchor (loc[0], loc[1], loc[2], idx) loss value
                        gt[loc[0], loc[1], loc[2], int(idx), 0] = 1
                        gt[loc[0], loc[1], loc[2], int(idx), 1] = 0
                        gt[loc[0], loc[1], loc[2], int(idx), 2:] = label[i]
            gts.append(gt)
        return gts

    @staticmethod
    def _smooth_l1_loss(label, pred):
        ae = tf.abs(label - pred)
        loss = tf.where(ae < 1, 0.5 * ae ** 2, ae - 0.5)
        loss = tf.reduce_sum(loss, axis=-1)
        return loss

    def cal_loss(self, preds, labels):
        gts = self._decode_label(preds, labels)
        cls_losses = 0
        box_losses = 0
        lmk_losses = 0
        pix_lossses = 0
        for i, gt in enumerate(gts):
            # for each scale
            pred = preds[i]
            zeros = tf.zeros_like(gt[..., 0])
            norm_gt = tf.concat((gt[..., :2], gt[..., 2:] / self.img_size), axis=-1)
            norm_pred = tf.concat((pred[..., :2], pred[..., 2:] / self.img_size), axis=-1)
            # cls loss part
            cce = tf.keras.losses.CategoricalCrossentropy()
            cls_loss = cce(norm_gt[..., :2], norm_pred[..., :2])
            # box loss part
            box_loss = self._smooth_l1_loss(norm_gt[..., 2:6], norm_pred[..., 2:6])
            box_loss = tf.where(gt[..., 0], box_loss, zeros)
            box_loss = tf.reduce_mean(box_loss)
            # lmk loss part
            lmk_loss = self._smooth_l1_loss(norm_gt[..., 6:], norm_pred[..., 6:])
            lmk_loss = tf.where(gt[..., 0], lmk_loss, zeros)
            lmk_loss = tf.reduce_mean(lmk_loss)

            cls_losses += cls_loss
            box_losses += box_loss
            lmk_losses += lmk_loss

        losses = cls_losses + self.lambda1 * box_losses + self.lambda2 * lmk_losses + self.lambda3 * pix_lossses
        return losses, cls_losses, box_losses, lmk_losses, pix_lossses


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
    lu = LossUtil(config)
    for img, label, _ in train_data.take(1):
        classes, boxes, lmks = model(img, training=True)  # shape=[(N, 160, 160, 12),(1/2),(1/4),(1/8),(1/16)],[],[]
        boxes = au.decode_box(boxes)
        lmks = au.decode_lmk(lmks)
        preds = [tf.concat((classes[i], boxes[i], lmks[i]), axis=-1) for i in range(len(classes))]

        loss = lu.cal_loss(preds, label)

        print(loss)


if __name__ == '__main__':
    main()

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys
from itertools import product

import numpy as np
import tensorflow as tf
import yaml

tf.enable_eager_execution()


class AnchorUtil:
    def __init__(self, config):
        self.anchor_type = config['anchor_type']
        self.base_anchors = config['base_anchors']
        self.ratios = config['anchor_ratios']
        self.strides = config['feat_strides']
        self.img_size = config['image_size']
        self.anchors = self._generate_anchors()

    @staticmethod
    def _trans_wh_xy(box):
        """
            trans (cx, cy, w, h) to (x1, y1, x2, y2)
        :param box:
        :return: box
        """
        # m = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]])
        # box = np.dot(box, m)
        m = tf.constant([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]])
        box = tf.matmul(box, m)
        return box

    def _limit_boundary(self, obj):
        """
            limit the boundary to [0, img_size)
        :param obj:
        :return: obj
        """
        # obj = np.where(obj < 0, 0, obj)
        # obj = np.where(obj >= self.img_size, self.img_size - 1, obj)
        zeros = tf.zeros_like(obj)
        obj = tf.where(obj < zeros, zeros, obj)
        ones = tf.ones_like(obj)
        size = self.img_size * ones
        obj = tf.where(obj >= size, size - 1, obj)
        return obj

    def decode_box(self, boxes):
        """
        :param boxes:
        :type boxes: list
        :type boxes[0]: EagerTensor, shape:[N, H, W, A, 4]
        :return: boxes which is applyed anchor
        """
        ret_boxes = []
        for k, anchor in enumerate(self.anchors):
            box = boxes[k]  # .numpy()
            # box = np.reshape(box, (box.shape[0], box.shape[1], box.shape[2], -1, 4))

            if self.anchor_type == 'faster-rcnn':
                # box[:, :, :, :, :2] = anchor[:, :, :, :2] + box[:, :, :, :, :2] * anchor[:, :, :, 2:]
                # box[..., :2] = anchor[..., :2] + box[..., :2] * anchor[..., 2:]
                box_xy = anchor[..., :2] + box[..., :2] * anchor[..., 2:]
            elif self.anchor_type == 'yolo':
                # box[..., :2] = anchor[..., :2] + self.strides[k] / (1 + tf.exp(-box[..., :2]))
                box_xy = anchor[..., :2] + self.strides[k] / (1 + tf.exp(-box[..., :2]))
            else:
                raise ValueError('Invalid Anchor Type!')

            # box[..., 2:] = anchor[..., 2:] * tf.exp(box[..., 2:])
            box_wh = anchor[..., 2:] * tf.exp(box[..., 2:])
            box = tf.concat([box_xy, box_wh], axis=-1)
            # trans (cx, cy, w, h) to (x1, y1, x2, y2)
            box = self._trans_wh_xy(box)
            # limit the boundary
            box = self._limit_boundary(box)
            ret_boxes.append(box)

        return ret_boxes

    def decode_lmk(self, lmks):
        ret_lmks = []
        for k, anchor in enumerate(self.anchors):
            lmk = lmks[k]  # .numpy()
            # lmk = np.reshape(lmk, (lmk.shape[0], lmk.shape[1], lmk.shape[2], -1, 10))
            lmk_part = None
            for i in range(5):
                if self.anchor_type == 'faster-rcnn':
                    # lmk[..., 2 * i:2 + 2 * i] = anchor[..., :2] + lmk[..., 2 * i:2 + 2 * i] * anchor[..., 2:]
                    lmk_temp = anchor[..., :2] + lmk[..., 2 * i:2 + 2 * i] * anchor[..., 2:]
                elif self.anchor_type == 'yolo':
                    # lmk[..., 2 * i:2 + 2 * i] = anchor[..., :2] + self.strides[k] / (
                    #         1 + np.exp(-lmk[..., 2 * i:2 + 2 * i]))
                    lmk_temp = anchor[..., :2] + self.strides[k] / (1 + tf.exp(-lmk[..., 2 * i:2 + 2 * i]))
                else:
                    raise ValueError('Invalid Anchor Type!')
                lmk_part = tf.concat([lmk_part, lmk_temp], axis=-1) if lmk_part is not None else lmk_temp
            # limit the boundary
            lmk = self._limit_boundary(lmk_part)
            ret_lmks.append(lmk)

        return ret_lmks

    @staticmethod
    def _make_anchor(w, h, pw, ph, s):
        anchor = np.zeros((h, w, 4))
        anchor[:, :, 0] = (np.arange(w) + 0.5) * s
        anchor[:, :, 1] = (np.arange(h).reshape(-1, 1) + 0.5) * s
        anchor[:, :, 2] = pw
        anchor[:, :, 3] = ph
        return anchor

    def _generate_anchors(self):
        """
        generate anchors, size response to origin img size,
           return a list which include some scales,
           each scale is a 4-D numpy array,
           0-D: feature map height,
           1-D: feature map width,
           2-D: anchor_num_per_scale,
           3-D: 4 number(cx, cy, pw, py)
        """

        all_anchors = []  # include some feature maps (in this case is 5)
        for k, base_anchor in enumerate(self.base_anchors):
            stride = self.strides[k]
            feat_size = self.img_size // stride

            # anchors = None  # anchors of a feature maps
            anchors = np.zeros((feat_size, feat_size, len(base_anchor) * len(self.ratios), 4))
            for i, (base, ratio) in enumerate(product(base_anchor, self.ratios)):
                pw = base / np.sqrt(ratio)
                ph = base * np.sqrt(ratio)
                anchor = self._make_anchor(feat_size, feat_size, pw, ph, stride)  # anchor of a size of a feature maps

                anchors[:, :, i, :] = anchor
                # anchors = np.concatenate((anchors, anchor), axis=-1) if anchors is not None else anchor

            all_anchors.append(anchors)
        return all_anchors


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Get anchor')
    parser.add_argument('--config_path', type=str, help='path to config path', default='../configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    args = parse_args(sys.argv[1:])
    # logger.info(args)

    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)

    au = AnchorUtil(config)


if __name__ == '__main__':
    main()

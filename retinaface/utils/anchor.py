from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys
from itertools import product
from math import sqrt

import numpy as np
import yaml


class AnchorUtil:
    def __init__(self, config):
        self.anchor_type = config['anchor_type']
        self.base_anchors = config['base_anchors']
        self.ratios = config['anchor_ratios']
        self.strides = config['feat_strides']
        self.img_size = config['image_size']
        self.anchors = self._generate_anchors()

    def decode_box(self, boxes):
        """
        :param boxes:
        :type boxes: list
        :type boxes[0]: EagerTensor, shape:[N, H, W, A, 4]
        :return: boxes which is applyed anchor
        """
        ret_boxes = []
        for k, anchor in enumerate(self.anchors):
            box = boxes[k].numpy()
            # box = np.reshape(box, (box.shape[0], box.shape[1], box.shape[2], -1, 4))

            if self.anchor_type == 'faster-rcnn':
                # box[:, :, :, :, :2] = anchor[:, :, :, :2] + box[:, :, :, :, :2] * anchor[:, :, :, 2:]
                box[..., :2] = anchor[..., :2] + box[..., :2] * anchor[..., 2:]
            elif self.anchor_type == 'yolo':
                box[..., :2] = anchor[..., :2] + self.strides[k] / (1 + np.exp(-box[..., :2]))
            else:
                raise ValueError('Invalid Anchor Type!')

            box[..., 2:] = anchor[..., 2:] * np.exp(box[..., 2:])

            ret_boxes.append(box)

        return ret_boxes

    def decode_lmk(self, lmks):
        ret_lmks = []
        for k, anchor in enumerate(self.anchors):
            lmk = lmks[k].numpy()
            # lmk = np.reshape(lmk, (lmk.shape[0], lmk.shape[1], lmk.shape[2], -1, 10))
            for i in range(5):
                if self.anchor_type == 'faster-rcnn':
                    lmk[..., 2 * i:2 + 2 * i] = anchor[..., :2] + lmk[..., 2 * i:2 + 2 * i] * anchor[..., 2:]

                elif self.anchor_type == 'yolo':
                    lmk[..., 2 * i:2 + 2 * i] = anchor[..., :2] + self.strides[k] / (
                            1 + np.exp(-lmk[..., 2 * i:2 + 2 * i]))
                else:
                    raise ValueError('Invalid Anchor Type!')

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
                pw = base / sqrt(ratio)
                ph = base * sqrt(ratio)
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

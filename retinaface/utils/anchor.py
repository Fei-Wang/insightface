from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys
from itertools import product
import yaml
import numpy as np
from math import sqrt


def _make_anchor(w, h, pw, ph, s):
    anchor = np.zeros((h, w, 4))
    anchor[:, :, 0] = (np.arange(w) + 0.5) * s
    anchor[:, :, 1] = (np.reshape(np.arange(h), (h, 1)) + 0.5) * s
    anchor[:, :, 2] = pw
    anchor[:, :, 3] = ph

    return anchor


def generate_anchors(config):
    """
    generate anchors, size response to origin img size,
       return a list which include some layers,
       each layer is a 3-D numpy array,
       0-D: feature map height,
       1-D: feature map width,
       2-D: 4(cx, cy, pw, py) * anchor_num_per_layer.
    """

    base_anchors = config['base_anchors']
    ratios = config['anchor_ratios']
    strides = config['feat_strides']

    all_anchors = []  # include some feature maps (in this case is 5)
    for k, base_anchor in enumerate(base_anchors):
        stride = strides[k]
        feat_size = config['image_size'] // stride

        anchors = None  # anchors of a feature maps
        for base, ratio in product(base_anchor, ratios):
            pw = base / sqrt(ratio)
            ph = base * sqrt(ratio)
            anchor = _make_anchor(feat_size, feat_size, pw, ph, stride)  # anchor of a size of a feature maps

            anchors = np.concatenate((anchors, anchor), axis=-1) if anchors is not None else anchor

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

    generate_anchors(config)


if __name__ == '__main__':
    main()

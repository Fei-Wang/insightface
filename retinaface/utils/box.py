from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


def cal_iou(box, boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    xx1 = np.maximum(box[0], x1)
    yy1 = np.maximum(box[1], y1)
    xx2 = np.minimum(box[2], x2)
    yy2 = np.minimum(box[3], y2)

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    iou = inter / (areas + area - inter)
    return iou


def _nms(dets, thresh, mode="Union"):
    scores = dets[:, 0]
    x1 = dets[:, 2]
    y1 = dets[:, 3]
    x2 = dets[:, 4]
    y2 = dets[:, 5]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # [:top_k]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        # keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def box_filter(preds, conf_thresh, iou_thresh, top_k):
    dets = []
    for i in range(preds.shape[0]):
        # for each img of batch
        pred = preds[i, :, :]
        # 2. according thresh to exclude
        idx = np.where(pred[:, 0] >= conf_thresh)
        pred = pred[idx]
        # 3. according nms to exclude
        idx = _nms(pred, iou_thresh)[:top_k]
        pred = pred[idx]
        dets.append(pred)
    return dets

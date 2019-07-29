from __future__ import absolute_import, division, print_function, unicode_literals

import math

import tensorflow as tf

tf.enable_eager_execution()


def softmax_loss(dense, labels):
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # do softmax
    loss = cce(labels, dense)

    return loss


def arcface_loss(x, normx_cos, labels, m1, m2, m3, s):
    norm_x = tf.norm(x, axis=1, keepdims=True)
    cos_theta = normx_cos / norm_x
    theta = tf.acos(cos_theta)
    mask = tf.one_hot(labels, depth=normx_cos.shape[-1])
    zeros = tf.zeros_like(mask)
    cond = tf.where(tf.greater(theta * m1 + m3, math.pi), zeros, mask)
    cond = tf.cast(cond, dtype=tf.bool)
    m1_theta_plus_m3 = tf.where(cond, theta * m1 + m3, theta)
    cos_m1_theta_plus_m3 = tf.cos(m1_theta_plus_m3)
    prelogits = tf.where(cond, cos_m1_theta_plus_m3 - m2, cos_m1_theta_plus_m3) * s

    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # do softmax
    loss = cce(labels, prelogits)

    return loss


def triplet_loss(anchor_emb, pos_emb, neg_emb, alpha):
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor_emb, pos_emb)), axis=1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor_emb, neg_emb)), axis=1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), axis=0)
    return loss


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
    from recognition.data.generate_data import GenerateData
    from recognition.backbones.resnet_v1 import ResNet_v1_50
    from recognition.models.models import MyModel
    import yaml
    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data, classes = gd.get_train_data()

    model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'], classes=classes)

    for img, label in train_data.take(1):
        prelogits, dense, norm_dense = model(img, training=False)
        sm_loss = softmax_loss(dense, label)
        norm_sm_loss = softmax_loss(norm_dense, label)

        arc_loss = arcface_loss(prelogits, norm_dense, label, config['logits_margin1'], config['logits_margin2'],
                                config['logits_margin3'], config['logits_scale'])

        # embeddings = tf.nn.l2_normalize(prelogits, axis=1)
        # tf.reduce_mean(tf.abs(real_image - cycled_image))
        # tf.add_n()
        print(sm_loss, norm_sm_loss, arc_loss)


if __name__ == '__main__':
    # log_cfg_path = '../../logging.yaml'
    # with open(log_cfg_path, 'r') as f:
    #     dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    # logging.config.dictConfig(dict_cfg)
    # logger = logging.getLogger("mylogger")
    # logger.info("hello, insightface/recognition")
    main()

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys
import time

import tensorflow as tf
import yaml

from backbones.resnet_v1 import ResNet_v1_50
from data.generate_data import GenerateData
from losses.logit_loss import softmax_loss, arcface_loss
from models.models import MyModel

tf.enable_eager_execution()


# log_cfg_path = '../logging.yaml'
# with open(log_cfg_path, 'r') as f:
#     dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
# logging.config.dictConfig(dict_cfg)
# logger = logging.getLogger("mylogger")


# @tf.function
def train_step(model, img, label, config):
    with tf.GradientTape(persistent=False) as tape:
        prelogits, dense, norm_dense = model(img, training=True)
        sm_loss = softmax_loss(dense, label)
        norm_sm_loss = softmax_loss(norm_dense, label)
        arc_loss = arcface_loss(prelogits, norm_dense, label, config['logits_margin1'], config['logits_margin2'],
                                config['logits_margin3'], config['logits_scale'])
    gradients = tape.gradient(arc_loss, model.trainable_variables)
    optimizer = tf.train.AdamOptimizer()
    # Apply the gradients to the optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return arc_loss


def train(model, train_data, config):
    for epoch in range(config['epoch_num']):
        start = time.time()

        for step, (input_image, target) in enumerate(train_data):
            loss = train_step(model, input_image, target, config)
            print('epoch: {}, step: {}, loss = {}'.format(epoch, step, loss))
        # valid
        # for inp, tar in test_dataset.take(1):
        #     generate_images(generator, inp, tar)

        # saving (checkpoint) the model every 20 epochs
        # if (epoch + 1) % 20 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)
        #     ckpt_save_path = ckpt_manager.save()
        #     print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
        #                                                         ckpt_save_path))

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--config_path', type=str, help='path to config path', default='configs/config.yaml')

    args = parser.parse_args(argv)

    return args


def main():
    args = parse_args(sys.argv[1:])
    # logger.info(args)

    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    train_data, cat_num = gd.get_train_data()
    model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'], classes=cat_num)

    train(model, train_data, config)


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
import time

import tensorflow as tf
import yaml

from backbones.resnet_v1 import ResNet_v1_50
from data.generate_data import GenerateData
from losses.logit_loss import softmax_loss, arcface_loss
from models.models import MyModel
from valid import Valid_Data

tf.enable_eager_execution()


# log_cfg_path = '../logging.yaml'
# with open(log_cfg_path, 'r') as f:
#     dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
# logging.config.dictConfig(dict_cfg)
# logger = logging.getLogger("mylogger")

class Trainer:
    def __init__(self, model, config, train_data, val_data=None):
        self.model = model
        self.epoch_num = config['epoch_num']
        self.m1 = config['logits_margin1']
        self.m2 = config['logits_margin2']
        self.m3 = config['logits_margin3']
        self.s = config['logits_scale']
        self.train_data = train_data
        self.thresh = config['thresh']
        self.optimizer = tf.keras.optimizers.Adam(0.001)

        ckpt_dir = os.path.expanduser(config['ckpt_dir'])
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=5, checkpoint_name='mymodel')

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        self.vd = None
        if val_data is not None:
            self.vd = Valid_Data(model, val_data)

    # @tf.function
    def __train_step(self, img, label):
        with tf.GradientTape(persistent=False) as tape:
            prelogits, dense, norm_dense = self.model(img, training=True)
            sm_loss = softmax_loss(dense, label)
            norm_sm_loss = softmax_loss(norm_dense, label)
            arc_loss = arcface_loss(prelogits, norm_dense, label, self.m1, self.m2, self.m3, self.s)
        gradients = tape.gradient(arc_loss, self.model.trainable_variables)
        # Apply the gradients to the optimizer
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return arc_loss

    def train(self):
        for epoch in range(self.epoch_num):
            start = time.time()

            for step, (input_image, target) in enumerate(self.train_data):
                loss = self.__train_step(input_image, target)
                print('epoch: {}, step: {}, loss = {}'.format(epoch, step, loss))
                # valid
                if self.vd is not None:
                    acc, p, r, fpr = self.vd.get_metric(self.thresh)
                    print('epoch: {}, acc: {:.3f}, p: {:.3f}, r=tpr: {:.3f}, fpr: {:.3f}'.format(epoch, acc, p, r, fpr))

                # if epoch % 5 == 0:
                save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch, save_path))

            print('Time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))


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
    valid_data = gd.get_val_data(config['valid_num'])
    model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'], classes=cat_num)

    t = Trainer(model, config, train_data, valid_data)
    t.train()


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

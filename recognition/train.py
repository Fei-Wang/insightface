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
        self.vd = None
        if val_data:
            self.vd = Valid_Data(model, val_data)

    # @tf.function
    def __train_step(self, img, label):
        with tf.GradientTape(persistent=False) as tape:
            prelogits, dense, norm_dense = self.model(img, training=True)
            sm_loss = softmax_loss(dense, label)
            norm_sm_loss = softmax_loss(norm_dense, label)
            arc_loss = arcface_loss(prelogits, norm_dense, label, self.m1, self.m2, self.m3, self.s)
        gradients = tape.gradient(arc_loss, self.model.trainable_variables)
        optimizer = tf.train.AdamOptimizer()
        # Apply the gradients to the optimizer
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

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

            # checkpoint_path = "./checkpoints/train"
            #
            # ckpt = tf.train.Checkpoint(generator_g=generator_g,
            #                            generator_f=generator_f,
            #                            discriminator_x=discriminator_x,
            #                            discriminator_y=discriminator_y,
            #                            generator_g_optimizer=generator_g_optimizer,
            #                            generator_f_optimizer=generator_f_optimizer,
            #                            discriminator_x_optimizer=discriminator_x_optimizer,
            #                            discriminator_y_optimizer=discriminator_y_optimizer)
            #
            # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
            #
            # # if a checkpoint exists, restore the latest checkpoint.
            # if ckpt_manager.latest_checkpoint:
            #     ckpt.restore(ckpt_manager.latest_checkpoint)
            #     print('Latest checkpoint restored!!')
            #
            # if (epoch + 1) % 5 == 0:
            #     ckpt_save_path = ckpt_manager.save()
            #     print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
            #                                                         ckpt_save_path))
            #
            #
            #
            # checkpoint_dir = './training_checkpoints'
            # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            # checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
            #                                  discriminator_optimizer=discriminator_optimizer,
            #                                  generator=generator,
            #                                  discriminator=discriminator)
            #
            # # Save the model every 15 epochs
            # if (epoch + 1) % 15 == 0:
            #     checkpoint.save(file_prefix=checkpoint_prefix)
            #
            # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            #
            # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
            # model.save_weights('path_to_my_weights', save_format='tf')
            #
            # model.load_weights(checkpoint_path)
            # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
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
    valid_data = gd.get_val_data(config['valid_num'])
    model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'], classes=cat_num)

    t = Trainer(model, config, train_data, valid_data)
    t.train()


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

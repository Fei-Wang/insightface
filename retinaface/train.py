from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
import os
import platform
import sys
import time

import tensorflow as tf
import yaml

from retinaface.backbones.resnet_v1_fpn import ResNet_v1_50_FPN
from retinaface.data.generate_data import GenerateData
from retinaface.losses.loss import LossUtil
from retinaface.models.models import RetinaFace
from retinaface.utils.anchor import AnchorUtil

# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# tf.enable_eager_execution(config=config)

tf.enable_eager_execution()


# log_cfg_path = '../logging.yaml'
# with open(log_cfg_path, 'r') as f:
#     dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
# logging.config.dictConfig(dict_cfg)
# logger = logging.getLogger("mylogger")

class Trainer:
    def __init__(self, config):
        self.gd = GenerateData(config)
        self.train_data = self.gd.get_train_data()
        # valid_data = self.gd.get_val_data(config['valid_num'])
        anchor_per_scale = len(config['base_anchors'][0]) * len(config['anchor_ratios'])
        self.model = RetinaFace(ResNet_v1_50_FPN, num_class=config['num_class'], anchor_per_scale=anchor_per_scale)
        self.au = AnchorUtil(config)
        self.lu = LossUtil(config)
        self.feat_strides = config['feat_strides']
        self.image_size = config['image_size']
        self.lambda1 = config['lambda1']
        self.lambda2 = config['lambda2']
        self.lambda3 = config['lambda3']

        self.epoch_num = config['epoch_num']
        self.learning_rate = config['learning_rate']

        optimizer = config['optimizer']
        if optimizer == 'ADADELTA':
            self.optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)
        elif optimizer == 'ADAGRAD':
            self.optimizer = tf.keras.optimizers.Adagrad(self.learning_rate)
        elif optimizer == 'ADAM':
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        elif optimizer == 'ADAMAX':
            self.optimizer = tf.keras.optimizers.Adamax(self.learning_rate)
        elif optimizer == 'FTRL':
            self.optimizer = tf.keras.optimizers.Ftrl(self.learning_rate)
        elif optimizer == 'NADAM':
            self.optimizer = tf.keras.optimizers.Nadam(self.learning_rate)
        elif optimizer == 'RMSPROP':
            self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        else:
            raise ValueError('Invalid optimization algorithm')

        ckpt_dir = os.path.expanduser(config['ckpt_dir'])

        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=5, checkpoint_name='mymodel')

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        # self.vd = Valid_Data(self.model, valid_data)

        summary_dir = os.path.expanduser(config['summary_dir'])
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(summary_dir, current_time, 'train')
        valid_log_dir = os.path.join(summary_dir, current_time, 'valid')

        if platform.system() == 'Windows':
            train_log_dir = train_log_dir.replace('/', '\\')
            valid_log_dir = valid_log_dir.replace('/', '\\')

        # self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
        self.train_summary_writer = tf.compat.v2.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.compat.v2.summary.create_file_writer(valid_log_dir)

    # @tf.function
    def _train_step(self, img, label):
        with tf.GradientTape(persistent=False) as tape:
            classes, boxes, lmks = self.model(img, training=True)
            boxes = self.au.decode_box(boxes)
            lmks = self.au.decode_lmk(lmks)
            preds = [tf.concat((classes[i], boxes[i], lmks[i]), axis=-1) for i in range(len(classes))]
            loss, cls_loss, box_loss, lmk_loss, pix_losss = self.lu.cal_loss(preds, label)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, cls_loss, box_loss, lmk_loss, pix_losss

    def train(self):
        for epoch in range(self.epoch_num):
            start = time.time()

            for step, (input_image, target, _) in enumerate(self.train_data):
                loss, cls_loss, box_loss, lmk_loss, pix_losss = self._train_step(input_image, target)
                with self.train_summary_writer.as_default():
                    tf.compat.v2.summary.scalar('loss', loss, step=step)
                    tf.compat.v2.summary.scalar('cls_loss', cls_loss, step=step)
                    tf.compat.v2.summary.scalar('box_loss', box_loss, step=step)
                    tf.compat.v2.summary.scalar('lmk_loss', lmk_loss, step=step)
                    tf.compat.v2.summary.scalar('pix_losss', pix_losss, step=step)

                print('epoch: {}, step: {}, loss = {}, cls_loss = {}, box_loss = {}, lmk_loss = {}, pix_loss = {}'
                      .format(epoch, step, loss, cls_loss, box_loss, lmk_loss, pix_losss))

            # valid
            # acc, p, r, fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr = self.vd.get_metric(self.thresh, self.below_fpr)

            # with self.valid_summary_writer.as_default():
            #     tf.compat.v2.summary.scalar('acc', acc, step=epoch)
            #     tf.compat.v2.summary.scalar('p', p, step=epoch)
            #     tf.compat.v2.summary.scalar('r=tpr', r, step=epoch)
            #     tf.compat.v2.summary.scalar('fpr', fpr, step=epoch)
            #     tf.compat.v2.summary.scalar('acc_fpr', acc_fpr, step=epoch)
            #     tf.compat.v2.summary.scalar('p_fpr', p_fpr, step=epoch)
            #     tf.compat.v2.summary.scalar('r=tpr_fpr', r_fpr, step=epoch)
            #     tf.compat.v2.summary.scalar('thresh_fpr', thresh_fpr, step=epoch)
            # print('epoch: {}, acc: {:.3f}, p: {:.3f}, r=tpr: {:.3f}, fpr: {:.3f} \n'
            #       'fix fpr <= {}, acc: {:.3f}, p: {:.3f}, r=tpr: {:.3f}, thresh: {:.3f}'
            #       .format(epoch, acc, p, r, fpr, self.below_fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr))

            # ckpt
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

    t = Trainer(config)
    t.train()


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

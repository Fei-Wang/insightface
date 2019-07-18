from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
import os
import platform
import sys
import time

import tensorflow as tf
import yaml

from backbones.resnet_v1 import ResNet_v1_50
from data.generate_data import GenerateData
from losses.logit_loss import softmax_loss, arcface_loss
from models.models import MyModel
from valid import Valid_Data

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
    def __init__(self, model, config, train_data, val_data=None):
        self.model = model
        self.epoch_num = config['epoch_num']
        self.m1 = config['logits_margin1']
        self.m2 = config['logits_margin2']
        self.m3 = config['logits_margin3']
        self.s = config['logits_scale']
        self.train_data = train_data
        self.thresh = config['thresh']
        self.below_fpr = config['below_fpr']
        self.optimizer = tf.keras.optimizers.Adam(0.001)

        ckpt_dir = os.path.expanduser(config['ckpt_dir'])
        self.ckpt = tf.train.Checkpoint(backbone=self.model.backbone, model=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=5, checkpoint_name='mymodel')

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        self.vd = None
        if val_data is not None:
            self.vd = Valid_Data(model, val_data)

        summary_dir = os.path.expanduser(config['summary_dir'])
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(summary_dir, current_time, 'train')
        valid_log_dir = os.path.join(summary_dir, current_time, 'valid')
        # self.graph_log_dir = os.path.join(summary_dir, current_time, 'graph')

        if platform.system() == 'Windows':
            train_log_dir = train_log_dir.replace('/', '\\')
            valid_log_dir = valid_log_dir.replace('/', '\\')
            # self.graph_log_dir = self.graph_log_dir.replace('/', '\\')

        # self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
        self.train_summary_writer = tf.compat.v2.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.compat.v2.summary.create_file_writer(valid_log_dir)

        # self.graph_writer = tf.compat.v2.summary.create_file_writer(self.graph_log_dir)
        # tf.compat.v2.summary.trace_on(graph=True, profiler=True)
        # with graph_writer.as_default():
        #     tf.compat.v2.summary.trace_export(name="graph_trace", step=0, profiler_outdir=graph_log_dir)

    # @tf.function
    def __train_step(self, img, label):
        with tf.GradientTape(persistent=False) as tape:
            prelogits, dense, norm_dense = self.model(img, training=True)
            embs = tf.nn.l2_normalize(prelogits, axis=-1)
            for i in range(embs.shape[0]):
                for j in range(embs.shape[0]):
                    val = 0
                    for k in range(512):
                        val += embs[i][k] * embs[j][k]
                    print(i, j, val)
            print(tf.argmax(dense, axis=-1))
            print(label)
            sm_loss = softmax_loss(dense, label)
            norm_sm_loss = softmax_loss(norm_dense, label)
            arc_loss = arcface_loss(prelogits, norm_dense, label, self.m1, self.m2, self.m3, self.s)
            loss = sm_loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def train(self):
        for epoch in range(self.epoch_num):
            start = time.time()

            for step, (input_image, target) in enumerate(self.train_data):
                loss = self.__train_step(input_image, target)

                with self.train_summary_writer.as_default():
                    tf.compat.v2.summary.scalar('loss', loss, step=step)
                print('epoch: {}, step: {}, loss = {}'.format(epoch, step, loss))

                # valid
                if self.vd is not None:
                    acc, p, r, fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr = self.vd.get_metric(self.thresh, self.below_fpr)

                    with self.valid_summary_writer.as_default():
                        tf.compat.v2.summary.scalar('acc', acc, step=step)
                        tf.compat.v2.summary.scalar('p', p, step=step)
                        tf.compat.v2.summary.scalar('r=tpr', r, step=step)
                        tf.compat.v2.summary.scalar('fpr', fpr, step=step)
                        tf.compat.v2.summary.scalar('acc_fpr', acc_fpr, step=step)
                        tf.compat.v2.summary.scalar('p_fpr', p_fpr, step=step)
                        tf.compat.v2.summary.scalar('r=tpr_fpr', r_fpr, step=step)
                        tf.compat.v2.summary.scalar('thresh_fpr', thresh_fpr, step=step)
                    print('epoch: {}, acc: {:.3f}, p: {:.3f}, r=tpr: {:.3f}, fpr: {:.3f} \n'
                          'fix fpr <= {}, acc: {:.3f}, p: {:.3f}, r=tpr: {:.3f}, thresh: {:.3f}'
                          .format(epoch, acc, p, r, fpr, self.below_fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr))

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
    gd = GenerateData(config)
    train_data, cat_num = gd.get_train_data()
    valid_data = gd.get_val_data(config['valid_num'])
    model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'], classes=cat_num)

    t = Trainer(model, config, train_data, valid_data)
    t.train()


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()

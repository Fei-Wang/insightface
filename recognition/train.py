from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import sys
import argparse
from logger import logger

tf.enable_eager_execution()


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train face network')

    args = parser.parse_args(argv)

    return args


def train_net(args=None):
    pass


def main():
    args = parse_args(sys.argv[1:])
    logger.info(args)
    train_net(args)


if __name__ == '__main__':
    logger.info("hello, insightface/recognition")
    main()

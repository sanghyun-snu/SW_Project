# -*- coding: utf-8 -*-

import argparse
import architecture as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--log-folder', type=str, default='train_log')
    parser.add_argument('--data', metavar='DIR',
                        default='/workspace',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total'
                             'batch size of all GPUs on the current node when'
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='use pre-trained model')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')

    # CASE NAME
    parser.add_argument('--name', type=str, default='test_case')
    parser.add_argument('-e', '--evaluate', type=str2bool, nargs='?', const=True, default=False,
                        help='evaluate model on validation set')

    # path
    parser.add_argument('--dataset', type=str, default='PASCAL', )
    parser.add_argument('--train-list', type=str, default='./datalist/PascalVOC/train_aug.txt')
    parser.add_argument('--val-list', type=str, default='./datalist/PascalVOC/val.txt')
    parser.add_argument('--test-list', type=str, default='./datalist/PascalVOC/test.txt')
    parser.add_argument('--size-list', type=str, default='./datalist/PascalVOC/sizes.txt')

    # basic hyperparameter
    parser.add_argument('--LR-decay', type=int, default=30, help='Reducing lr frequency')
    parser.add_argument('--lr-ratio', type=float, default=10)
    parser.add_argument('--nest', type=str2bool, nargs='?', const=True, default=False)


    # Hide and Seek
    parser.add_argument('--HaS-grid-size', type=int, default=4)
    parser.add_argument('--HaS-drop-rate', type=float, default=0.5)

    # CAM
    parser.add_argument('--cam-thr', type=float, default=0.2, help='cam threshold value')

    # data transform
    parser.add_argument('--resize-size', type=int, default=321, help='input resize size')
    parser.add_argument('--crop-size', type=int, default=321, help='input crop size')
    args = parser.parse_args()

    return args

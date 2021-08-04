""" Throughput Measuring Script
Copyright 2021 Shuyang Sun
"""
import time
import yaml
import torch
import models
import argparse

import timm.models
from timm.models import create_model


def get_args():
    parser = argparse.ArgumentParser('vis')
    parser.add_argument('--config', default=None)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--test_iter', default=50)
    parser.add_argument('--test_batch_size', default=128)
    parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                        help='Name of model to train (default: "countception"')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
    parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
    parser.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('--drop', type=float, default=0.1, metavar='PCT',
                    help='Dropout rate (default: 0.1)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
    
    args_config, remaining = parser.parse_known_args()
    
    if args_config.config is not None:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    
    args = parser.parse_args(remaining, namespace=args_config)
    
    return args


def test_speed(args):
    model_name = args.model_name
    data_shape = (args.test_batch_size, 3, 224, 224)
    if model_name.endswith('_384'):
        data_shape = (args.test_batch_size, 3, 384, 384)
    if model_name == 'efficientnet_b3':
        data_shape = (args.test_batch_size, 3, 300, 300)
    test_data = torch.zeros(data_shape).cuda()

    model = create_model(model_name)

    model.cuda()
    model.eval()

    all_run_time = 0.
    with torch.no_grad():
        for i in range(args.test_iter):
            start = time.time()
            model(test_data)
            tmp_run_time = time.time() - start
            all_run_time += tmp_run_time

    speed = args.test_batch_size * args.test_iter / all_run_time
    msg = model_name + '\t' + str(speed) + '\n-----------------'

    print(msg, flush=True)
        

if __name__ == '__main__':
    args = get_args()
    test_speed(args)

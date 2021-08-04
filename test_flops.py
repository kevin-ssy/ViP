"""FLOPS/Params Measuring Script
Copyright 2021 Shuyang Sun
"""
import yaml
import torch
import models
import argparse

import timm.models
from timm.models import create_model

from utils.flop_count.flop_count import flop_count


parser = argparse.ArgumentParser('vis')
parser.add_argument('--config', default=None)
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




def main(args):
    test_data = torch.zeros((1, 3, 224, 224))#.cuda()
    model_name = args.model
    model = create_model(model_name,
                         drop_rate=args.drop,
                         drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
                         drop_path_rate=args.drop_path,
                         drop_block_rate=args.drop_block)#.cuda()

    flop_dict, _ = flop_count(model, (test_data,))

    msg = model_name + '\t' + str(sum(flop_dict.values())) + '\t params:' + str(
        sum([m.numel() for m in model.parameters()])) + '\n-----------------'

    print(msg)


if __name__ == '__main__':
    args_config, remaining = parser.parse_known_args()

    if args_config.config is not None:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining, namespace=args_config)
    main(args)

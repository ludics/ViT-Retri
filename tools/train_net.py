# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import os
import time

from datetime import timedelta

import torch
import torch.distributed as dist

from vit_retri.utils.utils import initial_logger

from vit_retri.engine import setup
from vit_retri.engine import set_seed
from vit_retri.engine import train

def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds",
                                              "INat2017", "cifar10", "cifar100"],
                        default="CUB_200_2011",
                        help="Which downstream task.")
    parser.add_argument('--data_root', type=str, default='/opt/tiger/minist')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--save_iter", default=1000, type=int,
                        help="Resolution size")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--optim_type", choices=["SGD", "AdamW"], default="SGD",
                        help="How to optimize parameters.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--master_port", type=str, default="23456",
                        help="master_port for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    
    parser.add_argument('--use_xbm', action='store_true',
                        help="Whether to use XBM")
    parser.add_argument('--xbm_size', type=int, default=5000,
                        help="XBM queue size")
    parser.add_argument('--xbm_start_step', type=int, default=1000,
                        help="XBM queue size")
    return parser.parse_args()

def main():
    args = parse_args()
    # Setup CUDA, GPU & distributed training
    os.environ["MASTER_PORT"] = args.master_port
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.output_dir = os.path.join(args.output_dir, args.name)
    # Setup logging
    log_path = os.path.join("logs", args.name)
    os.makedirs(log_path, exist_ok=True)
    log_file_name = os.path.join(log_path, time.strftime("%m-%d-%H:%M", time.localtime()) + '.log')
    logger = initial_logger(log_file_name)

    #logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    #                    #filename=log_file_name, filemode='w',
    #                    datefmt='%m/%d/%Y %H:%M:%S',
    #                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args, logger)

    # Training
    train(args, model, logger)


if __name__ == "__main__":
    main()

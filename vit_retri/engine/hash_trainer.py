# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import time

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from vit_retri.models import VisionTransformer, CONFIGS, DSHNet
from vit_retri.utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from vit_retri.utils.data_utils import get_loader, get_disjoint_loader
from vit_retri.utils.dist_util import get_world_size
from vit_retri.models.contrastive_loss import ContrastiveLoss
from vit_retri.models.hash_loss import DCHLoss
from vit_retri.models.xbm import XBM
from vit_retri.utils.feat_extractor import feat_extractor, code_generator
from vit_retri.evaluations.eval import AccuracyCalculator
from vit_retri.evaluations.ret_metric import RetMetric
from vit_retri.utils.log_info import log_info
from vit_retri.utils.utils import initial_logger
from vit_retri.utils.utils import AverageMeter
from vit_retri.utils.tools import CalcTopMap, pr_curve


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, logger, step=-1, best=False):
    model_to_save = model.module if hasattr(model, 'module') else model
    if best:
        model_checkpoint = os.path.join(args.output_dir, f"{args.name}_best.ckpt")
    else:
        model_checkpoint = os.path.join(args.output_dir, f"{args.name}_step{step}.ckpt")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args, logger):
    # Prepare model
    config = CONFIGS[args.model_type]
    args.hidden_size = config.hidden_size

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    elif args.dataset == "cifar10":
        num_classes = 10 
    else:
        num_classes = 100

    model = DSHNet(args)
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def flush_log(writer, iteration):
    for k, v in log_info.items():
        if isinstance(v, np.ndarray):
            writer.add_histogram(k, v, iteration)
        else:
            writer.add_scalar(k, v, iteration)
    for k in list(log_info.keys()):
        del log_info[k]

best_mapr = 0
best_iter = -1

def valid(args, model, writer, test_loader, database_loader, global_step, logger):
    # Validation!
    global best_mapr
    global best_iter
    print("\n")
    logger.info("***** Running Validation *****")
    logger.info(f"  Num steps = {len(test_loader)}, Batch size = {args.eval_batch_size}")

    model.eval()
    test_codes, test_labels = code_generator(model, test_loader, logger)
    test_labels_onehot = torch.eye(200)[test_labels]
    gallery_codes, gallery_labels = code_generator(model, database_loader, logger)
    gallery_labels_onehot = torch.eye(200)[gallery_labels]
    log_info["mAP"] = CalcTopMap(gallery_codes, test_codes,
                                 gallery_labels_onehot.numpy(), test_labels_onehot.numpy(), 10000)
    logger.info(f"test size: {test_codes.shape[0]}")
    logger.info(f"gallery size: {gallery_codes.shape[0]}")
    map_curr = log_info["mAP"]
    pr_range = [10, 20, 40, 80]
    codes = [gallery_codes, test_codes]
    labels = [gallery_labels.numpy(), test_labels.numpy()]
    r_k_func = RetMetric(codes, labels, hamming_dis=True)
    # P, R = pr_curve(codes.numpy(), codes.numpy(), labels.numpy(), labels.numpy(), pr_range)
    for i, k in enumerate(pr_range):
        log_info[f"R@{k}"] = r_k_func.recall_k(k)
    if map_curr > best_mapr:
        best_mapr = map_curr
        best_iter = global_step
        logger.info(f"Best iteration {global_step}:")
    else:
        logger.info(f"Performance at iteration {global_step:06d}:")
    for k, v in log_info.items():
        logger.info(f"{k}: {v}")
    flush_log(writer, global_step)
    return map_curr


def train(args, model, logger):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    if args.dataset_joint:
        train_loader, test_loader, database_loader = get_loader(args)
        # database_loader = train_loader
    else:
        train_loader, test_loader = get_disjoint_loader(args)
        database_loader = test_loader
    # Prepare optimizer and scheduler
    if args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    
    criterion = DCHLoss(args, args.hash_bit)
    if args.use_xbm:
        logger.info(">>> use XBM")
        xbm = XBM(len(train_loader.dataset), args.hidden_size)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    c_losses = AverageMeter()
    q_losses = AverageMeter()
    global_step, best_acc = 0, 0
    if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
        mapr_curr = valid(args, model, writer, test_loader, database_loader, global_step, logger)
        if best_acc < mapr_curr:
            save_model(args, model, logger, best=True)
            best_acc = mapr_curr
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            codes  = model(x)
            labels = torch.eye(200)[y].to(args.device)
            if args.use_xbm and global_step > args.xbm_start_step:
                xbm.enqueue_dequeue(codes.detach(), y.detach())

            c_loss, q_loss = criterion(codes, labels)
            loss = c_loss + args.lambd * q_loss

            if args.use_xbm and global_step > args.xbm_start_step:
                xbm_feats, xbm_targets = xbm.get()
                xbm_loss = criterion(codes, labels)
                log_info["xbm_loss"] = xbm_loss.item()
                loss = loss + xbm_loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                c_losses.update(c_loss.item()*args.gradient_accumulation_steps)
                q_losses.update(q_loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f) (lr=%.6f)" % (global_step, t_total, losses.val, scheduler.get_lr()[0])
                )
                log_info["lr"] = scheduler.get_lr()[0]
                log_info["loss"] = losses.val
                log_info["c_loss"] = c_losses.val
                log_info["q_loss"] = q_losses.val
                if args.local_rank in [-1, 0]:
                    flush_log(writer, global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    mapr_curr = valid(args, model, writer, test_loader, database_loader, global_step, logger)
                    if best_acc < mapr_curr:
                        save_model(args, model, logger, best=True)
                        best_acc = mapr_curr
                    if global_step % args.save_iter == 0:
                        save_model(args, model, logger, step=global_step)
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        c_losses.reset()
        q_losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best MAPR: \t%f" % best_acc)
    logger.info("End Training!")


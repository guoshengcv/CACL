"""Finetune 3D CNN."""
import os
import argparse
import itertools
import time
import math
import random
import builtins
import warnings
import string
import numpy as np
import pandas as pd
from PIL import ImageFilter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datasets.ucf101 import UCF101Dataset_Classify
# from datasets.hmdb51 import HMDB51Dataset
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    ft_lr =args.ft_lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        ft_lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
            ft_lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        optimizer.param_groups[0]['lr'] = lr
        # optimizer.param_groups[1]['lr'] = ft_lr

def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    for name, params in pretrained_weights.items():
        if 'module' in name and 'base_network' in name and 'encoder_q' in name:
            print(name)
            name = name.split('base_network.')[1]
            adjusted_weights[name] = params
            print('Pretrained weight name: [{}]'.format(name))
#         if 'module' and 'base_network' in name:
#             name = name.split('base_network.')[1]
# #             name = name.replace('base_network.','')
#             adjusted_weights[name] = params
#             print('Pretrained weight name: [{}]'.format(name))
        # if 'base_network' in name:
        #     name = name[name.find('.')+1:]
        #     adjusted_weights[name] = params
        #     print('Pretrained weight name: [{}]'.format(name))
    return adjusted_weights


def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch,lr,ft_lr):
    torch.set_grad_enabled(True)
    model.train()

    running_loss = 0.0
    correct = 0
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        clips, idxs = data
        inputs = clips.to(device)
        targets = idxs.to(device)-1
        # forward and backward
        outputs = model(inputs)  # return logits here
        loss = criterion(outputs, targets)
        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # compute loss and acc
        running_loss += loss
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts)
        correct = correct.type(torch.float)

        torch.distributed.barrier()
        # print statistics and write summary every N batch
        if i % args.pf == 0:
            avg_loss = running_loss / args.pf
            avg_acc = correct / (args.pf * args.bs)
            reduced_avg_loss = reduce_mean(avg_loss, args.ngpus_per_node)
            reduced_avg_acc = reduce_mean(avg_acc, args.ngpus_per_node)
            print('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f},lr:{:.6f},ft_lr:{:.6f}'.format(epoch, i, reduced_avg_loss, reduced_avg_acc,lr,ft_lr))
            step = (epoch-1)*len(train_dataloader) + i
            if args.gpu == 0:
                writer.add_scalar('train/CrossEntropyLoss', reduced_avg_loss.item(), step)
                writer.add_scalar('train/Accuracy', reduced_avg_acc.item(), step)
            running_loss = 0.0
            correct = 0
    # summary params and grads per eopch
    for name, param in model.named_parameters():
        if args.gpu == 0:
            writer.add_histogram('params/{}'.format(name), param, epoch)
            writer.add_histogram('grads/{}'.format(name), param.grad, epoch)


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader):
        # get inputs
        clips, idxs = data
        inputs = clips.to(device)
        targets = idxs.to(device)-1
        # forward
        # forward
        outputs = model(inputs)  # return logits here
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts)
        correct = correct.type(torch.float)
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / (i + 1)
    avg_acc = correct / (i + 1) / args.bs
    torch.distributed.barrier()
    reduced_avg_loss = reduce_mean(avg_loss, args.ngpus_per_node)
    reduced_avg_acc = reduce_mean(avg_acc, args.ngpus_per_node)
    if args.gpu == 0:
        writer.add_scalar('val/CrossEntropyLoss', reduced_avg_loss.item(), epoch)
        writer.add_scalar('val/Accuracy', reduced_avg_acc.item(), epoch)
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(reduced_avg_loss, reduced_avg_acc))
    return reduced_avg_loss


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        sampled_clips, idxs = data
        targets = idxs.to(device)-1
        # forward
        # forward
        #outputs = model(inputs)
        outputs = []
        for clips in sampled_clips:
            inputs = clips.to(device)
            # forward
            o = model(inputs)
            # print(o.shape)
            o = torch.mean(o, dim=0)
            # print(o.shape)
            # exit()
            outputs.append(o)
        outputs = torch.stack(outputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss
        pts = torch.argmax(outputs, dim=1)
        # print(pts, targets)
        correct += torch.sum(targets == pts)
        #correct = correct.type(torch.float)
        # print('correct: {}, {}, {}'.format(correct, targets, pts))

    torch.distributed.barrier()
    reduced_sum_loss = reduce_sum(total_loss)
    reduced_sum_acc = reduce_sum(correct)

    print(len(test_dataloader), len(test_dataloader.dataset))
    reduced_avg_loss = reduced_sum_loss / len(test_dataloader)
    reduced_sum_acc = reduced_sum_acc.type(torch.float)
    reduced_avg_acc = reduced_sum_acc / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(reduced_avg_loss, reduced_avg_acc))
    return reduced_avg_loss, reduced_avg_acc


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def parse_args():
    parser = argparse.ArgumentParser(description='Video Clip Order Prediction')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='c3d', help='c3d/r3d/r21d')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--split', type=str, default='1', help='dataset split')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--ft_lr', type=float, default=1e-3, help='finetune learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--schedule', default=[120,150], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--epochs', type=int, default=180, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=None, help='seed for initializing training.')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    cudnn.benchmark = True

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    print(vars(args))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.dataset == 'ucf101':
        class_num = 101
    elif args.dataset == 'hmdb51':
        class_num = 51

    if args.model == 'c3d':
        model = C3D(with_classifier=True, num_classes=class_num)
    elif args.model == 'r3d':
        model = R3DNet(layer_sizes=(3,4,6,3), with_classifier=True, num_classes=class_num)
    elif args.model == 'r21d':
        model = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=True, num_classes=class_num)

    if args.mode == 'train':  # ########## Train #############
        if args.ckpt:  # resume training
            pretrained_weights = load_pretrained_weights(args.ckpt)
            print(model)
            load_result = model.load_state_dict(pretrained_weights, strict=False)
            print(load_result)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.bs = int(args.bs / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # vcopn.cuda()
        # # DistributedDataParallel will divide and allocate batch_size to all
        # # available GPUs if device_ids are not set
        # vcopn = torch.nn.parallel.DistributedDataParallel(vcopn)


        # log_dir = os.path.dirname(args.log)
        # print(log_dir)

        # else:
    writer = None
    if args.mode == 'train':
        if args.desp:
            exp_name = '{}_cl{}_{}_{}'.format(args.model, args.cl, args.desp, time.strftime('%m%d%H%M'))
        else:
            exp_name = '{}_cl{}_{}'.format(args.model, args.cl, time.strftime('%m%d%H%M'))
        log_dir = os.path.join(args.log, exp_name)
        print(log_dir)
        if args.gpu == 0:
            writer = SummaryWriter(log_dir)

        train_transforms = transforms.Compose([
            transforms.Resize((128, 171)),  # smaller edge to 128
            transforms.RandomCrop(112),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        if args.dataset == 'ucf101':
            train_dataset = UCF101Dataset_Classify('data/ucf101', args.cl, args.split, True, train_transforms)
            val_size = 800
        elif args.dataset == 'hmdb51':
            train_dataset = UCF101Dataset_Classify('data/hmdb51', args.cl, args.split, True, train_transforms)
            val_size = 400
             # split val for 800 videos
        train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset) - val_size, val_size))
        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        # train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
        #                             num_workers=args.workers, pin_memory=True)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.bs, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        # val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
        #                             num_workers=args.workers, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.bs, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)



        # save graph and clips_order samples
        for data in train_dataloader:
            clips, idxs = data
            if args.gpu == 0:
                writer.add_video('train/clips', clips, fps=8)
                writer.add_text('train/idxs', str(idxs.tolist()))
            #  clips = clips
            clips = clips.to(args.gpu)
            if args.gpu == 0:
                writer.add_graph(model.module, clips)
            break
        # save init params at step 0
        for name, param in model.named_parameters():
            if args.gpu == 0:
                writer.add_histogram('params/{}'.format(name), param, 0)

        #  ## loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = optim.SGD([
            {'params': [param for name, param in model.named_parameters() if
                        'linear' not in name and 'conv5' not in name and 'conv4' not in name]},
            {'params': [param for name, param in model.named_parameters() if
                        'linear' in name or 'conv5' in name or 'conv4' in name], 'lr': args.ft_lr}],
            lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
      #  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)

        prev_best_val_loss = float('inf')
        prev_best_model_path = None
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            time_start = time.time()
            adjust_learning_rate(optimizer, epoch, args)
            lr = optimizer.param_groups[0]['lr']
            # ft_lr = optimizer.param_groups[1]['lr']
            ft_lr = lr
            train(args, model, criterion, optimizer, args.gpu, train_dataloader, writer, epoch, lr, ft_lr)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            val_loss = validate(args, model, criterion, args.gpu, val_dataloader, writer, epoch)
            # scheduler.step(val_loss)
            if args.gpu == 0:
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            # writer.add_scalar('train/ft_lr', optimizer.param_groups[1]['lr'], epoch)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                # save model every 20 epoches
                if epoch % 10 == 0 or epoch == args.epochs:
                    torch.save(model.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
                # save model for the best val
                if val_loss < prev_best_val_loss:
                    model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
                    torch.save(model.state_dict(), model_path)
                    prev_best_val_loss = val_loss
                    if prev_best_model_path:
                        os.remove(prev_best_model_path)
                    prev_best_model_path = model_path

            if epoch == args.start_epoch+args.epochs -1 or epoch % 10 == 0:
               # model.load_state_dict(torch.load(args.ckpt))
                test_transforms = transforms.Compose([
                    transforms.Resize((128, 171)),
                    transforms.CenterCrop(112),
                    transforms.ToTensor()
                ])
                if args.dataset == 'ucf101':
                    test_dataset = UCF101Dataset_Classify('data/ucf101', args.cl, args.split, False, test_transforms)
                elif args.dataset == 'hmdb51':
                    test_dataset = UCF101Dataset_Classify('data/hmdb51', args.cl, args.split, False, test_transforms)

                test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
                # test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                #                         num_workers=args.workers, pin_memory=True)
                test_dataloader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=args.bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True, sampler=test_sampler)

                print('TEST video number: {}.'.format(len(test_dataset)))
                criterion = nn.CrossEntropyLoss().cuda(args.gpu)
                _, reduced_avg_acc = test(args, model, criterion, args.gpu, test_dataloader)
                if args.gpu == 0:
                    writer.add_scalar('test/Accuracy', reduced_avg_acc.item(), epoch)

    #  ########## Test #############
    elif args.mode == 'test':
        model.load_state_dict(torch.load(args.ckpt), strict=True)
        test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()
        ])
        if args.dataset == 'ucf101':
            test_dataset = UCF101Dataset_Classify('data/ucf101', args.cl, args.split, False, test_transforms)
        elif args.dataset == 'hmdb51':
            test_dataset = UCF101Dataset_Classify('data/hmdb51', args.cl, args.split, False, test_transforms)

        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        # test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
        #                         num_workers=args.workers, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.bs, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=test_sampler)

        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        test(args, model, criterion, args.gpu, test_dataloader)


if __name__ == '__main__':
    main()

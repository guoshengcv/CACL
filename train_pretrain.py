"""Video clip order prediction."""
import os
import math
import builtins
import itertools
import argparse
import time
import random
import warnings
import string
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import ImageFilter
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datasets.ucf101 import UCF101CACLDataset
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet
from models.video_transformer import VideoTransformer
from models.wrapper_model import Wrapper_Model
from models.vmoco_with_transformer import MoCo_Transformer
import Levenshtein
import random

# def adjust_learning_rate(optimizer, epoch, args):
#     """Decay the learning rate based on schedule"""
#     lr = args.lr
#     for milestone in args.schedule:
#         lr *= 0.1 if epoch >= milestone else 1.
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, args, lr):
    """Decay the learning rate based on schedule"""
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def order_class_index(order):
    """Return the index of the order in its full permutation.

    Args:
        order (tensor): e.g. [0,1,2]
    """
    classes = list(itertools.permutations(list(range(len(order)))))
    return classes.index(tuple(order.tolist()))

def order_classs_index_le(order):
    con_order = list(range(len(order)))
    order = order.numpy().tolist()
    con_order_str = '_'.join('%s' %id for id in con_order)
    order_str = '_'.join('%s' %id for id in order)
    con_order_str=con_order_str.replace('_','')
    order_str=order_str.replace('_','')
    class_id = Levenshtein.distance(con_order_str,order_str)
   # class_id =  Levenshtein.ratio(con_order_str,order_str)
    if class_id==0:
        dis_id =0
    else:
        dis_id = class_id-1
    return  dis_id

def order_classs_index_lev2(order):
    order =order.numpy()
    all_string = string.ascii_lowercase
    refre_str = all_string[:len(order)]
    refre_str_list =list(refre_str)
    refre_str_list = np.asarray(refre_str_list)
    con_refre = refre_str_list.copy()
    refre  = con_refre[order]
    con_base = '_'.join('%s' %id for id in refre_str_list)
    con_new = '_'.join('%s' %id for id in refre)
    con_order_str=con_base.replace('_','')
    order_str=con_new.replace('_','')
    class_id = Levenshtein.distance(con_order_str,order_str)
   # class_id =  Levenshtein.ratio(con_order_str,order_str)
    if class_id==0:
        dis_id =0
    else:
        dis_id = class_id-1
    return  dis_id

def accuracy(similarities, num_pos):
    with torch.no_grad():
        pos_sim = similarities[:, :num_pos]
        neg_sim = similarities[:, num_pos:]
        neg_sim_max = torch.max(neg_sim, dim=1, keepdim=True)[0]
        accuracy = (pos_sim > neg_sim_max).to(torch.float32).mean()
        pos_sim_mean = pos_sim.mean(dim=0)
        return accuracy, pos_sim_mean

def similarity_cross_entropy(similarities, num_pos):
    # modified from vince/loss_util.py

    # assert mask.shape == similarities.shape
    # log similarity over (self + all other entries as denom)
    row_maxes = torch.max(similarities, dim=-1, keepdim=True)[0]
    scaled_similarities = similarities - row_maxes

    pos_similarities = scaled_similarities[:, :num_pos]
    neg_similarities = scaled_similarities[:, num_pos:]

    neg_similarities_exp = torch.exp(neg_similarities).sum(-1, keepdim=True)

    pos_similarities_exp = torch.exp(pos_similarities)
    similarity_log_softmax = pos_similarities - torch.log(pos_similarities_exp + neg_similarities_exp)
    dists = -similarity_log_softmax

    loss = dists.mean()
    return loss


def train(args, moco, criterion, optimizer, device, train_dataloader, writer, epoch, lr):
    torch.set_grad_enabled(True)
    moco.train()

    running_cls_loss = 0.0
    correct = 0
    running_moco_loss = 0.0
    running_moco_correct = 0
    total_pos_sim_mean = None
    for i, data in tqdm(enumerate(train_dataloader, 1)):
        # get inputs
        stacked_clip_q, stacked_clip_k, tuple_orders = data
        stacked_clip_q = stacked_clip_q.to(device)
        stacked_clip_k = stacked_clip_k.to(device)
        targets = [order_classs_index_lev2(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # zero the parameter gradients

        # forward and backward
        #loss = self.criterion(logits, labels)
        moco_logits, num_pos, h_outputs = moco(stacked_clip_q, stacked_clip_k) # return logits here
        moco_loss = similarity_cross_entropy(moco_logits, num_pos)
        moco_accuracy, pos_sim_mean = accuracy(moco_logits, num_pos)
        running_moco_loss += moco_loss
        cls_loss = criterion(h_outputs, targets)
        running_cls_loss +=cls_loss
        pts = torch.argmax(h_outputs, dim=1)
        correct += torch.sum(targets == pts)
        correct = correct.type(torch.float)
        running_moco_correct +=moco_accuracy
        if total_pos_sim_mean is None:
            total_pos_sim_mean = pos_sim_mean
        else:
            total_pos_sim_mean += pos_sim_mean
        total_loss = cls_loss * 0.0 + moco_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        torch.distributed.barrier()

        # print statistics and write summary every N batch
        if i % args.pf == 0:
            avg_loss = running_cls_loss / args.pf
            avg_acc = correct / (args.pf * args.bs)
            avg_moco_loss = running_moco_loss / args.pf
            avg_moco_acc = running_moco_correct / args.pf
            avg_pos_sim = total_pos_sim_mean / args.pf
            torch.distributed.barrier()
            reduced_avg_loss = reduce_mean(avg_loss, args.ngpus_per_node)
            reduced_avg_acc = reduce_mean(avg_acc, args.ngpus_per_node)
            reduced_moco_loss = reduce_mean(avg_moco_loss, args.ngpus_per_node)
            reduced_moco_acc = reduce_mean(avg_moco_acc, args.ngpus_per_node)
            reduced_avg_pos_sim = reduce_mean(avg_pos_sim, args.ngpus_per_node)
            print('[TRAIN] epoch-{}, batch-{}, clsloss: {:.3f}, clsacc: {:.3f}, mocoloss: {:.3f}, mocoacc: {:.3f},lr:{:.6f},'.format(epoch, i, reduced_avg_loss.item(),
                     reduced_avg_acc.item(),reduced_moco_loss.item(),reduced_moco_acc.item(),lr))
            step = (epoch-1)*len(train_dataloader) + i
            if args.gpu == 0:
                writer.add_scalar('train/CrossEntropyLoss', reduced_avg_loss.item(), step)
                writer.add_scalar('train/Accuracy', reduced_avg_acc.item(), step)
                writer.add_scalar('train/moco_loss', reduced_moco_loss.item(), step)
                writer.add_scalar('train/moco_acc', reduced_moco_acc.item(), step)
                for sim_idx, x in enumerate(reduced_avg_pos_sim):
                    writer.add_scalar(f'train/pos_sim{sim_idx}', x.item(), step)
            running_cls_loss = 0.0
            correct = 0
            running_moco_loss = 0.0
            running_moco_correct = 0
            total_pos_sim_mean = None

    # summary params and grads per eopch
    # for name, param in moco.named_parameters():
    #     writer.add_histogram('params/{}'.format(name), param, epoch)
    #     writer.add_histogram('grads/{}'.format(name), param.grad, epoch)

def validate(args, moco, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    moco.eval()

    running_cls_loss = 0.0
    correct = 0
    running_moco_loss = 0.0
    running_moco_correct = 0
    for i, data in enumerate(val_dataloader):
        # get inputs
        stacked_clip_q, stacked_clip_k, tuple_orders = data
        stacked_clip_q = stacked_clip_q.to(device)
        stacked_clip_k = stacked_clip_k.to(device)
        targets = [order_classs_index_lev2(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # zero the parameter gradient
        # forward and backward
        moco_logits, num_pos, h_outputs = moco(stacked_clip_q, stacked_clip_k)  # return logits here
        moco_loss = similarity_cross_entropy(moco_logits, num_pos)
        moco_accuracy, _ = accuracy(moco_logits, num_pos)
        running_moco_loss +=moco_loss
        cls_loss = criterion(h_outputs, targets)
        running_cls_loss +=cls_loss
        pts = torch.argmax(h_outputs, dim=1)
        correct += torch.sum(targets == pts)
        correct = correct.type(torch.float)
        running_moco_correct +=moco_accuracy
    avg_loss = running_cls_loss / (i + 1)
    avg_acc = correct / (i + 1) / args.bs
    avg_moco_loss = running_moco_loss / (i + 1)
    avg_moco_acc = running_moco_correct / (i + 1)
    torch.distributed.barrier()
    reduced_avg_loss = reduce_mean(avg_loss, args.ngpus_per_node)
    reduced_avg_acc = reduce_mean(avg_acc, args.ngpus_per_node)
    reduced_moco_loss = reduce_mean(avg_moco_loss, args.ngpus_per_node)
    reduced_moco_acc = reduce_mean(avg_moco_acc, args.ngpus_per_node)
    print('[VAL] epoch-{}, batch-{}, clsloss: {:.3f}, clsacc: {:.3f}, mocoloss: {:.3f}, mocoacc: {:.3f}'.format(epoch, i, reduced_avg_loss.item(),reduced_avg_acc.item(),reduced_moco_loss.item(),reduced_moco_acc.item()))

    if args.gpu == 0:
        writer.add_scalar('val/CrossEntropyLoss', reduced_avg_loss.item(), epoch)
        writer.add_scalar('val/Accuracy', reduced_avg_acc.item(), epoch)
        writer.add_scalar('val/moco_loss', reduced_moco_loss.item(), epoch)
        writer.add_scalar('val/moco_acc', reduced_moco_acc.item(), epoch)

    return reduced_avg_loss  #, reduced_moco_loss


@torch.no_grad()
def test(args, model, criterion, device, test_dataloader):
    print(len(test_dataloader))
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    total_pos_sim = None
    for i, data in tqdm(enumerate(test_dataloader)):
        # get inputs
        stacked_clip_q, stacked_clip_k, tuple_orders = data
        stacked_clip_q = stacked_clip_q.to(device)
        stacked_clip_k = stacked_clip_k.to(device)
        targets = [order_classs_index_lev2(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # zero the parameter gradient
        # forward and backward
        moco_logits, num_pos, h_outputs = model(stacked_clip_q, stacked_clip_k)  # return logits here
        moco_loss = similarity_cross_entropy(moco_logits, num_pos)
        moco_accuracy, pos_sim = accuracy(moco_logits, num_pos)
        if total_pos_sim is None:
            total_pos_sim = pos_sim
        else:
            total_pos_sim += pos_sim
    total_pos_sim *= 0.07
    torch.distributed.barrier()
    total_pos_sim = reduce_mean(total_pos_sim, args.ngpus_per_node)
    avg_pos_sim = total_pos_sim / len(test_dataloader)
    print(f"test avg_pos_sim {avg_pos_sim}")

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def parse_args():
    parser = argparse.ArgumentParser(description='Video Clip Order Prediction')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='c3d', help='c3d/r3d/r21d')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--it', type=int, default=8, help='interval')
    parser.add_argument('--tl', type=int, default=3, help='tuple length')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--log', type=str, help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    # parser.add_argument('--schedule', default=[100, 150], nargs='*', type=int,
    #                     help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=None, help='seed for initializing training.')
    parser.add_argument('--world-size', default=-1, type=int,help='number of nodes for distributed training')
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
    # parser.add_argument('--twod-lr', default=0.01, type=float,
    #                     metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', default=[150, 250], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    # parser.add_argument('--twod-momentum', default=0.9, type=float, metavar='M',
    #                     help='momentum of moco SGD solver')
    # parser.add_argument('--twod-wd', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=6400, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    # options for moco v2
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--aug-plus', action='store_true',
                        help='use moco v2 data augmentation')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
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

    ngpus_per_node =  torch.cuda.device_count()
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
        args.bs = int(args.bs / ngpus_per_node)
    ########### model ##############
    if args.model == 'c3d':
        base = C3D(with_classifier=False)
    elif args.model == 'r3d':
        # base = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
        base = R3DNet(layer_sizes=(3,4,6,3), with_classifier=False) # r3d-50
    elif args.model == 'r21d':
        base = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    cnn_base = Wrapper_Model(base_network=base, feature_size=512, tuple_len=args.cl)
    tf_base = VideoTransformer(depth=6, num_heads=6)
    moco = MoCo_Transformer(cnn_base, tf_base, args, args.moco_dim,args.moco_k, args.moco_m, args.moco_t, args.mlp)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        moco.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        #
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        moco = torch.nn.parallel.DistributedDataParallel(moco, device_ids=[args.gpu])
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set

    if args.mode == 'train':  ########### Train #############
        if args.ckpt:  # resume training
            moco.load_state_dict(torch.load(args.ckpt))
            log_dir = os.path.dirname(args.ckpt)
        else:
            if args.desp:
                exp_name = '{}_cl{}_it{}_{}_{}'.format(args.model, args.cl, args.it, args.desp, time.strftime('%m%d%H%M'))
            else:
                exp_name = '{}_cl{}_it{}_{}'.format(args.model, args.cl, args.it, time.strftime('%m%d%H%M'))
            log_dir = os.path.join(args.log, exp_name)
        writer = None
        if args.gpu == 0:
            writer = SummaryWriter(log_dir)

        moco_transforms = transforms.Compose([
            transforms.Resize((128, 171)),  # smaller edge to 128
            transforms.RandomCrop(112),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize
        ])

        train_dataset = UCF101CACLDataset('data/ucf101', args.cl, args.it, True, moco_transforms)
        # split val for 800 videos
        train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset)-800, 800))
        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.bs, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler,drop_last=True)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.bs, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)

        if args.ckpt:
            pass
        else:
            # save graph and clips_order samples
            for data in train_dataloader:
                tuple_clips, tuple_clips_re,tuple_orders = data
                for i in range(2):
                    if args.gpu == 0:
                        writer.add_video('train/tuple_clips_re', tuple_clips_re[:, i, :, :, :, :], i, fps=8)
                        writer.add_video('train/tuple_clips', tuple_clips[:, i, :, :, :, :], i, fps=8)
                        writer.add_text('train/tuple_orders',  str(tuple_orders[:, i].tolist()), i)
                tuple_clips = tuple_clips.to(args.gpu)
                tuple_clips_re = tuple_clips_re.to(args.gpu)
                # writer.add_graph(moco.module,(tuple_clips_ori,tuple_clips))
                break
            # save init params at step 0
            for name, param in moco.named_parameters():
                if args.gpu == 0:
                    writer.add_histogram('params/{}'.format(name), param, 0)

        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(moco.parameters(), args.lr, momentum=args.momentum,
                                         weight_decay=args.wd)
        prev_best_val_loss = float('inf')
        prev_best_model_path = None
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            time_start = time.time()
            adjust_learning_rate(optimizer, epoch, args, args.lr)
            lr = optimizer.param_groups[0]['lr']
            train(args, moco, criterion, optimizer, args.gpu, train_dataloader, writer, epoch,lr)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            val_loss = validate(args, moco, criterion, args.gpu, val_dataloader, writer, epoch)
            # scheduler.step(val_loss)
            if args.gpu == 0:
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            # save model every 20 epoches

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                # save model every 20 epoches
                if epoch % 10 == 0:
                    torch.save(moco.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
                # save model for the best val
                if val_loss < prev_best_val_loss:
                    model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
                    torch.save(moco.state_dict(), model_path)
                    prev_best_val_loss = val_loss
                    if prev_best_model_path:
                        os.remove(prev_best_model_path)
                    prev_best_model_path = model_path

    elif args.mode == 'test':  ########### Test #############
        moco.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu')))

        moco_transforms = transforms.Compose([
            transforms.Resize((128, 171)),  # smaller edge to 128
            transforms.RandomCrop(112),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize
        ])

        test_dataset = UCF101CACLDataset('data/ucf101', args.cl, args.it, False, moco_transforms)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.bs, shuffle=(test_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=test_sampler,drop_last=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        test(args, moco, criterion, args.gpu, test_dataloader)


if __name__ == '__main__':
    main()
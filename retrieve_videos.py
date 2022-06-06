"""Video retrieval experiment, top-k."""
import os
import math
import builtins
import itertools
import argparse
import time
import random
import json

from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from datasets.ucf101 import UCF101ClipRetrievalDataset
from datasets.hmdb51 import HMDB51ClipRetrievalDataset
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet


def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    for name, params in pretrained_weights.items():
        if 'module' in name and 'base_network' in name and 'encoder_q' in name:
            name = name.split('base_network.')[1]
            adjusted_weights[name] = params
            print('Pretrained weight name: [{}]'.format(name))
#         if 'module' and 'base_network' in name:
#             name = name.split('base_network.')[1]
#             adjusted_weights[name] = params
#             print('Pretrained weight name: [{}]'.format(name))
        # if 'base_network' in name:
        #     name = name[name.find('.')+1:]
        #     adjusted_weights[name] = params
        #     print('Pretrained weight name: [{}]'.format(name))
    return adjusted_weights


def extract_feature(gpu, ngpus_per_node, args):
    """Extract and save features for train split, several clips per video."""
    torch.backends.cudnn.benchmark = True
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

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
        model = C3D(with_classifier=False,retrival_test_p4=False,retrival_test=False, return_conv=True).to(args.gpu)
    elif args.model == 'r3d':
        model = R3DNet(layer_sizes=(3,4,6,3), with_classifier=False, return_conv=True).to(args.gpu)
    elif args.model == 'r21d':
        model = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False, return_conv=True).to(args.gpu)

    if args.ckpt:
        pretrained_weights = load_pretrained_weights(args.ckpt)
        load_result = model.load_state_dict(pretrained_weights, strict=True)
        print(load_result)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model.eval()
    torch.set_grad_enabled(False)
    ### Exract for train split ###
    train_transforms = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.ToTensor()
    ])
    if args.dataset == 'ucf101':
        train_dataset = UCF101ClipRetrievalDataset('data/ucf101', 16, 10, True, train_transforms)
    elif args.dataset == 'hmdb51':
        train_dataset = HMDB51ClipRetrievalDataset('data/hmdb51', 16, 10, True, train_transforms)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    features = []
    classes = []
    train_sampler.set_epoch(0)
    for data in tqdm(train_dataloader):
        sampled_clips, idxs = data
        clips = sampled_clips.reshape((-1, 3, 16, 112, 112))
        inputs = clips.to(args.gpu)
        idxs = idxs.to(args.gpu)
        # forward
        outputs = model(inputs)
        # print(outputs.shape)
        # exit()
        torch.distributed.barrier()
        outputs = concat_all_gather(outputs)
        idxs = concat_all_gather(idxs)
        if args.gpu == 0:
            features.append(outputs.cpu().numpy().tolist())
            classes.append(idxs.cpu().numpy().tolist())

    if args.gpu == 0:
        features = np.array(features).reshape(-1, 10, outputs.shape[1])
        classes = np.array(classes).reshape(-1, 10)
        np.save(os.path.join(args.feature_dir, 'train_feature.npy'), features)
        np.save(os.path.join(args.feature_dir, 'train_class.npy'), classes)

    ### Exract for test split ###
    test_transforms = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.ToTensor()
    ])
    if args.dataset == 'ucf101':
        test_dataset = UCF101ClipRetrievalDataset('data/ucf101', 16, 10, False, test_transforms)
    elif args.dataset == 'hmdb51':
        test_dataset = HMDB51ClipRetrievalDataset('data/hmdb51', 16, 10, False, test_transforms)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True, drop_last=True, sampler=test_sampler)

    features = []
    classes = []
#    test_sampler.set_epoch(0)
    for data in tqdm(test_dataloader):
        sampled_clips, idxs = data
        clips = sampled_clips.reshape((-1, 3, 16, 112, 112))
        inputs = clips.to(args.gpu)
        idxs = idxs.to(args.gpu)
        # forward
        outputs = model(inputs)
        torch.distributed.barrier()
        outputs = concat_all_gather(outputs)
        idxs = concat_all_gather(idxs)
        if args.gpu == 0:
            features.append(outputs.cpu().numpy().tolist())
            classes.append(idxs.cpu().numpy().tolist())

    if args.gpu == 0:
        features = np.array(features).reshape(-1, 10, outputs.shape[1])
        classes = np.array(classes).reshape(-1, 10)
        np.save(os.path.join(args.feature_dir, 'test_feature.npy'), features)
        np.save(os.path.join(args.feature_dir, 'test_class.npy'), classes)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def topk_retrieval(args):
    """Extract features from test split and search on train split features."""
    print('Load local .npy files.')
    X_train = np.load(os.path.join(args.feature_dir, 'train_feature.npy'))
    y_train = np.load(os.path.join(args.feature_dir, 'train_class.npy'))
    X_train = np.mean(X_train,1)
    y_train = y_train[:,0]
    X_train = X_train.reshape((-1, X_train.shape[-1]))
    y_train = y_train.reshape(-1)

    X_test = np.load(os.path.join(args.feature_dir, 'test_feature.npy'))
    y_test = np.load(os.path.join(args.feature_dir, 'test_class.npy'))
    X_test = np.mean(X_test,1)
    y_test = y_test[:,0]
    X_test = X_test.reshape((-1, X_test.shape[-1]))
    y_test = y_test.reshape(-1)

    ks = [1, 5, 10, 20, 50]
    topk_correct = {k:0 for k in ks}
    temp = defaultdict(int)
    all_d = defaultdict(int)`

    distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)

    for k in ks:
        # print(k)
        top_k_indices = indices[:, :k]
        # print(top_k_indices.shape, y_test.shape)
        for ind, test_label in zip(top_k_indices, y_test):
            labels = y_train[ind]
            if test_label in labels:
                # print(test_label, labels)
                topk_correct[k] += 1
                if k == 1:
                    temp[test_label] += 1
            all_d[test_label] += 1
    with open("all_retrieve.txt", "w") as f:
        for i in range(101):
            f.write(str(all_d[i]) + "\n")

    for k in ks:
        correct = topk_correct[k]
        total = len(X_test)
        print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct/total))

    with open(os.path.join(args.feature_dir, 'topk_correct.json'), 'w') as fp:
        json.dump(topk_correct, fp)


def parse_args():
    parser = argparse.ArgumentParser(description='Frame Retrieval Experiment')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--model', type=str, default='c3d', help='c3d/r3d/r21d')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--feature_dir', type=str, default='data/features/ucf101/c3d', help='dir to store feature.npy')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--bs', type=int, default=64, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=32, help='number of data loading workers')
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node

    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)
        if args.multiprocessing_distributed:
            args.world_size = ngpus_per_node * args.world_size
            mp.spawn(extract_feature, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    topk_retrieval(args)
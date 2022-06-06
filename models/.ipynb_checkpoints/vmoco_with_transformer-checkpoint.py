# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import scipy
import scipy.optimize
import numpy as np
import copy


class MoCo_Transformer(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    """
    def __init__(self, base_encoder, tf_base_encoder, args, dim=128, K=6400, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_Transformer, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.args = args
        self.enqueue_size = None

        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)
        self.encoder_transformer = tf_base_encoder
        self.encoder_transformer_k = copy.deepcopy(self.encoder_transformer)

        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        #     self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.encoder_transformer.parameters(), self.encoder_transformer_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.encoder_transformer.parameters(), self.encoder_transformer_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if self.enqueue_size is None:
            self.enqueue_size = batch_size

        ptr = int(self.queue_ptr)
        if batch_size != self.enqueue_size:  # for simplicity
            print("drop last???")
            return  # may drop last
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, clip_q, clip_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # B, num_video_per_instance(2), C, num_frames, H, W
        im_q = clip_q
        im_k = clip_k
        order_outputs, q = self.encoder_q(im_q)
        transformer_q = self.encoder_transformer(im_q)

        q = nn.functional.normalize(q, dim=1)
        transformer_q = nn.functional.normalize(transformer_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            order_outputs_k, k =self.encoder_k(im_k)
            momentum_transformer_k = self.encoder_transformer_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            momentum_transformer_k = nn.functional.normalize(momentum_transformer_k, dim=1)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            momentum_transformer_k = self._batch_unshuffle_ddp(momentum_transformer_k, idx_unshuffle)

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_pos_transformer = torch.einsum('nc,nc->n', [q, transformer_q]).unsqueeze(-1)
        l_pos_transformer_k = torch.einsum('nc,nc->n', [q, momentum_transformer_k]).unsqueeze(-1)
        l_pos_transformer_qk = torch.einsum('nc,nc->n', [transformer_q, momentum_transformer_k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_pos_transformer, l_pos_transformer_k, l_pos_transformer_qk, l_neg], dim=1)
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # labels[:, :self.num_frames - 1] = 1  # set label to 1

        # dequeue and enqueue
        if self.training:
            k = torch.cat([k, momentum_transformer_k], dim=0)
            self._dequeue_and_enqueue(k)
        #  loss = self.criterion(logits, labels)
        #  acc1, acc5 = self.accuracy(logits, labels, topk=(1, 5))

        return logits, 4, order_outputs

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

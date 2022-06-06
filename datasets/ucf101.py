"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile

from random import choice
import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import  math


class UCF101Dataset_Classify(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir, clip_len, split='1', train=True, transforms_=None, test_sample_num=10, interval=2):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        self.interval = interval

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split'+ self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        # random select a clip for train
        if self.train:
            if length - self.clip_len * self.interval > 0:
                clip_start = random.randint(0, length - self.clip_len * self.interval)
                clip = videodata[clip_start: clip_start + self.clip_len * self.interval - 1: self.interval]
            else:
                # video is short
                # print("video length too short")
                clip_start = random.randint(0, length - self.clip_len)
                clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)

            return clip, torch.tensor(int(class_idx))
        # sample several clips for test
        else:
            all_clips = []
            all_idx = []
            # for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.test_sample_num):
            #     clip_start = int(i - self.clip_len/2)
            #     clip = videodata[clip_start: clip_start + self.clip_len]
            for clip_start in np.linspace(0, length - (self.clip_len * self.interval), self.test_sample_num):
                clip_start = max(int(clip_start), 0)
                clip = videodata[clip_start: clip_start + self.clip_len * self.interval - 1: self.interval]
                if len(clip) < self.clip_len:
                    # video is short
                    # print("video length too short")
                    # print(len(clip), clip_start, length)
                    clip = videodata[clip_start: clip_start + self.clip_len]
                if self.transforms_:
                    trans_clip = []
                    # fix seed, apply the sample `random transformation` for all frames in the clip 
                    seed = random.random()
                    for frame in clip:
                        random.seed(seed)
                        frame = self.toPIL(frame) # PIL image
                        frame = self.transforms_(frame) # tensor [C x H x W]
                        trans_clip.append(frame)
                    # (T x C X H x W) to (C X T x H x W)
                    clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                else:
                    clip = torch.tensor(clip)
                all_clips.append(clip)
                all_idx.append(torch.tensor(int(class_idx)))

            return torch.stack(all_clips), torch.tensor(int(class_idx))


class UCF101ClipRetrievalDataset(Dataset):
    """UCF101 dataset for Retrieval. Sample clips for each video. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        sample_num(int): number of clips per video.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, sample_num, train=True, transforms_=None, interval=2):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.sample_num = sample_num
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        self.interval = interval

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist01.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist01.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        all_clips = []
        all_idx = []
        # for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.sample_num):
        #     clip_start = int(i - self.clip_len/2)
        #     clip = videodata[clip_start: clip_start + self.clip_len]
        for clip_start in np.linspace(0, length - (self.clip_len * self.interval), self.sample_num):
            clip_start = max(int(clip_start), 0)
            clip = videodata[clip_start: clip_start + self.clip_len * self.interval - 1: self.interval]
            if len(clip) < self.clip_len:
                print("video length too short")
                print(len(clip), clip_start, length)
                clip = videodata[clip_start: clip_start + self.clip_len]
            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)
            all_clips.append(clip)
            all_idx.append(torch.tensor(int(class_idx)))

        return torch.stack(all_clips), torch.stack(all_idx)


class UCF101CACLDataset(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists.

    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """

    def __init__(self, root_dir, clip_len, interval, train=True, transforms_=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.clip_total_frames = clip_len + interval * (clip_len - 1)

        if self.train:
            vcop_train_split_name = 'vcop_train_{}_{}.txt'.format(clip_len, interval)
            vcop_train_split_path = os.path.join(root_dir, 'split', vcop_train_split_name)
            self.train_split = pd.read_csv(vcop_train_split_path, header=None)[0]
        else:
            vcop_test_split_name = 'vcop_test_{}_{}.txt'.format(clip_len, interval)
            vcop_test_split_path = os.path.join(root_dir, 'split', vcop_test_split_name)
            self.test_split = pd.read_csv(vcop_test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)


    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]

        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(0, length - self.clip_total_frames)
        else:
            random.seed(idx)
            tuple_start = random.randint(0, length - self.clip_total_frames)

        clip_start = tuple_start
        clip_end = clip_start + self.clip_len * self.interval - 1
        clip = videodata[clip_start:clip_end:self.interval]
        clip_q = clip.copy()
        clip_k = clip.copy()
        tuple_order = list(range(self.clip_len))
        # random shuffle for train, the same shuffle for test
        id = random.randint(1,len(tuple_order))
        if id ==1:
            shuffle_clip_q = clip
            new_tuple_order = tuple_order
        else:
            frame_len = len(tuple_order)
            tuple_order_copy = tuple_order.copy()
            random.shuffle(tuple_order_copy)
            id_x = tuple_order_copy[:id]
            id_v1 = id_x.copy()
            random.shuffle(id_v1)
            new_tuple_order = tuple_order.copy()
            for i in range(id):
                new_tuple_order[id_x[i]] = tuple_order[id_v1[i]]
            shuffle_clip_q = clip[new_tuple_order,:,:,:]
        shuffle_clip_k = shuffle_clip_q.copy()

        if self.transforms_:
            trans_clip_q = []
            trans_shuffle_clip_q = []
            # fix seed, apply the sample `random transformation` for all frames in the clip
            seed = random.random()
            for i in range(self.clip_len):
                frame = shuffle_clip_q[i]
                random.seed(seed)
                frame = self.toPIL(frame)  # PIL image
                frame = self.transforms_(frame)  # tensor [C x H x W]
                trans_shuffle_clip_q.append(frame)
            trans_shuffle_clip_q = torch.stack(trans_shuffle_clip_q).permute([1, 0, 2, 3])
            for i in range(self.clip_len) :
                random.seed(seed)
                frame = clip_q[i]
                frame = self.toPIL(frame)  # PIL image
                frame = self.transforms_(frame)  # tensor [C x H x W]
                trans_clip_q.append(frame)
            # (T x C X H x W) to (C X T x H x W)
            trans_clip_q = torch.stack(trans_clip_q).permute([1, 0, 2, 3])
            stacked_q = []
            stacked_q.append(trans_clip_q)
            stacked_q.append(trans_shuffle_clip_q)
            stacked_q = torch.stack(stacked_q)

            trans_clip_k = []
            trans_shuffle_clip_k = []
            seed = random.random()
            for i in range(self.clip_len):
                frame = shuffle_clip_k[i]
                random.seed(seed)
                frame = self.toPIL(frame)  # PIL image
                frame = self.transforms_(frame)  # tensor [C x H x W]
                trans_clip_k.append(frame)
            trans_clip_k = torch.stack(trans_clip_k).permute([1, 0, 2, 3])
            for i in range(self.clip_len) :
                random.seed(seed)
                frame = clip_k[i]
                frame = self.toPIL(frame)  # PIL image
                frame = self.transforms_(frame)  # tensor [C x H x W]
                trans_shuffle_clip_k.append(frame)
            # (T x C X H x W) to (C X T x H x W)
            trans_shuffle_clip_k = torch.stack(trans_shuffle_clip_k).permute([1, 0, 2, 3])
            stacked_k = []
            stacked_k.append(trans_clip_k)
            stacked_k.append(trans_shuffle_clip_k)
            stacked_k = torch.stack(stacked_k)

        return stacked_q, stacked_k,torch.tensor(new_tuple_order)


class UCF101SDPDataset(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists.

    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """

    def __init__(self, root_dir, clip_len, interval, train=True, transforms_=None, uniform_sdp=True):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.clip_total_frames = clip_len + interval * (clip_len - 1)
        self.uniform_sdp = uniform_sdp

        if self.train:
            vcop_train_split_name = 'vcop_train_{}_{}.txt'.format(clip_len, interval)
            vcop_train_split_path = os.path.join(root_dir, 'split', vcop_train_split_name)
            self.train_split = pd.read_csv(vcop_train_split_path, header=None)[0]
        else:
            vcop_test_split_name = 'vcop_test_{}_{}.txt'.format(clip_len, interval)
            vcop_test_split_path = os.path.join(root_dir, 'split', vcop_test_split_name)
            self.test_split = pd.read_csv(vcop_test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)


    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]

        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(0, length - self.clip_total_frames)
        else:
            random.seed(idx)
            tuple_start = random.randint(0, length - self.clip_total_frames)

        clip_start = tuple_start
        clip_end = clip_start + self.clip_len * self.interval - 1
        clip = videodata[clip_start:clip_end:self.interval]
        clip_origin = clip.copy()
        tuple_order = list(range(self.clip_len))
        # random shuffle for train, the same shuffle for test
        if self.uniform_sdp:
            id = random.randint(1,len(tuple_order))
            if id ==1:
                shuffle_clip = clip
                new_tuple_order = tuple_order
            else:
                frame_len = len(tuple_order)
                tuple_order_copy = tuple_order.copy()
                random.shuffle(tuple_order_copy)
                id_x = tuple_order_copy[:id]
                id_v1 = id_x.copy()
                random.shuffle(id_v1)
                new_tuple_order = tuple_order.copy()
                for i in range(id):
                    new_tuple_order[id_x[i]] = tuple_order[id_v1[i]]
                shuffle_clip = clip[new_tuple_order,:,:,:]
        else:
            new_tuple_order = tuple_order.copy()
            random.shuffle(new_tuple_order)
            shuffle_clip = clip[new_tuple_order,:,:,:]

        if self.transforms_:
            trans_clip = []
            trans_shuffle_clip = []
            # fix seed, apply the sample `random transformation` for all frames in the clip
            seed = random.random()
            for i in range(self.clip_len):
                frame = shuffle_clip[i]
                random.seed(seed)
                frame = self.toPIL(frame)  # PIL image
                frame = self.transforms_(frame)  # tensor [C x H x W]
                trans_shuffle_clip.append(frame)
            trans_shuffle_clip = torch.stack(trans_shuffle_clip).permute([1, 0, 2, 3])
            for i in range(self.clip_len) :
                random.seed(seed)
                frame = clip_origin[i]
                frame = self.toPIL(frame)  # PIL image
                frame = self.transforms_(frame)  # tensor [C x H x W]
                trans_clip.append(frame)
            # (T x C X H x W) to (C X T x H x W)
            trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            stacked = []
            stacked.append(trans_clip)
            stacked.append(trans_shuffle_clip)
            stacked = torch.stack(stacked)

        return stacked, torch.tensor(new_tuple_order)


def gen_ucf101_vfop_splits(root_dir, clip_len, interval):
    """Generate split files for different configs."""
    vcop_train_split_name = 'vcop_train_{}_{}.txt'.format(clip_len, interval)
    vcop_train_split_path = os.path.join(root_dir, 'split', vcop_train_split_name)
    vcop_test_split_name = 'vcop_test_{}_{}.txt'.format(clip_len, interval)
    vcop_test_split_path = os.path.join(root_dir, 'split', vcop_test_split_name)
    # minimum length of video to extract one tuple
    min_video_len = clip_len + interval * (clip_len - 1)

    def _video_longer_enough(filename):
        """Return true if video `filename` is longer than `min_video_len`"""
        path = os.path.join(root_dir, 'video', filename)
        metadata = ffprobe(path)['video']
        return eval(metadata['@nb_frames']) >= min_video_len

    train_split = pd.read_csv(os.path.join(root_dir, 'split', 'trainlist01.txt'), header=None, sep=' ')[0]
    train_split = train_split[train_split.apply(_video_longer_enough)]
    train_split.to_csv(vcop_train_split_path, index=None)

    test_split = pd.read_csv(os.path.join(root_dir, 'split', 'testlist01.txt'), header=None, sep=' ')[0]
    test_split = test_split[test_split.apply(_video_longer_enough)]
    test_split.to_csv(vcop_test_split_path, index=None)

def gen_kinetics_vfop_splits(root_dir, clip_len, interval):
    """
    get kinetics video lists, the @nb_frames of the video in lists must large
    than clip_len + interval * (clip_len - 1)
    """
    # minimum length of video to extract one tuple
    min_video_len = clip_len + interval * (clip_len - 1)
    video_list = glob(root_dir + "*.jpg")
    f = open("kinetics_trainlist.txt", "w")
    for video_path in video_list:
        metadata = ffprobe(video_path)['video']
        if eval(metadata['@nb_frames']) >= min_video_len:
            f.write(video_path + "\n")
        break
    f.close()


def ucf101_stats():
    """UCF101 statistics"""
    collects = {'nb_frames': [], 'heights': [], 'widths': [], 
                'aspect_ratios': [], 'frame_rates': []}

    for filename in glob('../data/ucf101/video/*/*.avi'):
        metadata = ffprobe(filename)['video']
        collects['nb_frames'].append(eval(metadata['@nb_frames']))
        collects['heights'].append(eval(metadata['@height']))
        collects['widths'].append(eval(metadata['@width']))
        collects['aspect_ratios'].append(metadata['@display_aspect_ratio'])
        collects['frame_rates'].append(eval(metadata['@avg_frame_rate']))

    stats = {key: sorted(list(set(collects[key]))) for key in collects.keys()}
    stats['nb_frames'] = [stats['nb_frames'][0], stats['nb_frames'][-1]]

    pprint(stats)


if __name__ == '__main__':
    seed = 632
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen_ucf101_vfop_splits('./data/ucf101', 16, 2)
    # gen_kinetics_vfop_splits("/data0/gaoziteng/")

    # ucf101_stats()
    # gen_ucf101_vcop_splits('../data/ucf101', 16, 16, 2)
    # gen_ucf101_vcop_splits('../data/ucf101', 16, 32, 3)
    # gen_ucf101_vcop_splits('../data/ucf101', 16, 8, 3)
    # gen_ucf101_vfop_splits('../data/ucf101', 8, 8)
    # gen_ucf101_vfop_splits('./data/ucf101', 16, 2)
    # gen_kinetics_vfop_splits("/data0/gaoziteng/")
    # train_transforms = transforms.Compose([
    #     transforms.Resize((128, 171)),
    #     transforms.RandomCrop(112),
    #     transforms.ToTensor()])
    # # train_dataset = UCF101FOPDataset('../data/ucf101', 8, 3, True, train_transforms)
    # # train_dataset = UCF101VCOPDataset('../data/ucf101', 16, 8, 3, True, train_transforms)
    # train_dataset = UCF101Dataset('../data/ucf101', 16, False, train_transforms)
    # # train_dataset = UCF101RetrievalDataset('../data/ucf101', 16, 10, True, train_transforms)    
    # train_dataloader = DataLoader(train_dataset, batch_size=8)

    # for i, data in enumerate(train_dataloader):
    #     clips, idxs = data
    #     # for i in range(10):
    #     #     filename = os.path.join('{}.mp4'.format(i))
    #     #     skvideo.io.vwrite(filename, clips[0][i])
    #     print(clips.shape)
    #     print(idxs)
    #     exit()
    # pass

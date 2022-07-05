# Cross-Architecture Self-supervised Video Representation Learning [arxiv](https://arxiv.org/abs/2205.13313)

## Usage

### Requirements

- python3.6
- ffmpeg-3.3.30-4
- tensorboard==1.15
- opencv-python==4.2.0.34
- python-Levenshtein==0.12.0
- scikit-video==1.1.11
- torchvision-0.5.0
- torch-1.4.0

### Pretrain SDP

```shell
python train_pretrain_sdp.py --cl 16 --it 2\
    --log pretrain_logs/sdp_c3d_ucf101 --model c3d\
    --dist-url 'tcp://localhost:10001' --multiprocessing-distributed\
    --world-size 1 --rank 0 --bs 64\
    --pf 10 --aug-plus --cos --epochs 300\
    --workers 32
```

### Pretrain CACL

```shell
python train_pretrain.py --cl 16 --it 2\
    --log pretrain_logs/cacl_c3d_ucf101 --model c3d\
    --dist-url 'tcp://localhost:10001' --multiprocessing-distributed\
    --world-size 1 --rank 0 --bs 64\
    --pf 10 --aug-plus --cos --epochs 300\
    --workers 32
```

### Retrieve videos

```shell
python retrieve_videos.py --feature_dir [save_feature_dir] --bs 8 --ckpt [pretrained_weight.pth] --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --workers 8
```

### Fintune

```shell
python train_finetune.py --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 \
    --rank 0 --bs 64 --pf 10 --epochs 150 \
    --workers 8 --log finetune_log/cacl_ucf_c3d --lr 0.1 --ckpt [pretrained_weight.pth] --cos
```

If you find our code or paper useful, please cite as
```
@InProceedings{guo_2022_CVPR,
    author    = {Guo, Sheng and Xiong, Zihua and Zhong, Yujie and Wang, Limin Wang and Guo, Xiaobo and Han, Bing and Huang Weilin},
    title     = {Cross-Architecture Self-supervised Video Representation Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022}
}
```

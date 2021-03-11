# ImVoteNet
**Boosting 3D Object Detection in Point Clouds with Image Votes**

<p align="center">
  <img src="http://xinleic.xyz/images/imvote.png" width="600" />
</p>

This repository contains the code release of the [paper](https://arxiv.org/abs/2001.10692):
```
@inproceedings{qi2020imvotenet,
  title={Imvotenet: Boosting 3d object detection in point clouds with image votes},
  author={Qi, Charles R and Chen, Xinlei and Litany, Or and Guibas, Leonidas J},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

## Installation
Overall, the installation is similar to [VoteNet](https://github.com/facebookresearch/votenet). GPU is required. The code is tested with Ubuntu 18.04, Python 3.7.7, PyTorch 1.4.0, CUDA 10.0 and cuDNN v7.4.

First install [PyTorch](https://pytorch.org/get-started/locally/), for example through [Anaconda](https://docs.anaconda.com/anaconda/install/):
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
Next, install Python dependencies via `pip` ([tensorboardX](https://github.com/lanpa/tensorboardX) is used for for visualizations):
```bash
pip install matplotlib opencv-python plyfile tqdm networkx==2.2 trimesh==2.35.39
pip install tensorboardX --no-deps
```
Now we are ready to clone this repository:
```bash
git clone git@github.com:facebookresearch/imvotenet.git
cd imvotenet
```
The code depends on [PointNet++](http://arxiv.org/abs/1706.02413) as a backbone, which needs compilation:
```bash
cd pointnet2
python setup.py install
cd ..
```

## Data
Please follow the steps listed [here](https://github.com/facebookresearch/votenet/blob/master/sunrgbd/README.md) to set up the SUN RGB-D dataset in the `sunrgbd` folder. The expected dataset structure under `sunrgbd` is:
```
sunrgbd/
  sunrgbd_pc_bbox_votes_50k_{v1,v2}_{train,val}/
  sunrgbd_trainval/
    # raw image data and camera used by ImVoteNet
    calib/*.txt
    image/*.jpg
```
For ImVoteNet, we provide 2D detection results from a pre-trained Faster R-CNN detector [here](https://dl.fbaipublicfiles.com/imvotenet/2d_bbox/sunrgbd_2d_bbox_50k_v1.tgz). Please download the file, uncompress it, and place the resulting folders (`sunrgbd_2d_bbox_50k_v1_{train,val}`) under `sunrgbd` as well.

## Training and Evaluation

Once the code and data are set up, one can train ImVoteNet by the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --use_imvotenet --log_dir log_imvotenet
```
The setting `CUDA_VISIBLE_DEVICES=0` forces the model to be trained on a single GPU (GPU `0` in this case). With the default batch size of 8, it takes about 7G memory during training. 

To reproduce the experimental results in the paper and in general have faster development cycles, one can use a shorter learning schedule: 
```bash
CUDA_VISIBLE_DEVICES=1 python train.py --use_imvotenet --log_dir log_140ep --max_epoch 140 --lr_decay_steps 80,120 --lr_decay_rates 0.1,0.1
```

As a baseline, this code also supports training of the original VoteNet, which is launched by:
```bash
CUDA_VISIBLE_DEVICES=2 python train.py --log_dir log_votenet
```
In fact, the code is based on the VoteNet repository at commit [2f6d6d3](https://github.com/facebookresearch/votenet/tree/2f6d6d3), as a reference, it gives around 58 mAP@0.25.

For other training options, one can use `python train.py -h` for assistance.

After the model is trained, the checkpoint can be tested and evaluated on the `val` set via:
```bash
python eval.py --use_imvotenet --checkpoint_path log_imvotenet/checkpoint.tar --dump_dir eval_imvotenet --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
```
For reference, ImVoteNet gives around 63 mAP@0.25.

## TODO
- Add docs for some functions
- Investigate the 0.5 mAP@0.25 gap after moving to PyTorch 1.4.0. (Originally the code is based on PyTorch 1.0.)

## LICENSE

The code is released under the [MIT license](LICENSE).


## 如何在windows10上跑通
1.无法编译pointnet2问题
- 尽量使用以上 readme 给的环境
- 若跟笔者一样使用NVIDIA 3080（或其他30系列显卡），必须安装CUDA11，然后按照下main的链接来修改代码，再编译。
链接：[votenet中找到的解决方案](https://github.com/facebookresearch/votenet/issues/108)
  。如下：
  ```
  Found a fix:

    1. change all instances of AT_CHECK to TORCH_CHECK inside all the source files inside pointnet2/_ext_src/src and pointnet2/_ext_src/include. This is due to an API change in PyTorch.
    2. change pointnet2/setup.py:
  ```
  ```
      # Copyright (c) Facebook, Inc. and its affiliates.
    # 
    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory of this source tree.
    
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    import glob
    import os
    
    _ext_src_root = "_ext_src"
    _ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
        "{}/src/*.cu".format(_ext_src_root)
    )
    _ext_headers = glob.glob("{}/include/*".format(_ext_src_root))
    
    headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ext_src', 'include')
    
    setup(
        name='pointnet2',
        ext_modules=[
            CUDAExtension(
                name='pointnet2._ext',
                sources=_ext_sources,
                extra_compile_args={
                    "cxx": ["-O2", headers],
                    "nvcc": ["-O2", headers]
                },
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
  ```
  
2.编译时遇到cl：xxxx的问题
- 安装vs2019，安装时选择c++就行，不用.NET等
- 把安装完后的vs2019的cl.exe所在的目录放到电脑的环境变量中

3.遇到utf-8的问题
- windows下将cpp_extension.py文件中的
```
match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode().strip())
```
改为
```
match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode('gbk').strip())
```

4.SUN-RGBD数据集获取：
- 使用MATLAB2020b，其他版本（如MATLAB2016b)会出错
- 拷入U盘备份
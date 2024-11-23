# IFTR Installation Guide

**Author**: Shaohong Wang  wangsh0111@zju.edu.cn

------

## Installation 

### Create a conda environment

All operations should be done **on machines with GPUs**. 

We expect a CUDA version displayed in the top right corner of the `nvidia-smi` to be higher than `11.6`

```Bash
conda create --name IFTR python=3.8 cmake=3.22.1
conda activate IFTR 

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
```

### **Install** **some other packages**

```Bash
cd IFTR-main
pip install -r requirements.txt
```

### Install mmdet3d

```Bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
# recommend downloading .whl directly and then installing it
# pip install mmcv_full-1.4.0-cp38-cp38-manylinux1_x86_64.whl

pip install mmdet==2.14.0 mmsegmentation==0.14.1
pip install ninja tensorboard nuscenes-devkit==1.1.10 scikit-image==0.19.0 lyft-dataset-sdk
pip install numpy==1.19.5 pandas==1.4.4 scikit-image==0.19.3 llvmlite

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install

pip install einops fvcore seaborn iopath timm  typing-extensions pylint ipython numba yapf==0.40.1 scikit-image==0.19.3

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# # Or, to install it from a local clone:
# git clone https://github.com/facebookresearch/detectron2.git
# cd detectron2
# python -m pip install -e ./
pip install numpy==1.19.5 pandas==1.4.4 scikit-image==0.19.3 nuscenes-devkit==1.1.10 
```

### Install pypcd

```Bash
git clone https://github.com/klintan/pypcd.git
cd pypcd
python setup.py install
```

### **Install** **IFTR**

```PowerShell
cd IFTR-main

python setup.py develop
# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace 
```

### **Troubleshooting**

AttributeError: module 'distutils' has no attribute 'version'

```Bash
pip uninstall setuptools                # cannot use conda uninstall setuptools
pip install setuptools==59.5.0
```

mmdetection error TypeError: FormatCode() got an unexpected keyword argument ‘verify‘

```Bash
pip uninstall yapf
pip install yapf==0.40.1
```

libGL.so.1: cannot open shared object file: No such file or directory

```Bash
apt-get update && apt-get install libgl1
```

## **Command**

Suppose you are in the folder IFTR-main

### Training

```Shell
# Single GPU Training
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py -y YAML_FILE [--model_dir MODEL_DIR]

# Multi-GPU training, nproc_per_node set to GPU number
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env opencood/tools/train.py -y YAML_FILE [--model_dir MODEL_DIR]
```

- `-y YAML_FILE`  the yaml configuration file
- `[--model_dir MODEL_FOLDER]` is optional, indicating that training continues from this log (resume training). It will read config.yaml from under `MODEL_FOLDER` instead of the input `-y YAML_FILE`. so it can be written `-y None` and no yaml file is provided

### Testing

```Shell
python opencood/tools/inference.py --model_dir MODEL_DIR [--fusion_method iftr]
```

- `inference.py` There are a number of optional parameters. Please go to the code to see details
- `[--fusion_method iftr]` The default `fusion_method` is `iftr`, which corresponds to the iftr model with the addition of late fusion strategy

**Optional** **fusion_method** **include**

- `[--fusion_method intermediate]` This corresponds to the naive iftr model



------


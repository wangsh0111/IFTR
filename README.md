# IFTR (ECCV2024)
IFTR: An Instance-Level Fusion Transformer for Visual Collaborative Perception [[Paper](https://arxiv.org/abs/2407.09857)]


## Abstract
 Multi-agent collaborative perception has emerged as a widely recognized technology in the field of autonomous driving in recent years. However, current collaborative perception predominantly relies on LiDAR point clouds, with significantly less attention given to methods using camera images. This severely impedes the development of budget-constrained collaborative systems and the exploitation of the advantages offered by the camera modality. This work proposes an instance-level fusion transformer for visual collaborative perception (IFTR), which enhances the detection performance of camera-only collaborative perception systems through the communication and sharing of visual features. To capture the visual information from multiple agents, we design an instance feature aggregation that interacts with the visual features of individual agents using predefined grid-shaped bird eye view (BEV) queries, generating more comprehensive and accurate BEV features. Additionally, we devise a cross-domain query adaptation as a heuristic to fuse 2D priors, implicitly encoding the candidate positions of targets. Furthermore, IFTR optimizes communication efficiency by sending instance-level features, achieving an optimal performance-bandwidth trade-off. We evaluate the proposed IFTR on a real dataset, DAIR-V2X, and two simulated datasets, OPV2V and V2XSet, achieving performance improvements of 57.96\%, 9.23\% and 12.99\% in AP@70 metrics compared to the previous SOTAs, respectively. Extensive experiments demonstrate the superiority of IFTR and the effectiveness of its key components. 


## Methods
![Original1](images/iftr.jpg)


## Installation
Please visit the docs [IFTR Installation Guide](docs/install.md) to learn how to install and run this repo. 


## Data Preparation
mkdir a `dataset` folder under IFTR. Put your OPV2V, V2XSet, DAIR-V2X data in this folder. You just need to put in the dataset you want to use.

```
IFTR/dataset

. 
├── my_dair_v2x 
│   ├── v2x_c
│   ├── v2x_i
│   └── v2x_v
├── OPV2V
│   ├── additional
│   ├── test
│   ├── train
│   └── validate
└── V2XSET
    ├── test
    ├── train
    └── validate
```


## Citation
```
@article{wang2024iftr,
  title={IFTR: An Instance-Level Fusion Transformer for Visual Collaborative Perception},
  author={Wang, Shaohong and Bin, Lu and Xiao, Xinyu and Xiang, Zhiyu and Shan, Hangguan and Liu, Eryun},
  journal={arXiv preprint arXiv:2407.09857},
  year={2024}
}
```


## Acknowledgement
This project is impossible without the code of [CoAlign](https://github.com/yifanlu0227/CoAlign), [BEVFormer](https://github.com/fundamentalvision/BEVFormer) and [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)!

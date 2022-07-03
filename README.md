FALCON: FAst and Lightweight CONvolution
===

This package provides implementations of FALCON/FALCONBranch convolution with their corresponding CNN model.

## Overview
#### Code structure
``` Unicode
FALCON
  │ 
  ├── src
  │    │     
  │    ├── models
  │    │     ├── vgg.py: VGG model
  │    │     ├── resnet.py: ResNet model
  │    │     ├── model_imagenet.py: Pretrained model (from pytorch) 
  │    │     ├── falcon.py: FALCON
  │    │     └── stconv_branch.py: StConvBranch & FALCONBranch
  │    │      
  │    ├── train_test
  │    │     ├── imagenet.py: train/validate on ImageNet 
  │    │     ├── main.py: train/test on CIFAR10/CIFAR100/SVHN 
  │    │     ├── train.py: training process
  │    │     ├── test.py: testing process
  │    │     └── validation.py: validation process
  │    │     
  │    └── utils
  │          ├── compression_cal.py: calculate the number of parameters and FLOPs
  │          ├── default_param.py: default cfgs 
  │          ├── load_data.py: load datasets
  │          ├── lr_decay.py: control learning rate
  │          ├── optimizer_option.py: choose optimizer 
  │          ├── save_restore.py: save and restore trained model
  │          └── timer.py: timer for inference time
  │
  └── script: shell scripts for execution of training/testing codes
```

#### Naming convention
**StandardConv**: Standard Convolution (baseline)

**FALCON**: FAst and Lightweight CONvolution - the new convolution architecture we proposed

**Rank**: Rank of convolution. Copy the conv layer for k times, run independently and add output together at the end of the layer. This hyperparameter helps balace compression rate/ accelerate rate and accuracy.

**FALCONBranch**: New version of FALCON - for fitting FALCON into ShuffleUnitV2 architecture.

#### Data description
* CIFAR-10 datasets
* CIFAR-100 datasets
* SVHN
* ImageNet
* Note that: 
    * CIFAR and SVHN datasets depend on torchvision (https://pytorch.org/docs/stable/torchvision/datasets.html#cifar). You don't have to download anything. When executing the source code, the datasets will be automaticly downloaded if it is not detected.
    * ImageNet is downloaded from http://www.image-net.org/challenges/LSVRC/
   
#### Output
* For CIFAR datasets, the trained model will be saved in `train_test/trained_model/` after training.
* For ImageNet, the checkpoint will be saved in `train_test/checkpoint`
* You can test the model only if there is a trained model in `train_test/trained_model/`.

## Install
#### Environment 
* Unbuntu
* CUDA 9.0
* Python 3.6
* torch
* torchvision
#### Dependence Install
    pip install torch torchvision

## How to use 
#### Clone the repository
    git clone https://
    cd FALCON
#### Training & Testing
* To train the model on CIFAR-10/CIFAR-100/SVHN datasets, run script:
    ```    
    cd src/train_test
    python main.py -train -conv StandardConv -m VGG19 -data cifar10
    python main.py -train -conv FALCON -m VGG19 -data cifar10 -init
    ```
    The trained model will be saved in `src/train_test/trained_model/`
* To test the model, run script:
    ```
    cd src/train_test
    python main.py -conv StandardConv -m VGG19 -data cifar10
    python main.py -conv FALCON -m VGG19 -data cifar10 -init
    ```
    The testing accuracy, inference time, number of parameters and number of FLOPs will be printed on the screen.
* Pre-trained model is saved in `src/train_test/trained_model/`
    * For example:
        * Standard model:
            conv=StandardConv,model=VGG19,data=cifar100,rank=1,alpha=1.pkl
        * FALCON model:
            conv=FALCON,model=VGG19,data=cifar100,rank=1,alpha=1,init.pkl

#### DEMO
* There are two demo scripts: `script/train.sh` and `script/inference.sh`.
* You can change arguments in `.sh` files to train/test different model.
    * `train.sh`: Execute training process of the model
        * Accuracy/ loss/ training time for 100 iteration will be printed on the screen during training.
        * Accuracy/ inference time/ number of parameters/ number of FLOPs will be printed on the screen after training.
    * `inference.sh`: Execute inference process of the model
        * Accuracy/ inference time/ number of parameters/ number of FLOPs will be printed on the screen.
        * You can run this file only when the trained model exist.
        * Sample trained model is provided in `src/train_test/trained_model/`.

## Contact us
- Chun Quan (quanchun@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

## Reference

If you use this code, please cite the following paper.

```
@article{DBLP:journals/corr/abs-1909-11321,
  author    = {Chun Quan and
               Jun{-}Gi Jang and
               Hyun Dong Lee and
               U Kang},
  title     = {{FALCON:} Fast and Lightweight Convolution for Compressing and Accelerating
               {CNN}},
  journal   = {CoRR},
  volume    = {abs/1909.11321},
  year      = {2019},
  url       = {http://arxiv.org/abs/1909.11321},
  eprinttype = {arXiv},
  eprint    = {1909.11321},
  timestamp = {Fri, 27 Sep 2019 13:04:21 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1909-11321.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## License
Licensed under the Apache License, Version 2.0

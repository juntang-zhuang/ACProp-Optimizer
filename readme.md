This repository contains code to reproduce results for submission NeurIPS 2021, "Momentum Centering and Asynchronous Update for Adaptive Gradient Methods".

This repo heavily depends on the official implementation of AdaBound: https://github.com/Luolc/AdaBound and AdaBelief: https://github.com/juntang-zhuang/Adabelief-Optimizer.

### Dependencies
python 3.7
pytorch 1.1.0
torchvision 0.3.0
jupyter notebook
AdaBelief (Please install by "pip install adabelief-pytorch==0.2.0")


### Visualization of pre-trained curves
Please use the jupyter notebook "visualization.ipynb" to visualize the training and test curves of different optimizers.

### Training and evaluation code

(1) train network with
CUDA_VISIBLE_DEVICES=0 python main.py --model vgg --optim acprop --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9

--model: name of model, choices include ['vgg','resnet','densenet']
--optim: name of optimizers, choices include ['acprop','adashift','sgd', 'adam', 'adamw', 'adabelief', 'radam',]
--lr: learning rate
--eps: epsilon value used for optimizers. Note that Yogi uses a default of 1e-03, other optimizers typically uses 1e-08
--beta1, --beta2: beta values in adaptive optimizers
--momentum: momentum used for SGD.

(2) visualize using the notebook "visualization.ipynb"


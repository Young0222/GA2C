# GA2C

## This is the PyTorch implementation code for our paper: Graph Contrastive Learning with Reinforcement Augmentation


## Environment Requirements

The code has been tested under Python 3.8.8. The required packages are as follows:

* torch==1.8.0
* torch-geometric==2.0.2
* torch-scatter==2.0.9
* torch-sparse==0.6.9
* torch-spline-conv==1.2.2
* torchvision==0.9.0
* ...
  
You can use 'pip install -r requirement.txt' to install dependency packages.

## Unsupervised learning

python a2c_gcl_tu.py --seed 2024 --downstream_classifier SVC

## Transfer learning

Pretrain on ZINC-2M: python a2c_transfer_pretrain_chem.py

Fintune on a specific dataset: python a2c_transfer_finetune_chem.py

## Semi-supervised learning

In the folder of 'semi_supervised', please run:

python main.py --exp=joint_cl_exp --semi_split=10 --dataset=COLLAB --save=joint_cl_exp --epochs=100 --batch_size=32 --lr=0.001



# GA2C

## This is the PyTorch implementation code for our paper: Graph Contrastive Learning with Reinforcement Augmentation


## 🔬 Environment Requirements

The code has been tested under Python 3.8.8. The required packages are as follows:

* torch==1.8.0
* torch-geometric==2.0.2
* torch-scatter==2.0.9
* torch-sparse==0.6.9
* torch-spline-conv==1.2.2
* torchvision==0.9.0
* ...
  
You can use 'pip install -r requirement.txt' to install dependency packages.

## 🚀 Unsupervised learning

```
python a2c_gcl_tu.py --seed 2024 --downstream_classifier SVC
```

## 🚀 Transfer learning

```
Pretrain on ZINC-2M: python a2c_transfer_pretrain_chem.py

Fintune on a specific dataset: python a2c_transfer_finetune_chem.py
```

## 🚀 Semi-supervised learning

In the folder of 'semi_supervised', please run:

```
python main.py --exp=joint_cl_exp --semi_split=10 --dataset=COLLAB --save=joint_cl_exp --epochs=100 --batch_size=32 --lr=0.001
```

## 📚 Log file

We have preserved the model training logs in the 'log_file' folder. For instance, the unsupervised learning log results of GA2C on the MUTAG dataset are as follows:

```
INFO:root:Running tu......  
INFO:root:Using Device: cuda:2  
INFO:root:Seed: 2024  
INFO:root:Namespace(actor_view_lr=0.001, batch_size=128, config='config.yaml', critic_view_lr=0.001, dataset='MUTAG', downstream_classifier='SVC', drop_ratio=0.0, emb_dim=32, epochs=30, eval_interval=5, mlp_edge_model_dim=128, model_lr=0.001, num_gc_layers=3, pooling_type='layerwise', reg_lambda=0.0, seed=2024)  
INFO:root:n_features: 1  
INFO:root:Before training Embedding Eval Scores: Train: 0.8731 Val: 0.8731 Test: 0.8731  
INFO:root:current reward: 5.539564609527588  
INFO:root:Epoch 1, Model Loss 499.3180, Actor Loss 1100.0768, Critic Loss 12201.1574  
INFO:root:Epoch 2, Model Loss 492.7890, Actor Loss 0.0000, Critic Loss 0.0000  
INFO:root:Epoch 3, Model Loss 490.6491, Actor Loss 0.0000, Critic Loss 0.0000  
INFO:root:current reward: 5.54935359954834  
INFO:root:Epoch 4, Model Loss 489.2530, Actor Loss 907.6128, Critic Loss 14274.4207  
INFO:root:Epoch 5, Model Loss 493.5720, Actor Loss 0.0000, Critic Loss 0.0000  
INFO:root:Metric: accuracy Train: 0.8985 Val: 0.8985 Test: 0.8985  
  
...  
  
INFO:root:FinishedTraining!  
INFO:root:Dataset: MUTAG  
INFO:root:reg_lambda: 0.0  
INFO:root:drop_ratio: 0.0  
INFO:root:BestEpoch: 3  
INFO:root:BestTrainScore: 0.9096491228070175  
INFO:root:BestValidationScore: 0.9096491228070175  
INFO:root:FinalTestScore: 0.9096491228070175  
INFO:root:Mean Testscore: 90.34±0.39
```

The semi-supervised learning log results of GA2C on the MUV dataset are as follows:

```
INFO:root:Using Device: cuda:2  
INFO:root:Seed: 2024  
INFO:root:Namespace(JK='last', batch_size=32, dataset='muv', decay=0.0, device=0, dropout_ratio=0.5, emb_dim=300, epochs=100, eval_train=1, gnn_type='gin', graph_pooling='mean', input_model_file='./models_ga2c/chem/1024pretrain_ga2c_seed_2024_reg_0.0_epoch_10.pth', lr=0.001, lr_scale=1, num_layer=3, num_workers=4, seed=2024, split='scaffold')  
INFO:root:MoleculeDataset(93087)  
INFO:root:scaffold  
INFO:root:Data(edge_attr=[30, 2], id=[1], y=[17], x=[15, 2], edge_index=[2, 30])  
INFO:root:====epoch 1 SupervisedLoss 1.5139828881726405  
INFO:root:====Evaluation  
INFO:root:EvalTrain: 0.6646693010909368 EvalVal: 0.6188019984544774 EvalTestt: 0.6005540732476364  
INFO:root:====epoch 2 SupervisedLoss 0.4534396019915234  
INFO:root:====Evaluation  
INFO:root:EvalTrain: 0.6454371420363606 EvalVal: 0.6449092016253598 EvalTestt: 0.646384124498649  
INFO:root:====epoch 3 SupervisedLoss 0.4506925367691132
INFO:root:====Evaluation
INFO:root:EvalTrain: 0.7177383277189476 EvalVal: 0.6803288988563876 EvalTestt: 0.6787941707186023
INFO:root:====epoch 4 SupervisedLoss 0.44905699498268764
INFO:root:====Evaluation
INFO:root:EvalTrain: 0.7410288882863967 EvalVal: 0.7447307387734368 EvalTestt: 0.7058885491854934
INFO:root:====epoch 5 SupervisedLoss 0.44011415983783064
...
INFO:root:====epoch 100 SupervisedLoss 0.3141621045593042
INFO:root:====Evaluation
INFO:root:EvalTrain: 0.9368990614094688 EvalVal: 0.745116806865475 EvalTestt: 0.7747370145508443
INFO:root:FinalTestScore: 0.7925660626051265
INFO:root:Mean Testscore: 79.76±0.40
```

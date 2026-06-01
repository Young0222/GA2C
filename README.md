# 🧠 GA2C: Graph Contrastive Learning with Reinforcement Augmentation

> A PyTorch implementation of **GA2C**, a graph contrastive learning framework that models graph augmentation as a sequential decision process.

[![Paper](https://img.shields.io/badge/Paper-IJCAI%202024-blue)](https://www.ijcai.org/proceedings/2024/0246)
[![Python](https://img.shields.io/badge/Python-3.8.8-green)](#-environment)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0-red)](#-environment)

## ✨ Overview

Graph contrastive learning often depends on graph data augmentation, but many methods treat each augmentation step as an isolated action. GA2C takes a different path. It models augmentation as a **Markov decision process** and learns graph views with an **advantage actor critic** strategy.

This repository includes three settings:

- 🔬 **Unsupervised learning** on TU datasets
- 🧪 **Transfer learning** for molecular graphs
- 🧬 **Semi-supervised learning** on graph classification benchmarks

## 📄 Paper

**Graph Contrastive Learning with Reinforcement Augmentation**  
Ziyang Liu, Chaokun Wang, Cheng Wu  
IJCAI 2024

- Paper: [🔗](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ijcai.org/proceedings/2024/0246.pdf)
- Official IJCAI page: [🔗](https://www.ijcai.org/proceedings/2024/0246)
- DOI: [10.24963/ijcai.2024/246](https://doi.org/10.24963/ijcai.2024/246)

## 🌟 Key Ideas

- 🎯 **Sequential augmentation**: graph augmentation is modeled as a multi step decision process instead of a one shot perturbation.
- 🎭 **Actor critic design**: an actor generates augmented views, and a critic estimates their long term value.
- 🔁 **Joint optimization**: the augmentation policy and the graph encoder improve each other during training.
- 🧪 **Broad evaluation**: the codebase covers unsupervised, transfer, and semi-supervised graph learning.
- 🧾 **Saved training logs**: final logs are included for quick reference and reproduction support.

## 🗂️ Repository Structure

```text
GA2C-main/
├── a2c_gcl_tu.py                   # Unsupervised learning on TU datasets
├── a2c_transfer_pretrain_chem.py   # Molecular pretraining on ZINC
├── a2c_transfer_finetune_chem.py   # Molecular finetuning on downstream tasks
├── semi_supervised/                # Semi-supervised experiments
├── transfer/                       # Transfer learning models and utilities
├── unsupervised/                   # Unsupervised encoders, learners, and utils
├── datasets/                       # Dataset wrappers
├── case_study/                     # Molecule view examples
├── log_file/                       # Saved training logs
└── requirements.txt                # Dependency list
```

## ⚙️ Environment

The code was developed with the following setup:

- Python `3.8.8`
- PyTorch `1.8.0`
- torch-geometric `2.0.2`
- torch-scatter `2.0.9`
- torch-sparse `0.6.9`
- torch-spline-conv `1.2.2`
- torchvision `0.9.0`
- rdkit `2023.3.1`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Unsupervised learning

Run GA2C on TU datasets:

```bash
python a2c_gcl_tu.py --seed 2024 --downstream_classifier SVC
```

Useful default settings in `a2c_gcl_tu.py`:

- dataset: `REDDIT-MULTI-5K`
- epochs: `60`
- batch size: `128`
- embedding dimension: `32`

### 2. Transfer learning

Pretrain on ZINC style molecular data:

```bash
python a2c_transfer_pretrain_chem.py
```

Finetune on downstream molecular benchmarks:

```bash
python a2c_transfer_finetune_chem.py
```

The finetuning script loops over these datasets:

- `bbbp`
- `bace`
- `tox21`
- `toxcast`
- `sider`
- `clintox`
- `muv`
- `hiv`

### 3. Semi-supervised learning

Move into the semi-supervised folder and run:

```bash
cd semi_supervised
python main.py --exp=joint_cl_exp --semi_split=10 --dataset=COLLAB --save=joint_cl_exp --epochs=100 --batch_size=32 --lr=0.001
```

More examples are provided in:

- `run_us_ts_ga2c.sh`
- `semi_supervised/run_ss_ga2c.sh`

## 📊 Logged Results

This repository already includes training logs in `log_file/`. A few final scores from the saved logs are listed below.

### Unsupervised learning

| Dataset | Mean Test Score |
| --- | --- |
| MUTAG | `90.34±0.39` |
| NCI1 | `80.62±0.39` |
| DD | `77.20±0.67` |
| PROTEINS | `75.67±0.52` |
| COLLAB | `72.13±0.34` |

### Transfer learning

| Dataset | Mean Test Score |
| --- | --- |
| BACE | `82.34±0.12` |
| MUV | `79.76±0.40` |
| BBBP | `74.30±1.03` |
| ToxCast | `64.18±0.26` |

### Semi-supervised learning

Saved experiment logs under `semi_supervised/exp/joint_cl_exp/` include:

| Dataset | Test Accuracy |
| --- | --- |
| MUTAG | `87.25 ± 7.27` |
| PROTEINS | `75.84 ± 2.58` |

## 📝 Notes

- The saved logs in `log_file/` are a good starting point if you want to compare your own runs with the authors' outputs.
- The scripts expect datasets under paths such as `original_datasets/` and `original_datasets/transfer/`.
- The current `a2c_gcl_tu.py` file contains debugging lines (`print(...)` and `sys.exit()`), so you may need to remove them before running a full unsupervised training job from the current snapshot.

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{liu_ga2c,
  title     = {Graph Contrastive Learning with Reinforcement Augmentation},
  author    = {Liu, Ziyang and Wang, Chaokun and Wu, Cheng},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {2225--2233},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/246},
  url       = {https://doi.org/10.24963/ijcai.2024/246}
}
```

## 🙌 Acknowledgment

Thanks for your interest in GA2C. If this project helps your research, a citation is greatly appreciated.

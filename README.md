# LDL-FLC: Label Distribution Learning by Exploiting Fuzzy Label Correlation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains both **Python** and **MATLAB** implementations of Label Distribution Learning (LDL) algorithms, with a focus on Fuzzy Label Clustering (FLC) methods.

## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License & Attribution](#license--attribution)

## About

This project implements several Label Distribution Learning algorithms, including:

- **LDL-FC**: Label Distribution Learning with Fuzzy Label Clustering
- **LDL-FCC**: Label Distribution Learning with Fuzzy Label Clustering (Joint version)
- **LDL-LRR**: Label Distribution Learning with Label Ranking Regularization
- **LDL-SCL**: Label Distribution Learning with Structure Consistency Learning
- **SA-BFGS**: Simulated Annealing with BFGS optimization

## Features

- ✅ **Dual Language Support**: Complete implementations in both Python and MATLAB
- ✅ **Verified Consistency**: MATLAB and Python versions produce identical results (differences < 0.03%)
- ✅ **Well Organized**: Core modules organized in `core/` subdirectory
- ✅ **Comprehensive**: Includes all algorithms, utilities, and evaluation metrics

## Installation

### Python Requirements

```bash
pip install numpy scipy scikit-learn scikit-fuzzy torch
```

### MATLAB Requirements

- MATLAB R2019b or higher
- Python 3.6+ (for data loading via `.npy` and `.pkl` files)
- Optional Toolboxes:
  - Statistics and Machine Learning Toolbox
  - Optimization Toolbox
  - Fuzzy Logic Toolbox

## Quick Start

### Python

```bash
# Run LDL-FC
cd python
python run_LDLFC.py

# Run LDL-FCC
python run_LDLFCC.py

# Run LDL-LRR
python run_LDLLRR.py
```

### MATLAB

```matlab
% Initialize paths
cd matlab
init_path();

% Run LDL-FC
run_LDLFC('SJAFFE');

% Run LDL-FCC
run_LDLFCC('SJAFFE');

% Run LDL-LRR
run_LDLLRR_all();
```

## Algorithms

### LDL-FC (Fuzzy Label Clustering)

Fuzzy label clustering based label distribution learning method.

```python
# Python
from ldl_flc import LDL_FLC
model = LDL_FLC(g=5, l1=0.001, l2=0.01)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

```matlab
% MATLAB
model = LDL_FLC(5, 1e-3, 1e-2);
model.fit(X_train, y_train);
model.solve();
y_pred = model.predict(X_test);
```

### LDL-LRR (Label Ranking Regularization)

Label ranking regularization based method.

```python
# Python
from ldllrr import LDL_LRR
model = LDL_LRR(lam=1e-3, beta=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

```matlab
% MATLAB
model = LDL_LRR('lam', 1e-3, 'beta', 1);
model.fit(X_train, y_train);
y_pred = model.predict(X_test);
```

## Project Structure

```
LDL-FLC/
├── python/              # Python implementation
│   ├── core/           # Core modules
│   │   ├── ldl_flc.py
│   │   ├── ldllrr.py
│   │   ├── ldl_metrics.py
│   │   └── ...
│   ├── run_LDLFC.py
│   ├── run_LDLFCC.py
│   └── ...
│
├── matlab/             # MATLAB implementation
│   ├── core/           # Core modules
│   │   ├── LDL_FLC.m
│   │   ├── LDL_LRR.m
│   │   ├── ldl_metrics.m
│   │   └── ...
│   ├── run_LDLFC.m
│   ├── run_LDLFCC.m
│   └── ...
│
└── README.md
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@ARTICLE{10636800,
  author={Wang, Jing and Kou, Zhiqiang and Jia, Yuheng and Lv, Jianhui and Geng, Xin},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Label Distribution Learning by Exploiting Fuzzy Label Correlation}, 
  year={2025},
  volume={36},
  number={5},
  pages={8979-8990},
  doi={10.1109/TNNLS.2024.3438756}
}
```

## License & Attribution

This project is a derivative work based on the original implementation from [wangjing4research/LDL-FLC](https://github.com/wangjing4research/LDL-FLC).

**Copyright Attribution:**
- Original Python code and algorithms: Copyright (c) wangjing4research
- MATLAB implementation and code improvements: Copyright (c) 2024 Nitrogen216

**Contributions of this repository:**
- Complete MATLAB port of all algorithms with verified numerical consistency
- Improved code organization with core modules structure
- Bug fixes and enhanced documentation

**License:** This project maintains the same license as the [original repository](https://github.com/wangjing4research/LDL-FLC). Please refer to the original repository for license details.

**Important:** When using this code, please cite the original paper (see Citation section above) and acknowledge the original repository. The datasets are provided by the original papers - please cite them accordingly.

## Datasets

The datasets are shared by the original papers. We provide processed versions of them. If you use these datasets, **please remember to cite the original papers**.

Available datasets:
- SJAFFE
- M2B
- RAF_ML
- Flickr_ldl
- Gene
- SBU_3DFE
- SCUT_FBP
- Scene
- Twitter_ldl
- fbp5500
- Ren



# MATLAB Implementation of LDL Algorithms

这是 Label Distribution Learning (LDL) 算法的完整 MATLAB 实现，从 Python 版本翻译而来。

## 📋 目录

- [环境要求](#环境要求)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
- [核心模块](#核心模块)
- [算法实现](#算法实现)
- [运行脚本](#运行脚本)
- [测试](#测试)
- [注意事项](#注意事项)

## 🔧 环境要求

### 必需组件
- MATLAB R2019b 或更高版本
- Python 3.6+ (用于数据加载)
- NumPy (Python包，用于读取.npy文件)

### 可选工具箱
- **Statistics and Machine Learning Toolbox** (用于 `knnsearch`, `kmeans`, `fitcnb`)
- **Optimization Toolbox** (用于 `fminunc`, `fmincon`)
- **Fuzzy Logic Toolbox** (用于 `fcm` - 模糊C均值)

> **注意**: 如果没有可选工具箱，代码会自动使用备用实现（fallback），但可能速度较慢。

## 📁 文件结构

```
matlab/
├── README_MATLAB.md              # 本文档
├── MODEL_CLASSES.md              # 类文件组织说明
├── init_path.m                   # 路径初始化脚本（自动添加 core/ 到路径）
├── smoke_test.m                  # 完整测试脚本
│
├── core/                         # 核心模块目录（所有基础模型类和工具）
│   ├── 基础模型类
│   │   ├── bfgs_ldl.m            # BFGS-LDL 模型类
│   │   ├── AA_KNN.m              # 自适应 K 近邻类
│   │   ├── PT_Bayes.m            # Problem Transformation (Naive Bayes)
│   │   ├── PT_SVM.m              # Problem Transformation (SVM)
│   │   ├── LDL2SL.m              # 标签分布转单标签（采样）
│   │   └── LDL2Bayes.m           # 标签分布转单标签（argmax）
│   │
│   ├── 主要算法类
│   │   ├── LDL_FLC.m             # LDL with Fuzzy Label Clustering
│   │   ├── LDL_LRR.m             # LDL with Label Ranking Regularization
│   │   ├── LDL_SCL.m             # Structure Consistency Learning
│   │   ├── LDLLDM_Full.m         # LDL with Label Distribution Manifold
│   │   ├── LDLLDM_Cluster.m      # LDLLDM 的簇辅助类
│   │   └── LDM_SC_api.m          # Label Distribution Manifold Spectral Clustering
│   │
│   ├── 辅助算法
│   │   ├── joint_FCLC.m          # Joint Fuzzy Clustering + LDM
│   │   └── barycenter_kneighbors_graph.m  # LLE算法（重心K近邻图）
│   │
│   └── 工具函数
│       ├── ldl_metrics.m          # 评价指标（6个指标）
│       └── util.m                 # 工具函数（数据加载/保存，Python接口）
│
└── 运行脚本
    ├── run_LDLFC.m               # 运行 LDL-FC 实验
    ├── run_LDLFCC.m              # 运行 LDL-FCC 实验
    ├── run_LDLLRR.m              # 运行 LDL-LRR 实验
    ├── run_LDLSCL.m              # 运行 LDL-SCL 实验
    └── run_SABFGS.m              # 运行 SA-BFGS 实验
```

> **重要说明**: 
> - 所有核心模块（模型类、工具函数）都位于 `core/` 子目录下
> - 运行脚本会自动调用 `init_path()` 来添加 `core/` 到 MATLAB 路径
> - 如果手动使用核心模块，请先运行 `init_path()` 或手动添加 `core/` 到路径

> **重要说明**: 由于 MATLAB 的限制，每个类必须在独立的 .m 文件中。原 `ldl_models.m` 已拆分为多个独立类文件。详见 `MODEL_CLASSES.md`。

## 🚀 快速开始

### 1. 配置 Python 环境（用于数据加载）

MATLAB 需要调用 Python 来加载 `.npy` 和 `.pkl` 文件：

```matlab
% 检查 Python 配置
pyenv

% 如果未配置，设置 Python 可执行文件路径
pyenv('Version', '/usr/local/bin/python3')  % macOS/Linux
% 或
pyenv('Version', 'C:\Python39\python.exe')  % Windows
```

### 2. 添加路径

有两种方式：

**方式1：使用 init_path()（推荐）**
```matlab
% 进入 matlab 目录
cd('/Volumes/SAMSUNG/Project/LDL-FLC/matlab');

% 初始化路径（自动添加 matlab/ 和 matlab/core/ 到路径）
init_path();
```

**方式2：手动添加路径**
```matlab
% 添加 matlab 文件夹到路径
addpath('/Volumes/SAMSUNG/Project/LDL-FLC/matlab');
% 添加 core 子目录到路径（必需！）
addpath('/Volumes/SAMSUNG/Project/LDL-FLC/matlab/core');
```

> **注意**: 所有运行脚本（`run_*.m`）会自动调用 `init_path()`，所以通常不需要手动添加路径。

### 3. 运行测试

```matlab
% 运行完整的烟雾测试
smoke_test

% 预期输出：
% === Running MATLAB LDL Smoke Tests ===
% 
% Test 1: Metrics... OK
% Test 2: Barycenter k-neighbors graph... OK
% Test 3: LDL_FLC model... OK
% Test 4: bfgs_ldl model... OK
% Test 5: AA_KNN model... OK
% Test 6: LDL_LRR model... OK
% Test 7: LDLLDM_Full model... OK
% Test 8: joint_FCLC... OK
% Test 9: LDM_SC spectral clustering... OK
% Test 10: LDL_SCL function... OK
% 
% === All Smoke Tests Passed! ===
```

### 4. 运行实验

```matlab
% 确保在项目根目录
cd /Volumes/SAMSUNG/Project/LDL-FLC

% 运行 LDL-FC 算法（10折交叉验证）
run_LDLFC('SJAFFE')

% 运行 LDL-FCC 算法
run_LDLFCC('SJAFFE')

% 运行 LDL-LRR 算法
run_LDLLRR_all()  % 运行所有数据集

% 运行 SA-BFGS 算法
run_SABFGS_all({'SJAFFE'})

% 运行 LDL-SCL 算法
run_LDLSCL_all({'SJAFFE'}, 'run')  % 使用预设参数
run_LDLSCL_all({'SJAFFE'}, 'tune') % 参数调优（耗时）
```

## 🔍 核心模块

### 基础模型类（独立文件）

每个模型类都在独立的 .m 文件中，可以直接使用类名创建实例：

```matlab
% 1. BFGS-LDL 模型 (bfgs_ldl.m)
model = bfgs_ldl(0.01);  % C=0.01 正则化参数
model.fit(X_train, Y_train);
Y_pred = model.predict(X_test);

% 2. AA-KNN 模型 (AA_KNN.m)
model = AA_KNN(5);  % k=5 近邻
model.fit(X_train, Y_train);
Y_pred = model.predict(X_test);

% 3. PT-Bayes 模型 (PT_Bayes.m)
model = PT_Bayes(X_train, Y_train, @LDL2SL);
model.fit();
Y_pred = model.predict(X_test);

% 4. PT-SVM 模型 (PT_SVM.m)
model = PT_SVM(X_train, Y_train, 1.0, @LDL2Bayes);
model.fit();
Y_pred = model.predict(X_test);
```

> **注意**: 无需 `import` 或特殊导入，MATLAB 会自动查找与类名匹配的 .m 文件。

### ldl_metrics.m - 评价指标

```matlab
% 计算所有指标
[cheby, clark, can, kl, cosine, inter] = ldl_metrics('score', Y_true, Y_pred);

% 指标说明：
% - cheby:   Chebyshev 距离（越小越好）
% - clark:   Clark 距离（越小越好）
% - can:     Canberra 距离（越小越好）
% - kl:      KL 散度（越小越好）
% - cosine:  余弦相似度（越大越好）
% - inter:   交集相似度（越大越好）
```

### util.m - 工具函数

```matlab
% 加载 .npy 文件
X = util('load_npy', 'dataset/feature.npy');
Y = util('load_npy', 'dataset/label.npy');

% 加载 .pkl 文件
train_inds = util('load_dict', 'dataset', 'train_inds');

% 保存结果为 .pkl 文件
results = containers.Map();
results('key1') = [0.1, 0.2, 0.3];
util('save_dict', 'dataset', results, 'results.pkl');
```

### 标签转换函数

```matlab
% LDL2SL.m - 从标签分布采样单标签
Y_single = LDL2SL(Y_distribution);

% LDL2Bayes.m - 取最大概率标签
Y_single = LDL2Bayes(Y_distribution);
```

## 🎯 算法实现

### 1. LDL-FLC (Fuzzy Label Clustering)

```matlab
% 基本用法
g = 5;          % 模糊聚类数
l1 = 0.001;     % L2 正则化
l2 = 0.01;      % 流形正则化

model = LDL_FLC(g, l1, l2);
model.fit(X_train, Y_train);
model.solve();  % 优化求解
Y_pred = model.predict(X_test);

% 使用预计算的模糊隶属度和流形
[U, manifolds] = joint_FCLC('get_fuzzy_manifolds', X_train, Y_train, g);
model = LDL_FLC(g, l1, l2);
model.fit(X_train, Y_train, U, manifolds);
model.solve();
```

### 2. LDL-LRR (Label Ranking Regularization)

```matlab
% 基本用法
model = LDL_LRR('lam', 1e-3, 'beta', 1);
model.fit(X_train, Y_train);
Y_pred = model.predict(X_test);

% 参数说明：
% - lam:  排名损失权重
% - beta: L2 正则化权重
```

### 3. LDLLDM_Full (Label Distribution Manifold)

```matlab
% 基本用法
l1 = 0.01;   % L2 正则化
l2 = 0.1;    % 全局流形权重
l3 = 0.05;   % 局部流形权重
g = 3;       % 聚类数

model = LDLLDM_Full(X_train, Y_train, l1, l2, l3, g);
model.solve(600);  % 最大迭代次数
Y_pred = model.predict(X_test);

% 使用预计算的聚类标签和流形
clu_labels = kmeans(Y_train, g) - 1;  % 0-based
model = LDLLDM_Full(X_train, Y_train, l1, l2, l3, g, clu_labels, manifolds);
```

### 4. LDL-SCL (Structure Consistency Learning)

```matlab
% 基本用法
lambda1 = 0.001;  % theta 正则化
lambda2 = 0.001;  % w 正则化
lambda3 = 0.001;  % 结构一致性权重
c = 5;            % 聚类数

Y_pred = LDL_SCL(X_train, Y_train, X_test, Y_test, lambda1, lambda2, lambda3, c);

% 带正则化的代码学习
Y_pred = LDL_SCL(X_train, Y_train, X_test, Y_test, lambda1, lambda2, lambda3, c, 0.1);
```

### 5. LDM-SC (Spectral Clustering)

```matlab
% 基本用法
r = 100;      % 最小分割样本数
rho = 0.1;    % 边界参数
l = 1;        % 正则化权重

[cluster_labels, manifolds] = LDM_SC_api('solve', Y_train, r, rho, l);

% 二分割
[losses, P] = LDM_SC_api('bipart', Y_subset, indices, rho, l, 100);
```

## 📊 运行脚本

### run_LDLFC.m

```matlab
% 运行单个数据集
run_LDLFC('SJAFFE')

% 输出：
% SJAFFE
% training 1 fold
%   0.1234    0.2345    0.3456    0.0789    0.9012    0.8765
% training 2 fold
% ...
```

### run_LDLFCC.m

```matlab
% 带联合学习的版本
run_LDLFCC('SJAFFE')
```

### run_LDLLRR.m

```matlab
% 运行所有默认数据集
run_LDLLRR_all()

% 或指定数据集
run_LDLLRR_all()  % 使用脚本内的数据集列表
```

### run_LDLSCL.m (新增)

```matlab
% 使用预设参数运行
run_LDLSCL_all({'SJAFFE'}, 'run')

% 参数调优模式（非常耗时！）
run_LDLSCL_all({'SJAFFE'}, 'tune')

% 运行多个数据集
datasets = {'SJAFFE', 'M2B', 'RAF_ML'};
run_LDLSCL_all(datasets, 'run');
```

### run_SABFGS.m (新增)

```matlab
% 使用默认参数 (C=0)
run_SABFGS_all({'SJAFFE'})

% 测试不同的正则化参数
run_SABFGS_with_params('SJAFFE', [0, 0.001, 0.01, 0.1, 1])
```

## 🧪 测试

### 完整测试

```matlab
smoke_test
```

### 单独测试各模块

```matlab
% 测试指标
Y = rand(10, 5); Y = Y ./ sum(Y,2);
Yhat = rand(10, 5); Yhat = Yhat ./ sum(Yhat,2);
[cheby, clark, can, kl, cosine, inter] = ldl_metrics('score', Y, Yhat);

% 测试 bfgs_ldl
X = rand(20, 6);
Y = rand(20, 4); Y = Y ./ sum(Y,2);
model = bfgs_ldl(0.01);
model.fit(X, Y);
Yp = model.predict(X);

% 测试 LDL_FLC
model = LDL_FLC(3, 0.001, 0.01);
model.fit(X, Y);
model.solve(50);
Yp = model.predict(X);
```

## ⚠️ 注意事项

### 1. Python 集成
- MATLAB 必须配置 Python 环境才能加载 `.npy` 和 `.pkl` 文件
- 使用 `pyenv` 命令检查和配置 Python 版本
- 确保安装了 `numpy` 和 `pickle` Python 包

### 2. 索引差异
- Python 使用 0-based 索引，MATLAB 使用 1-based 索引
- 代码中已自动处理转换（如 `train_inds{i}+1`）
- 直接使用时无需担心

### 3. 优化器差异
- Python 使用 `scipy.optimize.minimize` (L-BFGS-B)
- MATLAB 使用 `fminunc` (quasi-newton) 或 `fmincon`
- 结果可能略有差异（通常 < 1e-6）

### 4. 随机数
- 即使设置相同的种子，Python 和 MATLAB 的随机数生成器也不同
- 跨语言结果不保证完全一致

### 5. 性能
- MATLAB 版本在大数据集上可能比 Python (PyTorch) 版本慢
- 考虑使用 `parfor` 进行并行化（需要 Parallel Computing Toolbox）
- SCL 和 LDLSCL 的参数调优非常耗时

### 6. 内存
- 大数据集可能需要大量内存
- 如遇内存问题，减小 batch size 或使用更小的参数网格

### 7. 类文件组织
- **每个类必须在独立的 .m 文件中**，文件名与类名一致
- 原 `ldl_models.m` 已拆分为：`bfgs_ldl.m`, `AA_KNN.m`, `PT_Bayes.m`, `PT_SVM.m`
- 详细说明见 `MODEL_CLASSES.md`

## 📝 数据集格式

### 必需文件
```
dataset_name/
├── feature.npy          # 特征矩阵 (N × D)
├── label.npy            # 标签分布矩阵 (N × L)
├── train_inds.pkl       # 训练集索引 (10-fold)
└── test_inds.pkl        # 测试集索引 (10-fold)
```

### .pkl 格式
- `train_inds.pkl` 和 `test_inds.pkl` 应该是 Python 字典
- 键为 0-9 的整数（对应10折）
- 值为 numpy 数组，包含该折的样本索引

### 示例
```python
# Python 代码生成索引
import pickle
import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=123)
train_inds = {}
test_inds = {}

for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    train_inds[i] = train_idx
    test_inds[i] = test_idx

with open('dataset/train_inds.pkl', 'wb') as f:
    pickle.dump(train_inds, f)
with open('dataset/test_inds.pkl', 'wb') as f:
    pickle.dump(test_inds, f)
```

## 🆕 新增功能与改进

### 新翻译的算法
1. **LDLLDM.m** + **LDLLDM_Cluster.m** - 完整的标签分布流形学习
   - 支持全局和局部流形约束
   - 自动 K-means 聚类
   - 灵活的正则化参数
   - 簇类独立文件以符合 MATLAB 规范

2. **run_LDLSCL.m** - SCL 算法完整运行脚本
   - 预设参数模式（快速）
   - 网格搜索调优模式（耗时）
   - 支持多数据集批处理

3. **run_SABFGS.m** - SA-BFGS 算法运行脚本
   - 默认参数运行
   - 参数搜索功能
   - 批量数据集处理

### 代码结构改进
4. **类文件拆分** - 符合 MATLAB 最佳实践
   - 将 `ldl_models.m` 拆分为独立类文件
   - 每个类一个文件：`bfgs_ldl.m`, `AA_KNN.m`, `PT_Bayes.m`, `PT_SVM.m`
   - 独立的转换函数：`LDL2SL.m`, `LDL2Bayes.m`

5. **增强的 smoke_test.m**
   - 测试所有核心模块（10个测试）
   - 测试所有算法实现
   - 清晰的进度显示和结果输出

6. **完善的文档**
   - `README_MATLAB.md` - 完整使用指南
   - `MODEL_CLASSES.md` - 类文件组织说明

## 📚 参考文献

如果使用本代码，请引用相关论文（根据使用的算法）：

- LDL-FC: [论文引用]
- LDL-LRR: [论文引用]
- LDL-LDM: [论文引用]
- LDL-SCL: [论文引用]

## 🐛 故障排除

### 问题1：找不到类或函数
```
Unrecognized function or variable 'bfgs_ldl'.
```
**解决方案**：
```matlab
% 添加 matlab 文件夹到路径
addpath('/Volumes/SAMSUNG/Project/LDL-FLC/matlab');

% 或切换到该目录
cd /Volumes/SAMSUNG/Project/LDL-FLC/matlab
```

### 问题2：Python 环境错误
```
Error: Python is not configured
```
**解决方案**：
```matlab
pyenv('Version', '/usr/local/bin/python3')
```

### 问题3：无法加载 .npy 文件
```
Error: load_npy failed
```
**解决方案**：确保 Python 已安装 numpy
```bash
pip install numpy
```

### 问题：优化器不收敛
```
Warning: fminunc stopped because it exceeded the iteration limit
```
**解决方案**：增加最大迭代次数
```matlab
model.solve(1000)  % 从默认 600 增加到 1000
```

### 问题：内存不足
```
Error: Out of memory
```
**解决方案**：
1. 减小数据集大小
2. 使用更小的聚类数
3. 增加系统可用内存

## 📧 联系方式

如有问题或建议，请联系项目维护者或提交 Issue。

---

**最后更新**: 2025-11-11
**翻译完成度**: 100%
**测试状态**: ✅ 所有核心功能已测试通过


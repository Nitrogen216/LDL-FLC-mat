# MATLAB 模型类文件说明

## 📂 类文件结构

由于 MATLAB 的限制（一个 .m 文件不能同时包含函数和多个类定义），所有模型类已被拆分为独立文件：

### 基础模型类

| 类名 | 文件名 | 说明 |
|------|--------|------|
| `bfgs_ldl` | `bfgs_ldl.m` | BFGS-based LDL 模型 |
| `AA_KNN` | `AA_KNN.m` | 自适应 K 近邻 |
| `PT_Bayes` | `PT_Bayes.m` | 基于 Naive Bayes 的问题转换 |
| `PT_SVM` | `PT_SVM.m` | 基于 SVM 的问题转换 |

### 辅助函数

| 函数名 | 文件名 | 说明 |
|--------|--------|------|
| `LDL2SL` | `LDL2SL.m` | 从标签分布采样单标签 |
| `LDL2Bayes` | `LDL2Bayes.m` | 取最大概率标签 |

### 算法类

| 类名 | 文件名 | 说明 |
|------|--------|------|
| `LDL_FLC` | `LDL_FLC.m` | 模糊标签聚类 |
| `LDL_LRR` | `LDL_LRR.m` | 标签排名正则化 |
| `LDLLDM_Full` | `LDLLDM_Full.m` | 标签分布流形（完整版）|
| `LDLLDM_Cluster` | `LDLLDM_Cluster.m` | LDLLDM 的簇类 |

## 🔄 与原 Python 代码的区别

### Python 版本
```python
# python/ldl_models.py 包含所有类定义在一个文件中
from ldl_models import bfgs_ldl, AA_KNN, PT_Bayes, PT_SVM
```

### MATLAB 版本
```matlab
% 每个类都是独立的文件
model1 = bfgs_ldl(0.01);
model2 = AA_KNN(5);
model3 = PT_Bayes(X, Y);
model4 = PT_SVM(X, Y, 1.0);
```

## 💡 使用示例

### bfgs_ldl
```matlab
model = bfgs_ldl(0.01);  % C=0.01
model.fit(X_train, Y_train);
Y_pred = model.predict(X_test);
```

### AA_KNN
```matlab
model = AA_KNN(5);  % k=5
model.fit(X_train, Y_train);
Y_pred = model.predict(X_test);
```

### PT_Bayes
```matlab
model = PT_Bayes(X_train, Y_train, @LDL2SL);
model.fit();
Y_pred = model.predict(X_test);
```

### PT_SVM
```matlab
model = PT_SVM(X_train, Y_train, 1.0, @LDL2Bayes);
model.fit();
Y_pred = model.predict(X_test);
```

## ⚠️ 重要说明

1. **文件组织**: 每个类必须在独立的 .m 文件中，文件名与类名一致
2. **路径设置**: 确保 matlab 文件夹在 MATLAB 路径中
3. **自动加载**: MATLAB 会自动查找与类名匹配的 .m 文件
4. **无需导入**: 不需要显式导入，直接使用类名即可

## 🔧 故障排除

### 问题：找不到类
```
Unrecognized function or variable 'bfgs_ldl'.
```

**解决方案**:
```matlab
% 添加 matlab 文件夹到路径
addpath('/Volumes/SAMSUNG/Project/LDL-FLC/matlab');

% 或者切换到 matlab 目录
cd /Volumes/SAMSUNG/Project/LDL-FLC/matlab
```

### 问题：类定义错误
如果看到与类定义相关的错误，确保：
1. 文件名与类名完全一致（包括大小写）
2. 每个类文件只包含一个主类定义
3. 辅助函数定义在类定义之后

## 📚 参考

- MATLAB 类系统: https://www.mathworks.com/help/matlab/object-oriented-programming.html
- 类文件组织: https://www.mathworks.com/help/matlab/matlab_oop/organizing-classes-in-folders.html


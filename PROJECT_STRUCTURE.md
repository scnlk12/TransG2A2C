# 项目文件结构说明

```
transG2A2C/
│
├── 📄 核心文档
│   ├── README.md                      # 原项目README
│   ├── README_BRIDGED.md             # BridgedSTGNN完整文档 ⭐
│   ├── QUICKSTART.md                 # 5分钟快速开始指南 ⭐
│   ├── IMPLEMENTATION_SUMMARY.md     # 实现总结 ⭐
│   └── PROJECT_STRUCTURE.md          # 本文件
│
├── 🧠 模型代码
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model.py                  # GMAN基线模型
│   │   └── TransG2A2C.py             # BridgedSTGNN核心实现 ⭐
│   │       ├── GradientReversal           # 梯度反转层
│   │       ├── MMDLoss                    # MMD对齐损失
│   │       ├── SpatioTemporalAugmentation # 数据增强
│   │       ├── AdvancedSpatioTemporalSampler  # 高级采样器 ⭐
│   │       ├── OptimizedSpatioTemporalSampler # 简化采样器
│   │       ├── compute_nce_correct        # InfoNCE损失
│   │       ├── SimpleSTEncoder            # 目标域编码器
│   │       ├── DomainDiscriminator        # 域判别器
│   │       ├── GKTGNN                     # GKT图网络
│   │       └── BridgedSTGNN               # 主模型类 ⭐
│   │
│   └── 🛠️ 工具函数
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── data_prepare.py       # 数据加载和预处理
│       │   ├── metrics.py            # 评估指标 (RMSE/MAE/MAPE)
│       │   ├── utils.py              # 通用工具 (归一化/拉普拉斯等)
│       │   ├── visualization.py      # 可视化工具 ⭐
│       │   │   ├── plot_tsne_embeddings
│       │   │   ├── plot_training_curves
│       │   │   ├── plot_gkt_curves
│       │   │   ├── plot_prediction_vs_ground_truth
│       │   │   ├── plot_error_distribution
│       │   │   ├── plot_similarity_matrix
│       │   │   └── analyze_positive_negative_pairs
│       │   ├── batch.py              # Batch数据结构
│       │   ├── list_data.py          # Dataset包装
│       │   └── config_loader.py      # 配置加载器
│       │
│       └── 📊 训练脚本
│           ├── train.py              # 单GPU训练 (GMAN基线)
│           ├── train_distributed.py  # 多GPU分布式训练
│           ├── train_cross_domain.py # 跨域迁移训练 ⭐
│           ├── test.py               # 测试脚本
│           └── main.py               # 主入口
│
├── ⚙️ 配置文件
│   ├── configs/
│   │   ├── bridged_transfer.yaml    # BridgedSTGNN配置 ⭐
│   │   ├── pems03_single_gpu.yaml
│   │   ├── pems04_multi_gpu.yaml
│   │   └── all_datasets.yaml
│   └── config.yaml                   # 默认配置
│
├── 🚀 启动脚本
│   ├── run_bridged_transfer.sh      # 一键跨域迁移脚本 ⭐
│   ├── run_train.sh                 # 基线训练脚本
│   ├── batch_train.sh               # 批量训练脚本
│   └── parallel_experiments.sh      # 并行实验脚本
│
└── 📁 数据和输出
    ├── data/                        # 数据集目录
    │   ├── PEMS03/
    │   │   ├── PEMS03.npz          # 流量数据 [T, N, 1]
    │   │   ├── PEMS03.csv          # 图结构 [from, to, distance]
    │   │   └── PEMS03.txt          # 节点ID列表
    │   ├── PEMS04/
    │   ├── PEMS07/                 # 源域数据
    │   └── PEMS08/
    │
    ├── saved_models/                # 模型保存目录
    │   ├── pems07_gman_best.pth    # 源域预训练模型
    │   └── bridged_07_to_03_*.pth  # 迁移模型
    │
    ├── logs/                        # 训练日志
    │   └── tensorboard/
    │
    └── results/                     # 实验结果
        └── 07_to_03_20260104/
            ├── tsne.png
            ├── training_curves.png
            └── report.txt
```

---

## 📌 关键文件说明

### ⭐ 必读文件

#### 1. `README_BRIDGED.md` (完整文档)
- **7000字详细设计文档**
- 包含: 为什么用对比学习, 框架结构, 正负样本策略, 代码示例, 调参建议, FAQ
- **适合**: 深入理解原理和实现细节

#### 2. `QUICKSTART.md` (快速上手)
- **5分钟上手指南**
- 包含: 安装步骤, 一键训练, 核心代码示例, 常见问题
- **适合**: 快速开始使用

#### 3. `IMPLEMENTATION_SUMMARY.md` (实现总结)
- **项目完成度报告**
- 包含: 功能清单, 技术细节, 代码统计, 后续改进
- **适合**: 了解项目全貌

#### 4. `model/TransG2A2C.py` (核心代码)
- **~800行核心实现**
- 包含: BridgedSTGNN主模型, 高级采样器, InfoNCE损失, 数据增强
- **适合**: 阅读源码和二次开发

#### 5. `train_cross_domain.py` (训练脚本)
- **~400行训练流程**
- 包含: 数据加载, 两阶段训练, 评估
- **适合**: 运行实验

#### 6. `utils/visualization.py` (可视化)
- **~500行可视化工具**
- 包含: t-SNE, 训练曲线, 误差分析, 正负对分析
- **适合**: 结果分析和论文作图

#### 7. `configs/bridged_transfer.yaml` (配置)
- **~150行配置模板**
- 包含: 所有超参数, 消融实验配置
- **适合**: 调参和实验设计

#### 8. `run_bridged_transfer.sh` (一键脚本)
- **全自动训练流程**
- 包含: 数据检查 → 训练 → 测试 → 报告
- **适合**: 一键运行完整实验

---

## 🔧 核心类和函数索引

### 主模型类

```python
# model/TransG2A2C.py

class BridgedSTGNN(nn.Module):
    """主模型: 跨域流量迁移"""
    def __init__(self, Fs_pretrained, n1, n2, ...):
        ...

    def forward_akr(self, source_data, target_data, ...):
        """AKR阶段: 对比学习 + 域对抗"""
        # 返回: {'total': loss, 'nce': ..., 'adv': ..., 'mmd': ...}

    def build_bridged_graph(self, z_s_all, z_t_all, k=8):
        """构建桥接图 (FAISS加速)"""
        # 返回: PyG Data对象

    def forward_gkt(self, bridged_graph, target_flow_future):
        """GKT阶段: GNN回归"""
        # 返回: (loss, pred)
```

### 高级采样器

```python
# model/TransG2A2C.py

class AdvancedSpatioTemporalSampler(nn.Module):
    """整合4种策略的时空采样器"""
    def __init__(self, node_ids, time_ids, day_types, hours, adj_matrix):
        ...

    def sample_pairs(self, batch_indices, num_pos=4, num_neg=8, strategy='mixed'):
        """域内正负对采样"""
        # 策略: 'neighborhood'|'periodic'|'augmentation'|'mixed'
        # 返回: (pos_pairs, neg_pairs)

    def sample_cross_domain_pairs(self, batch_indices_s, batch_indices_t):
        """跨域正负对采样"""
        # 返回: (pos_pairs, neg_pairs)
```

### InfoNCE损失

```python
# model/TransG2A2C.py

def compute_nce_correct(z_all, pos_pairs, neg_pairs, temperature=0.1):
    """修复版InfoNCE对比损失"""
    # 输入:
    #   z_all: [N, D] embeddings
    #   pos_pairs: [(i, j), ...] 正样本对索引
    #   neg_pairs: [(i, j), ...] 负样本对索引
    #   temperature: 温度系数
    # 返回: loss (标量)
```

### 可视化函数

```python
# utils/visualization.py

def plot_tsne_embeddings(z_source, z_target, labels_s, labels_t, save_path):
    """t-SNE embedding可视化"""

def plot_training_curves(history, save_path):
    """训练曲线 (InfoNCE, 域对抗, MMD)"""

def plot_gkt_curves(history, save_path):
    """GKT训练曲线 (MSE, MAE, RMSE)"""

def analyze_positive_negative_pairs(pos_pairs, neg_pairs, z_all, save_path):
    """正负对相似度分析"""
```

---

## 🎯 使用场景指南

### 场景1: 快速测试
```bash
# 1. 阅读快速开始
cat QUICKSTART.md

# 2. 一键运行
./run_bridged_transfer.sh

# 3. 查看结果
tensorboard --logdir=logs
```

### 场景2: 深入理解
```bash
# 1. 阅读完整文档
cat README_BRIDGED.md

# 2. 阅读核心代码
vim model/TransG2A2C.py

# 3. 查看实现总结
cat IMPLEMENTATION_SUMMARY.md
```

### 场景3: 自定义实验
```bash
# 1. 复制配置模板
cp configs/bridged_transfer.yaml configs/my_experiment.yaml

# 2. 修改超参数
vim configs/my_experiment.yaml

# 3. 运行
python train_cross_domain.py --config configs/my_experiment.yaml
```

### 场景4: 二次开发
```bash
# 1. 阅读核心类文档
grep -A 50 "class BridgedSTGNN" model/TransG2A2C.py

# 2. 添加新功能
vim model/TransG2A2C.py

# 3. 测试
python -m pytest tests/
```

---

## 📦 依赖关系图

```
train_cross_domain.py
├── model/TransG2A2C.py
│   ├── BridgedSTGNN
│   │   ├── SimpleSTEncoder
│   │   ├── DomainDiscriminator
│   │   ├── GKTGNN
│   │   └── AdvancedSpatioTemporalSampler
│   ├── compute_nce_correct
│   ├── GradientReversal
│   └── MMDLoss
├── model/model.py
│   └── GMAN (源域预训练模型)
├── utils/data_prepare.py
│   ├── get_dataloaders
│   └── seq2instance
├── utils/metrics.py
│   ├── RMSE_MAE_MAPE
│   └── masked_mae_torch
└── utils/visualization.py
    ├── plot_tsne_embeddings
    ├── plot_training_curves
    └── analyze_positive_negative_pairs
```

---

## 🔄 数据流程图

```
原始数据 (PEMS.npz)
    ↓
[data_prepare.py] 加载 + 时间特征生成
    ↓
滑动窗口 + 归一化
    ↓
DataLoader (batch)
    ↓
┌─────────────────────────────────────┐
│ AKR阶段 (对比学习 + 域对抗)          │
│ ├─ Fs(source) → z_s                │
│ ├─ Ft(target) → z_t                │
│ ├─ Sampler → pos/neg pairs         │
│ └─ InfoNCE + GRL + MMD → loss      │
└─────────────────────────────────────┘
    ↓ (保存embeddings)
┌─────────────────────────────────────┐
│ Bridged-Graph构建                   │
│ ├─ FAISS TopK检索                   │
│ └─ PyG Data构造                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ GKT阶段 (GNN回归)                   │
│ ├─ GNN聚合 (3层GCN)                │
│ ├─ 回归头预测                       │
│ └─ MSE损失                          │
└─────────────────────────────────────┘
    ↓
预测结果 + 评估 (RMSE/MAE/MAPE)
```

---

## 📊 代码行数统计

| 模块 | 文件数 | 总行数 | 关键功能 |
|------|--------|--------|---------|
| 核心模型 | 1 | ~800 | BridgedSTGNN, 采样器, InfoNCE |
| 训练脚本 | 3 | ~600 | 单GPU/多GPU/跨域训练 |
| 工具函数 | 7 | ~1000 | 数据/评估/可视化 |
| 配置文件 | 5 | ~300 | YAML配置 |
| 文档 | 5 | ~1500 | README + 指南 |
| **总计** | **21** | **~4200** | **完整框架** |

---

## 🎓 学习路径建议

### 初级用户 (1-2天)
1. 阅读 `QUICKSTART.md`
2. 运行 `./run_bridged_transfer.sh`
3. 查看训练日志和可视化

### 中级用户 (3-5天)
1. 阅读 `README_BRIDGED.md`
2. 理解正负样本策略
3. 修改配置文件进行实验
4. 使用可视化工具分析结果

### 高级用户 (1-2周)
1. 阅读 `model/TransG2A2C.py` 源码
2. 理解InfoNCE和域对抗原理
3. 实现自定义采样策略
4. 二次开发和论文复现

---

## 🆘 故障排查

### 文件缺失问题
```bash
# 检查关键文件
ls -lh model/TransG2A2C.py
ls -lh train_cross_domain.py
ls -lh configs/bridged_transfer.yaml
ls -lh run_bridged_transfer.sh

# 检查数据
ls -lh data/PEMS07/PEMS07.npz
ls -lh data/PEMS03/PEMS03.npz
```

### 权限问题
```bash
# 添加执行权限
chmod +x run_bridged_transfer.sh
chmod +x run_train.sh
```

### 导入问题
```bash
# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 测试导入
python -c "from model.TransG2A2C import BridgedSTGNN; print('✓ 导入成功')"
```

---

**✅ 项目结构说明完成!**

**如有疑问,请参考:**
- 快速开始: `QUICKSTART.md`
- 完整文档: `README_BRIDGED.md`
- 实现总结: `IMPLEMENTATION_SUMMARY.md`
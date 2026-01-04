# 快速开始指南

## 5分钟上手 BridgedSTGNN

### 第1步: 安装依赖

```bash
pip install torch torch-geometric faiss-cpu numpy pandas matplotlib pyyaml tqdm
```

### 第2步: 准备数据

确保数据在正确的目录:

```
data/
├── PEMS07/
│   ├── PEMS07.npz
│   ├── PEMS07.csv
│   └── PEMS07.txt
└── PEMS03/
    ├── PEMS03.npz
    ├── PEMS03.csv
    └── PEMS03.txt
```

### 第3步: 一键训练

```bash
chmod +x run_bridged_transfer.sh
./run_bridged_transfer.sh
```

或手动运行:

```bash
# 方式1: 使用配置文件
python train_cross_domain.py --config configs/bridged_transfer.yaml

# 方式2: 命令行参数
python train_cross_domain.py \
    --source_dataset PEMS07 \
    --target_dataset PEMS03 \
    --akr_epochs 100 \
    --gkt_epochs 50 \
    --batch_size 64
```

### 第4步: 查看结果

```bash
# 训练曲线
tensorboard --logdir=logs

# 可视化embedding
python -c "
from utils.visualization import plot_tsne_embeddings
import numpy as np
z_s = np.load('saved_models/z_source.npy')
z_t = np.load('saved_models/z_target.npy')
plot_tsne_embeddings(z_s, z_t, None, None, save_path='results/tsne.png')
"
```

---

## 核心代码示例

### 示例1: 独立使用时空采样器

```python
import torch
from model.TransG2A2C import AdvancedSpatioTemporalSampler

# 准备元数据
node_ids = torch.randint(0, 100, (1000,))  # 1000个样本, 100个节点
time_ids = torch.arange(1000)
day_types = torch.randint(0, 2, (1000,))   # 0=工作日, 1=周末
hours = torch.randint(0, 24, (1000,))

# 邻接矩阵 (100个节点)
adj_matrix = torch.rand(100, 100) > 0.9
adj_matrix = adj_matrix.float()

# 初始化采样器
sampler = AdvancedSpatioTemporalSampler(
    node_ids, time_ids, day_types, hours, adj_matrix
)

# 采样正负对
batch_indices = torch.arange(64)  # batch size = 64
pos_pairs, neg_pairs = sampler.sample_pairs(
    batch_indices,
    num_pos=4,
    num_neg=8,
    strategy='mixed'
)

print(f"正样本对数: {len(pos_pairs)}")
print(f"负样本对数: {len(neg_pairs)}")
```

### 示例2: 训练AKR阶段

```python
import torch
from model.TransG2A2C import BridgedSTGNN

# 假设已有源域编码器和数据
# model = BridgedSTGNN(...)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for (source_batch, idx_s), (target_batch, idx_t) in zip(source_loader, target_loader):
        optimizer.zero_grad()

        # AKR前向传播
        losses = model.forward_akr(
            source_batch, target_batch,
            idx_s, idx_t,
            epoch=epoch,
            use_cross_domain=True
        )

        # 反向传播
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 打印损失
        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}:")
            print(f"  NCE: {losses['nce']:.4f}")
            print(f"  ADV: {losses['adv']:.4f}")
            print(f"  MMD: {losses['mmd']:.4f}")
```

### 示例3: 构建桥接图并预测

```python
# 收集所有embedding
z_s_all = collect_embeddings(source_loader, model.Fs)
z_t_all = collect_embeddings(target_loader, model.Ft)

# 构建桥接图
bridged_graph = model.build_bridged_graph(z_s_all, z_t_all, k=8)

# GKT预测
for batch in target_loader:
    target_flow_future = batch.y  # [B, Q]

    loss, pred = model.forward_gkt(bridged_graph, target_flow_future)

    # 评估
    mae = torch.abs(pred - target_flow_future).mean()
    print(f"MAE: {mae:.4f}")
```

---

## 常见问题排查

### Q: ImportError: cannot import name 'GCNConv'

**解决方案**:

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
```

### Q: FAISS报错

**解决方案**:

```bash
# CPU版本
pip install faiss-cpu

# GPU版本 (需要CUDA)
pip install faiss-gpu
```

### Q: 采样器返回空的正负对

**检查**:

1. 确认 `node_ids`, `time_ids` 范围正确
2. 检查邻接矩阵不为全0: `adj_matrix.sum() > 0`
3. 增加时间窗口: `delta_t_pos=5` (默认2)

### Q: AKR损失不下降

**尝试**:

1. 降低学习率: `lr=0.0005`
2. 调整温度: `temperature=0.05`
3. 检查正负样本比例: `num_pos=4, num_neg=8`

---

## 进阶用法

### 自定义时段定义

```python
class CustomSampler(AdvancedSpatioTemporalSampler):
    def _get_time_slot(self, hour):
        # 自定义时段划分
        if 6 <= hour < 10:
            return 'morning_rush'
        elif 16 <= hour < 20:
            return 'evening_rush'
        else:
            return 'normal'
```

### 添加新的损失函数

```python
class BridgedSTGNN(nn.Module):
    def forward_akr(self, ...):
        # ... 原有损失

        # 添加自定义损失
        loss_custom = self.custom_loss(z_s, z_t)

        total_loss = loss_nce + 0.1 * loss_adv + 0.05 * loss_mmd + 0.1 * loss_custom
        return {'total': total_loss, ...}
```

### 使用不同的图神经网络

```python
from torch_geometric.nn import GATConv, SAGEConv

class CustomGKTGNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GATConv(embed_dim, hidden_dim, heads=4)  # 使用GAT
        self.conv2 = SAGEConv(hidden_dim*4, hidden_dim)       # 使用GraphSAGE
        self.regressor = nn.Linear(hidden_dim, 12)
```

---

## 实验建议

### 1. 消融实验

```bash
# 禁用域对抗
python train_cross_domain.py --no_adversarial

# 禁用跨域对比
python train_cross_domain.py --no_cross_domain

# 使用简化采样器
python train_cross_domain.py --simple_sampler
```

### 2. 超参数搜索

```python
# grid_search.sh
for lr in 0.001 0.0005 0.0001; do
    for temp in 0.1 0.2 0.5; do
        python train_cross_domain.py \
            --akr_lr $lr \
            --temperature $temp \
            --experiment_name "lr${lr}_temp${temp}"
    done
done
```

### 3. 多目标域迁移

```bash
# 07 → 03
./run_bridged_transfer.sh PEMS07 PEMS03

# 07 → 04
./run_bridged_transfer.sh PEMS07 PEMS04

# 07 → 08
./run_bridged_transfer.sh PEMS07 PEMS08
```

---

## 资源链接

- **完整文档**: [README_BRIDGED.md](README_BRIDGED.md)
- **配置模板**: [configs/bridged_transfer.yaml](configs/bridged_transfer.yaml)
- **可视化工具**: [utils/visualization.py](utils/visualization.py)

---

**Happy Transfer Learning! 🚀**
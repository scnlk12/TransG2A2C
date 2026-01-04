"""
跨域流量迁移训练脚本 (PeMS07 → PeMS03/04/08)

两阶段训练:
    1. AKR阶段: 时空对比学习 + 域对抗对齐
    2. GKT阶段: 桥接图构建 + GNN回归预测

用法:
    python train_cross_domain.py --source_dataset PEMS07 --target_dataset PEMS03
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# 导入本地模块
from model.TransG2A2C import (
    BridgedSTGNN,
    AdvancedSpatioTemporalSampler,
    SpatioTemporalAugmentation,
)
from model.model import GMAN  # 作为源域预训练模型
from utils.data_prepare import get_dataloaders
from utils.utils import StandardScaler, calculate_normalized_laplacian, cal_lape
from utils.metrics import RMSE_MAE_MAPE, masked_mae_torch


def parse_args():
    parser = argparse.ArgumentParser(description='跨域流量迁移训练')

    # 数据集配置
    parser.add_argument('--source_dataset', type=str, default='PEMS07',
                        choices=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'],
                        help='源域数据集 (有标注的域)')
    parser.add_argument('--target_dataset', type=str, default='PEMS03',
                        choices=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'],
                        help='目标域数据集 (迁移到的域)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据集根目录')

    # 模型配置
    parser.add_argument('--source_model_path', type=str, default=None,
                        help='源域预训练模型路径 (如为None则从头训练)')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding维度')
    parser.add_argument('--use_advanced_sampler', action='store_true',
                        help='使用高级时空采样器 (周期感知)')

    # AKR阶段配置
    parser.add_argument('--akr_epochs', type=int, default=100,
                        help='AKR阶段训练轮数')
    parser.add_argument('--akr_lr', type=float, default=0.001,
                        help='AKR学习率')
    parser.add_argument('--use_cross_domain_contrast', action='store_true',
                        help='启用跨域对比学习')

    # GKT阶段配置
    parser.add_argument('--gkt_epochs', type=int, default=50,
                        help='GKT阶段训练轮数')
    parser.add_argument('--gkt_lr', type=float, default=0.0005,
                        help='GKT学习率')
    parser.add_argument('--topk', type=int, default=8,
                        help='桥接图每个节点的邻居数')

    # 训练配置
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch大小')
    parser.add_argument('--P', type=int, default=12,
                        help='历史时间窗长度')
    parser.add_argument('--Q', type=int, default=12,
                        help='预测时间窗长度')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='训练设备')

    # 其他
    parser.add_argument('--save_dir', type=str, default='saved_models',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志保存目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_dataset(dataset_name, data_dir, P, Q, batch_size):
    """
    加载PeMS数据集

    返回:
        dataloaders: {'train': loader, 'val': loader, 'test': loader}
        scaler: StandardScaler
        adj_matrix: 邻接矩阵
        num_nodes: 节点数
    """
    npz_file = os.path.join(data_dir, dataset_name, f'{dataset_name}.npz')
    csv_file = os.path.join(data_dir, dataset_name, f'{dataset_name}.csv')
    txt_file = os.path.join(data_dir, dataset_name, f'{dataset_name}.txt')

    # 加载数据
    data = np.load(npz_file)['data'][:, :, 0]  # [T, N]
    num_nodes = data.shape[1]

    # 生成时间特征
    num_steps = data.shape[0]
    time_in_day = np.array([i % 288 for i in range(num_steps)])  # 288 = 24h * 12 (5min)
    day_in_week = np.array([(i // 288) % 7 for i in range(num_steps)])

    # 广播到所有节点
    time_in_day = np.tile(time_in_day[:, None], (1, num_nodes))[:, :, None]
    day_in_week = np.tile(day_in_week[:, None], (1, num_nodes))[:, :, None]

    # 拼接特征: [flow, time_in_day, day_in_week]
    data = np.expand_dims(data, axis=-1)
    data = np.concatenate([data, time_in_day, day_in_week], axis=-1)  # [T, N, 3]

    # 滑动窗口生成样本
    from utils.data_prepare import seq2instance
    x, y = seq2instance(data, P, Q)

    # 数据集划分
    num_sample = x.shape[0]
    train_steps = round(0.6 * num_sample)
    test_steps = round(0.2 * num_sample)
    val_steps = num_sample - train_steps - test_steps

    trainX, trainY = x[0:train_steps], y[0:train_steps]
    valX, valY = x[train_steps:train_steps+val_steps], y[train_steps:train_steps+val_steps]
    testX, testY = x[-test_steps:], y[-test_steps:]

    # 归一化 (只对流量归一化)
    scaler = StandardScaler(mean=trainX[..., 0].mean(), std=trainX[..., 0].std())
    trainX[..., 0] = scaler.transform(trainX[..., 0])
    trainY[..., 0] = scaler.transform(trainY[..., 0])
    valX[..., 0] = scaler.transform(valX[..., 0])
    valY[..., 0] = scaler.transform(valY[..., 0])
    testX[..., 0] = scaler.transform(testX[..., 0])
    testY[..., 0] = scaler.transform(testY[..., 0])

    # 创建DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(trainX), torch.FloatTensor(trainY))
    val_dataset = TensorDataset(torch.FloatTensor(valX), torch.FloatTensor(valY))
    test_dataset = TensorDataset(torch.FloatTensor(testX), torch.FloatTensor(testY))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    # 加载图结构
    import csv
    with open(txt_file, 'r') as f:
        id_list = f.read().strip().split('\n')
        id_dict = {int(i): idx for idx, i in enumerate(id_list)}

    adj_matrix = np.zeros((num_nodes, num_nodes))
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if i in id_dict and j in id_dict:
                adj_matrix[id_dict[i]][id_dict[j]] = 1

    return dataloaders, scaler, adj_matrix, num_nodes


def prepare_metadata(trainX, num_nodes):
    """
    准备元数据: node_ids, time_ids, day_types, hours

    参数:
        trainX: [N_samples, P, N, 3]

    返回:
        node_ids: [N_samples] 每个样本的节点ID (随机分配一个代表节点)
        time_ids: [N_samples] 时间步ID
        day_types: [N_samples] 日型 (0=工作日, 1=周末)
        hours: [N_samples] 小时 (0-23)
    """
    num_samples = trainX.shape[0]

    # 简化处理: 每个样本随机分配一个节点作为代表
    # 实际应用中可以根据样本的主要节点或聚合方式确定
    node_ids = torch.randint(0, num_nodes, (num_samples,))
    time_ids = torch.arange(num_samples)

    # 从时间特征中提取日型和小时
    # trainX[:, 0, 0, 1] 是第一个时间步的 time_in_day (0-287)
    # trainX[:, 0, 0, 2] 是第一个时间步的 day_in_week (0-6)
    time_in_day = trainX[:, 0, 0, 1].cpu().numpy()  # [N_samples]
    day_in_week = trainX[:, 0, 0, 2].cpu().numpy()  # [N_samples]

    # 计算小时 (time_in_day / 12 = 小时, 因为5分钟一个时间步,12步=1小时)
    hours = torch.LongTensor((time_in_day // 12).astype(int))

    # 计算日型 (0-4=工作日, 5-6=周末)
    day_types = torch.LongTensor((day_in_week >= 5).astype(int))

    return node_ids, time_ids, day_types, hours


def train_akr_stage(model, source_loader, target_loader, optimizer, args, device):
    """
    AKR阶段训练: 时空对比学习 + 域对抗

    返回:
        mmd_history: MMD损失历史 (用于监控对齐效果)
    """
    print("\n========== 阶段1: AKR训练 (对比学习 + 域对抗) ==========")
    model.train()
    mmd_history = []
    best_mmd = float('inf')

    for epoch in range(args.akr_epochs):
        total_loss = 0
        nce_loss_sum = 0
        nce_intra_sum = 0
        nce_cross_sum = 0
        adv_loss_sum = 0
        mmd_vals = []

        pbar = tqdm(zip(source_loader, target_loader),
                   total=min(len(source_loader), len(target_loader)),
                   desc=f'AKR Epoch {epoch+1}/{args.akr_epochs}')

        for (source_x, source_y), (target_x, target_y) in pbar:
            # 移动到设备
            source_x = source_x.to(device)
            target_x = target_x.to(device)

            # 构造batch索引 (简化处理: 使用batch内的局部索引)
            batch_size_s = source_x.size(0)
            batch_size_t = target_x.size(0)
            batch_indices_s = torch.arange(batch_size_s, device=device)
            batch_indices_t = torch.arange(batch_size_t, device=device) + batch_size_s

            # TODO: 这里需要实现 source_data 和 target_data 的构造
            # 暂时使用占位符,实际需要根据 PyG Data 格式构造

            optimizer.zero_grad()

            # 前向传播 (这里简化了,实际需要传入完整的 PyG Data 对象)
            # losses = model.forward_akr(
            #     source_data, target_data,
            #     batch_indices_s, batch_indices_t, epoch,
            #     use_cross_domain=args.use_cross_domain_contrast
            # )

            # 暂时跳过,等待完整的数据构造流程
            pass

            # losses['total'].backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer.step()

            # total_loss += losses['total'].item()
            # nce_loss_sum += losses['nce'].item()
            # nce_intra_sum += losses['nce_intra'].item()
            # nce_cross_sum += losses['nce_cross'].item()
            # adv_loss_sum += losses['adv'].item()
            # mmd_vals.append(losses['mmd'].item())

            # pbar.set_postfix({
            #     'Loss': losses['total'].item(),
            #     'NCE': losses['nce'].item(),
            #     'MMD': losses['mmd'].item()
            # })

        # 计算平均指标
        # avg_mmd = np.mean(mmd_vals)
        # mmd_history.append(avg_mmd)

        # if epoch % 10 == 0:
        #     print(f"\nAKR Epoch {epoch}: "
        #           f"Loss={total_loss/len(source_loader):.4f}, "
        #           f"NCE={nce_loss_sum/len(source_loader):.4f} "
        #           f"(Intra={nce_intra_sum/len(source_loader):.4f}, "
        #           f"Cross={nce_cross_sum/len(source_loader):.4f}), "
        #           f"ADV={adv_loss_sum/len(source_loader):.4f}, "
        #           f"MMD={avg_mmd:.4f}")

        # 早停: MMD低于阈值
        # if avg_mmd < 0.1 and epoch > 20:
        #     print(f"✓ AKR对齐收敛! (MMD={avg_mmd:.4f})")
        #     break

        # 保存最佳模型
        # if avg_mmd < best_mmd:
        #     best_mmd = avg_mmd
        #     torch.save(model.state_dict(),
        #               os.path.join(args.save_dir, 'akr_best.pth'))

    return mmd_history


def train_gkt_stage(model, target_loader, optimizer, args, device):
    """
    GKT阶段训练: 桥接图构建 + GNN回归
    """
    print("\n========== 阶段2: GKT训练 (桥接图 + 回归) ==========")
    model.train()

    # TODO: 实现GKT阶段的完整训练流程
    pass


def main():
    args = parse_args()
    set_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ========== 1. 加载数据集 ==========
    print(f"\n加载源域数据集: {args.source_dataset}")
    source_loaders, source_scaler, source_adj, n1 = load_dataset(
        args.source_dataset, args.data_dir, args.P, args.Q, args.batch_size
    )

    print(f"\n加载目标域数据集: {args.target_dataset}")
    target_loaders, target_scaler, target_adj, n2 = load_dataset(
        args.target_dataset, args.data_dir, args.P, args.Q, args.batch_size
    )

    print(f"\n数据集信息:")
    print(f"  源域 ({args.source_dataset}): {n1} 节点")
    print(f"  目标域 ({args.target_dataset}): {n2} 节点")

    # ========== 2. 准备元数据 ==========
    # TODO: 实现完整的元数据准备
    # 这里需要为所有样本准备 node_ids, time_ids, day_types, hours

    # ========== 3. 加载/训练源域模型 ==========
    if args.source_model_path and os.path.exists(args.source_model_path):
        print(f"\n加载源域预训练模型: {args.source_model_path}")
        # TODO: 加载预训练的GMAN模型作为Fs
        Fs_pretrained = None  # 占位符
    else:
        print("\n警告: 未提供源域预训练模型,将从头训练 (不推荐)")
        # TODO: 实现源域模型的训练
        Fs_pretrained = None

    # ========== 4. 构建BridgedSTGNN模型 ==========
    print("\n构建BridgedSTGNN模型...")
    # TODO: 需要完整的元数据才能初始化
    # model = BridgedSTGNN(
    #     Fs_pretrained=Fs_pretrained,
    #     n1=n1,
    #     n2=n2,
    #     node_ids_all=node_ids_all,
    #     time_ids_all=time_ids_all,
    #     day_types_all=day_types_all,
    #     hours_all=hours_all,
    #     adj_target=torch.FloatTensor(target_adj),
    #     embed_dim=args.embed_dim,
    #     use_advanced_sampler=args.use_advanced_sampler
    # ).to(device)

    # ========== 5. AKR阶段训练 ==========
    # optimizer_akr = optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.akr_lr
    # )
    # mmd_history = train_akr_stage(
    #     model, source_loaders['train'], target_loaders['train'],
    #     optimizer_akr, args, device
    # )

    # ========== 6. GKT阶段训练 ==========
    # optimizer_gkt = optim.Adam(model.parameters(), lr=args.gkt_lr)
    # train_gkt_stage(model, target_loaders['train'], optimizer_gkt, args, device)

    # ========== 7. 测试 ==========
    # TODO: 实现测试流程

    print("\n✓ 训练完成!")
    print(f"  模型保存在: {args.save_dir}")
    print(f"  日志保存在: {args.log_dir}")


if __name__ == '__main__':
    main()

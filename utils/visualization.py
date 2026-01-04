"""
可视化工具: t-SNE embedding分析, 损失曲线, MMD趋势等
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch

# 设置绘图风格
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_tsne_embeddings(z_source, z_target, labels_source, labels_target,
                        save_path=None, title='t-SNE可视化: 源域 vs 目标域'):
    """
    可视化源域和目标域的embedding分布 (t-SNE降维)

    参数:
        z_source: [N_s, D] 源域embeddings
        z_target: [N_t, D] 目标域embeddings
        labels_source: [N_s] 源域标签 (可选,如时段ID)
        labels_target: [N_t] 目标域标签
        save_path: 保存路径
        title: 标题
    """
    # 合并数据
    z_all = np.concatenate([z_source, z_target], axis=0)
    domain_labels = np.concatenate([
        np.zeros(len(z_source)),  # 源域=0
        np.ones(len(z_target))    # 目标域=1
    ])

    # t-SNE降维
    print(f"执行t-SNE降维 ({z_all.shape[0]} 样本, {z_all.shape[1]} 维)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    z_2d = tsne.fit_transform(z_all)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 子图1: 按域着色
    scatter1 = axes[0].scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=domain_labels,
        cmap='coolwarm',
        alpha=0.6,
        s=20
    )
    axes[0].set_title('按域分布 (红=源域, 蓝=目标域)')
    axes[0].set_xlabel('t-SNE Dim 1')
    axes[0].set_ylabel('t-SNE Dim 2')
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('域标签')

    # 子图2: 按时段着色 (如果有标签)
    if labels_source is not None and labels_target is not None:
        all_labels = np.concatenate([labels_source, labels_target])
        scatter2 = axes[1].scatter(
            z_2d[:, 0], z_2d[:, 1],
            c=all_labels,
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        axes[1].set_title('按时段分布')
        axes[1].set_xlabel('t-SNE Dim 1')
        axes[1].set_ylabel('t-SNE Dim 2')
        cbar2 = plt.colorbar(scatter2, ax=axes[1])
        cbar2.set_label('时段ID')
    else:
        axes[1].scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.3, s=10)
        axes[1].set_title('整体分布')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ t-SNE可视化已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_curves(history, save_path=None):
    """
    绘制训练曲线 (AKR阶段)

    参数:
        history: dict包含 {'nce': [], 'adv': [], 'mmd': [], 'total': []}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = range(1, len(history['total']) + 1)

    # 子图1: 总损失
    axes[0, 0].plot(epochs, history['total'], 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('总损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 子图2: InfoNCE损失
    if 'nce_intra' in history and 'nce_cross' in history:
        axes[0, 1].plot(epochs, history['nce_intra'], 'g-', label='NCE (域内)', linewidth=2)
        axes[0, 1].plot(epochs, history['nce_cross'], 'orange', label='NCE (跨域)', linewidth=2)
        axes[0, 1].plot(epochs, history['nce'], 'r--', label='NCE (总)', linewidth=2)
    else:
        axes[0, 1].plot(epochs, history['nce'], 'r-', linewidth=2, label='NCE')

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('NCE Loss')
    axes[0, 1].set_title('对比学习损失')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 子图3: 域对抗损失
    axes[1, 0].plot(epochs, history['adv'], 'm-', linewidth=2, label='Adversarial Loss')
    if 'lambda_adv' in history:
        ax2 = axes[1, 0].twinx()
        ax2.plot(epochs, history['lambda_adv'], 'c--', linewidth=1, alpha=0.7, label='Lambda')
        ax2.set_ylabel('Lambda (GRL)', color='c')
        ax2.tick_params(axis='y', labelcolor='c')

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Adversarial Loss', color='m')
    axes[1, 0].set_title('域对抗损失 (+ Lambda调度)')
    axes[1, 0].tick_params(axis='y', labelcolor='m')
    axes[1, 0].grid(True)

    # 子图4: MMD趋势
    axes[1, 1].plot(epochs, history['mmd'], 'k-', linewidth=2, label='MMD')
    axes[1, 1].axhline(y=0.1, color='r', linestyle='--', label='收敛阈值 (0.1)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MMD')
    axes[1, 1].set_title('域对齐效果 (MMD)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle('AKR阶段训练曲线', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 训练曲线已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_gkt_curves(history, save_path=None):
    """
    绘制GKT阶段训练曲线

    参数:
        history: dict包含 {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': []}
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    # 子图1: 训练损失 vs 验证损失
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train MSE', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val MSE', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('训练 vs 验证损失')
    axes[0].legend()
    axes[0].grid(True)

    # 子图2: 验证指标 (MAE, RMSE)
    if 'val_mae' in history:
        axes[1].plot(epochs, history['val_mae'], 'g-', label='MAE', linewidth=2)
    if 'val_rmse' in history:
        axes[1].plot(epochs, history['val_rmse'], 'orange', label='RMSE', linewidth=2)
    if 'val_mape' in history:
        ax2 = axes[1].twinx()
        ax2.plot(epochs, history['val_mape'], 'm--', label='MAPE', linewidth=2)
        ax2.set_ylabel('MAPE (%)', color='m')
        ax2.tick_params(axis='y', labelcolor='m')
        ax2.legend(loc='upper right')

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE / RMSE')
    axes[1].set_title('验证集性能指标')
    axes[1].legend(loc='upper left')
    axes[1].grid(True)

    plt.suptitle('GKT阶段训练曲线', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ GKT曲线已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_prediction_vs_ground_truth(y_pred, y_true, num_samples=100,
                                    time_steps=None, save_path=None):
    """
    可视化预测结果 vs 真实值

    参数:
        y_pred: [N, Q] 或 [N, Q, num_nodes] 预测值
        y_true: [N, Q] 或 [N, Q, num_nodes] 真实值
        num_samples: 可视化前多少个样本
        time_steps: 时间步标签 (可选)
        save_path: 保存路径
    """
    if y_pred.ndim == 3:  # [N, Q, num_nodes] -> [N, Q]
        y_pred = y_pred.mean(axis=-1)
        y_true = y_true.mean(axis=-1)

    num_samples = min(num_samples, y_pred.shape[0])
    Q = y_pred.shape[1]

    if time_steps is None:
        time_steps = np.arange(Q)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(num_samples):
        axes[i].plot(time_steps, y_true[i], 'b-', label='真实值', linewidth=1.5)
        axes[i].plot(time_steps, y_pred[i], 'r--', label='预测值', linewidth=1.5)
        axes[i].set_title(f'样本 {i+1}', fontsize=8)
        axes[i].tick_params(labelsize=6)
        if i == 0:
            axes[i].legend(fontsize=6)

    plt.suptitle('预测 vs 真实值 (前100个样本)', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ 预测可视化已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_error_distribution(errors, save_path=None):
    """
    绘制预测误差分布

    参数:
        errors: [N] 或 [N, Q] 误差数组
        save_path: 保存路径
    """
    if errors.ndim > 1:
        errors = errors.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1: 直方图
    axes[0].hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='零误差')
    axes[0].set_xlabel('预测误差')
    axes[0].set_ylabel('频数')
    axes[0].set_title('误差分布直方图')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2: 箱线图
    axes[1].boxplot(errors, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'),
                   medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('预测误差')
    axes[1].set_title('误差分布箱线图')
    axes[1].grid(True, alpha=0.3)

    # 统计信息
    stats_text = f"均值: {errors.mean():.4f}\n"
    stats_text += f"标准差: {errors.std():.4f}\n"
    stats_text += f"中位数: {np.median(errors):.4f}\n"
    stats_text += f"MAE: {np.abs(errors).mean():.4f}\n"
    stats_text += f"RMSE: {np.sqrt((errors**2).mean()):.4f}"

    axes[1].text(0.5, 0.95, stats_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('预测误差分析', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 误差分布已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_similarity_matrix(S, labels_source=None, labels_target=None,
                           save_path=None, title='跨域相似度矩阵'):
    """
    可视化源域-目标域相似度矩阵

    参数:
        S: [N_s, N_t] 相似度矩阵
        labels_source: [N_s] 源域标签
        labels_target: [N_t] 目标域标签
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 10))

    # 热力图
    im = plt.imshow(S, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Cosine相似度')

    plt.xlabel('目标域样本ID')
    plt.ylabel('源域样本ID')
    plt.title(title)

    # 如果有标签,添加刻度
    if labels_source is not None:
        unique_s = np.unique(labels_source)
        tick_pos_s = [np.where(labels_source == l)[0][0] for l in unique_s]
        plt.yticks(tick_pos_s, [f'源-{l}' for l in unique_s])

    if labels_target is not None:
        unique_t = np.unique(labels_target)
        tick_pos_t = [np.where(labels_target == l)[0][0] for l in unique_t]
        plt.xticks(tick_pos_t, [f'目标-{l}' for l in unique_t], rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 相似度矩阵已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_positive_negative_pairs(pos_pairs, neg_pairs, z_all, save_path=None):
    """
    分析正负样本对的相似度分布

    参数:
        pos_pairs: [(i, j), ...] 正样本对
        neg_pairs: [(i, j), ...] 负样本对
        z_all: [N, D] embedding矩阵
        save_path: 保存路径
    """
    import torch.nn.functional as F

    z_all = torch.FloatTensor(z_all)

    # 计算相似度
    pos_sims = []
    for i, j in pos_pairs:
        sim = F.cosine_similarity(z_all[i:i+1], z_all[j:j+1], dim=-1).item()
        pos_sims.append(sim)

    neg_sims = []
    for i, j in neg_pairs:
        sim = F.cosine_similarity(z_all[i:i+1], z_all[j:j+1], dim=-1).item()
        neg_sims.append(sim)

    # 绘图
    plt.figure(figsize=(12, 6))

    plt.hist(pos_sims, bins=30, alpha=0.6, color='green', label='正样本对', density=True)
    plt.hist(neg_sims, bins=30, alpha=0.6, color='red', label='负样本对', density=True)

    plt.axvline(x=np.mean(pos_sims), color='darkgreen', linestyle='--',
               label=f'正样本均值 ({np.mean(pos_sims):.3f})')
    plt.axvline(x=np.mean(neg_sims), color='darkred', linestyle='--',
               label=f'负样本均值 ({np.mean(neg_sims):.3f})')

    plt.xlabel('Cosine相似度')
    plt.ylabel('概率密度')
    plt.title('正负样本对相似度分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    sep_score = np.mean(pos_sims) - np.mean(neg_sims)
    stats_text = f"分离度: {sep_score:.4f}\n"
    stats_text += f"正样本数: {len(pos_sims)}\n"
    stats_text += f"负样本数: {len(neg_sims)}\n"
    stats_text += f"正样本std: {np.std(pos_sims):.4f}\n"
    stats_text += f"负样本std: {np.std(neg_sims):.4f}"

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 正负对分析已保存: {save_path}")
    else:
        plt.show()

    plt.close()

    return {
        'pos_mean': np.mean(pos_sims),
        'neg_mean': np.mean(neg_sims),
        'separation': sep_score,
        'pos_std': np.std(pos_sims),
        'neg_std': np.std(neg_sims)
    }


# ========== 使用示例 ==========
if __name__ == '__main__':
    print("可视化工具模块加载成功!")
    print("\n可用函数:")
    print("  - plot_tsne_embeddings(): t-SNE embedding可视化")
    print("  - plot_training_curves(): AKR训练曲线")
    print("  - plot_gkt_curves(): GKT训练曲线")
    print("  - plot_prediction_vs_ground_truth(): 预测vs真实值")
    print("  - plot_error_distribution(): 误差分布分析")
    print("  - plot_similarity_matrix(): 相似度矩阵热力图")
    print("  - analyze_positive_negative_pairs(): 正负对分析")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class SimpleSTEncoder(nn.Module):
    """简化的时空编码器 for Ft (目标域)"""

    def __init__(self, num_nodes, in_channels=1, hidden_dim=64, embed_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        # 时空卷积
        self.temporal_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.spatial_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.spatial_conv2 = GCNConv(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, edge_index, edge_weight=None):
        # x: [B, T, N, C] -> [B*N, C, T]
        B, T, N, C = x.shape
        x = x.transpose(1, 3).reshape(B * N, C, T)  # [B*N, C, T]
        x = F.relu(self.temporal_conv(x))  # [B*N, H, T]
        x = x.mean(dim=-1)  # 时序池化 [B*N, H]
        x = x.view(B, N, -1)  # [B, N, H]

        # 空间图卷积
        x = F.relu(self.spatial_conv1(x, edge_index, edge_weight))
        x = self.spatial_conv2(x, edge_index, edge_weight)
        x = self.norm(x)  # [B, N, D]
        return x


class DomainDiscriminator(nn.Module):
    """域判别器"""

    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.model(z)


class GKTGNN(nn.Module):
    """GKT阶段的图神经网络"""

    def __init__(self, embed_dim, hidden_dim=128, out_dim=12):  # 预测12步
        super().__init__()
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, target_mask):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        pred = self.regressor(x[target_mask])
        return pred


class SpatioTemporalSampler(nn.Module):
    """
    根据时空关系为对比学习构造正负样本对的采样器。
    假设：
    - 每个样本索引 i 对应一个 (node_id[i], time_id[i])。
    """

    def __init__(self,
                 node_ids,  # [N_total], 每个样本的路段编号
                 time_ids,  # [N_total], 每个样本的时间索引(整数)
                 adj_matrix,  # [N_node, N_node], 路网邻接(0/1或权重)
                 delta_t_pos=2,  # 时间相近正对阈值
                 delta_t_neg=12  # 时间相远负对阈值
                 ):
        """
        node_ids: LongTensor, shape [N_total]
        time_ids: LongTensor, shape [N_total]
        adj_matrix: Tensor, shape [N_node, N_node]
        """
        self.node_ids = node_ids
        self.time_ids = time_ids
        self.adj = adj_matrix
        self.delta_t_pos = delta_t_pos
        self.delta_t_neg = delta_t_neg

        # 预先把每个节点的邻居和非邻居存下来，加速采样
        self.num_nodes = adj_matrix.size(0)
        self.neighbors = []
        self.non_neighbors = []
        for i in range(self.num_nodes):
            neigh = torch.where(adj_matrix[i] > 0)[0]
            non_neigh = torch.where(adj_matrix[i] == 0)[0]
            self.neighbors.append(neigh)
            self.non_neighbors.append(non_neigh)

    def sample_pairs(self, batch_indices, num_pos=1, num_neg=1):
        """
        根据一批样本索引，构造正负样本对：
        返回：
        pos_pairs: List[(i, j)]  正对索引
        neg_pairs: List[(i, j)]  负对索引
        """
        device = batch_indices.device
        pos_pairs = []
        neg_pairs = []

        # 取出这批样本的路段和时间
        b_node = self.node_ids[batch_indices]  # [B]
        b_time = self.time_ids[batch_indices]  # [B]

        N_total = self.node_ids.size(0)

        for idx_in_batch, idx_global in enumerate(batch_indices):
            node_i = b_node[idx_in_batch].item()
            time_i = b_time[idx_in_batch].item()

            # -------- 正样本构造 --------
            # (1) 时间邻近：同一路段，时间差 <= delta_t_pos
            # 在全局中寻找满足条件的样本
            mask_same_node = (self.node_ids == node_i)
            mask_time_close = (torch.abs(self.time_ids - time_i) <= self.delta_t_pos)
            pos_candidates = torch.where(mask_same_node & mask_time_close)[0]

            # 排除自己
            pos_candidates = pos_candidates[pos_candidates != idx_global]

            if len(pos_candidates) > 0:
                # 随机挑 num_pos 个
                perm = torch.randperm(len(pos_candidates))[:num_pos]
                for k in perm:
                    j = pos_candidates[k].item()
                    pos_pairs.append((idx_global.item(), j))

            # (2) 空间邻居：空间邻居路段 & 同一时间
            neigh_nodes = self.neighbors[node_i]
            if len(neigh_nodes) > 0:
                # 候选样本：节点属于邻居集合 & 时间相同
                mask_neigh = torch.isin(self.node_ids, neigh_nodes.to(device))
                mask_same_time = (self.time_ids == time_i)
                pos_candidates2 = torch.where(mask_neigh & mask_same_time)[0]
                if len(pos_candidates2) > 0:
                    perm = torch.randperm(len(pos_candidates2))[:num_pos]
                    for k in perm:
                        j = pos_candidates2[k].item()
                        pos_pairs.append((idx_global.item(), j))

            # -------- 负样本构造 --------
            # (1) 时间相远：同一路段 & |Δt| >= delta_t_neg
            mask_time_far = (torch.abs(self.time_ids - time_i) >= self.delta_t_neg)
            neg_candidates = torch.where(mask_same_node & mask_time_far)[0]
            neg_candidates = neg_candidates[neg_candidates != idx_global]

            # (2) 空间远：非邻居 & 时间不太近
            far_nodes = self.non_neighbors[node_i]
            if len(far_nodes) > 0:
                mask_far_node = torch.isin(self.node_ids, far_nodes.to(device))
                # 时间约束：不与 time_i 太近
                mask_not_too_close = (torch.abs(self.time_ids - time_i) > self.delta_t_pos)
                neg_candidates2 = torch.where(mask_far_node & mask_not_too_close)[0]
                neg_candidates = torch.cat([neg_candidates, neg_candidates2])

            if len(neg_candidates) > 0:
                perm = torch.randperm(len(neg_candidates))[:num_neg]
                for k in perm:
                    j = neg_candidates[k].item()
                    neg_pairs.append((idx_global.item(), j))

        return pos_pairs, neg_pairs


class BridgedSTGNN(nn.Module):
    def __init__(self, source_model_path, n1, n2, embed_dim=128):
        super().__init__()
        # 源域预训练模型 (冻结)
        self.Fs = self._load_pretrained_stgnn(source_model_path, n1, embed_dim)
        for param in self.Fs.parameters():
            param.requires_grad = False

        # 目标域编码器 (训练)
        self.Ft = SimpleSTEncoder(n2, embed_dim=embed_dim)

        # AKR组件
        self.discriminator = DomainDiscriminator(embed_dim)
        self.pos_sampler = SpatioTemporalSampler()  # 正负样本采样器

        # GKT组件
        self.gkt_gnn = GKTGNN(embed_dim)

    def _load_pretrained_stgnn(self, path, n1, embed_dim):
        """加载并适配源域预训练ST-GNN作为Fs"""
        model = torch.load(path)  # 假设你有保存的模型
        model.eval()
        # 截取到embedding层，去掉最后的回归头
        # 这里需要根据你的具体ST-GNN结构调整
        return model

    def compute_nce_loss(self, z_s, z_t, temperature=0.1):
        """时空对比损失 InfoNCE"""
        # 简单实现：batch内对比
        z_all = torch.cat([z_s, z_t], dim=0)  # [2B, N, D]
        z_all = z_all.view(-1, z_all.size(-1))  # [2B*N, D]

        sim_matrix = torch.cosine_similarity(
            z_all.unsqueeze(1), z_all.unsqueeze(0), dim=-1
        ) / temperature

        # 假设正样本是同一batch内的对应位置
        labels = torch.arange(z_all.size(0)).to(z_all.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def forward_akr(self, source_batch, target_batch):
        """AKR阶段：对比学习 + 域对抗"""
        # 编码
        z_s = self.Fs(source_batch.x_s, source_batch.adj_s)  # [B, n1, D]
        z_t = self.Ft(target_batch.x_t, target_batch.adj_t)  # [B, n2, D]

        # 1. InfoNCE对比损失
        loss_nce = self.compute_nce_loss(z_s, z_t)

        # 2. 域对抗损失 (梯度反转)
        z_s_flat = z_s.reshape(-1, z_s.size(-1))
        z_t_flat = z_t.reshape(-1, z_t.size(-1))
        domain_s = self.discriminator(z_s_flat)
        domain_t = self.discriminator(z_t_flat)

        loss_adv_s = F.binary_cross_entropy_with_logits(domain_s, torch.zeros_like(domain_s))
        loss_adv_t = F.binary_cross_entropy_with_logits(domain_t, torch.ones_like(domain_t))
        loss_adv = (loss_adv_s + loss_adv_t) / 2

        total_loss = loss_nce + 0.1 * loss_adv
        return total_loss, z_s, z_t

    def build_bridged_graph(self, z_s, z_t, top_k=8):
        """构建Bridged-Graph"""
        # 计算相似度矩阵 [n1+n2, n1+n2]
        z_s_flat = z_s.reshape(-1, z_s.size(-1))
        z_t_flat = z_t.reshape(-1, z_t.size(-1))
        z_all = torch.cat([z_s_flat, z_t_flat], dim=0)

        # 余弦相似度
        sim_matrix = torch.cosine_similarity(
            z_all.unsqueeze(1), z_all.unsqueeze(0), dim=-1
        )

        # top-K邻居
        topk_indices = torch.topk(sim_matrix, top_k, dim=0).indices

        # 构造边
        src = torch.arange(z_all.size(0)).unsqueeze(1).expand(-1, top_k).flatten()
        dst = topk_indices.flatten()
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = sim_matrix[src, dst]

        # 节点特征
        x = z_all

        source_mask = torch.zeros(z_all.size(0), dtype=torch.bool)
        source_mask[:z_s_flat.size(0)] = True
        target_mask = ~source_mask

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    source_mask=source_mask, target_mask=target_mask)

    def forward_gkt(self, bridged_graph, target_flow_future):
        """GKT阶段：图回归"""
        h = self.gkt_gnn(bridged_graph.x, bridged_graph.edge_index,
                         bridged_graph.target_mask)
        loss = F.mse_loss(h, target_flow_future)
        return loss, h


# 使用示例
def train_akr(model, source_loader, target_loader, optimizer):
    model.train()
    total_loss = 0
    for source_batch, target_batch in zip(source_loader, target_loader):
        optimizer.zero_grad()
        loss, z_s, z_t = model.forward_akr(source_batch, target_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(source_loader)


def train_gkt(model, bridged_graph_loader, optimizer):
    model.train()
    total_loss = 0
    for data, target_flow in bridged_graph_loader:
        optimizer.zero_grad()
        loss, _ = model.forward_gkt(data, target_flow)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(bridged_graph_loader)


# 训练流程
model = BridgedSTGNN(source_model_path='pems07_stgnn.pth', n1=883, n2=170)
optimizer_akr = torch.optim.Adam(model.Ft.parameters(), lr=1e-3)

# 阶段1：AKR训练
for epoch in range(100):
    loss = train_akr(model, source_loader, target_loader, optimizer_akr)
    print(f"AKR Epoch {epoch}, Loss: {loss:.4f}")

# 阶段2：构建Bridged-Graph并训练GKT
optimizer_gkt = torch.optim.Adam(model.gkt_gnn.parameters(), lr=1e-3)
for epoch in range(50):
    loss = train_gkt(model, bridged_graph_loader, optimizer_gkt)
    print(f"GKT Epoch {epoch}, Loss: {loss:.4f}")

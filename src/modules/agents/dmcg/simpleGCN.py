import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class ActionEncoder(nn.Module):
    """动作表示编码器（简化版）"""

    def __init__(self, n_actions, repr_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_actions, 64),
            nn.ReLU(),
            nn.Linear(64, repr_dim))

    def forward(self, actions_onehot):
        return self.encoder(actions_onehot.float())  # 输入为one-hot格式


class NTXentLoss(nn.Module):
    """简化对比损失（适用于同批次样本对比）"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature
        self.cossim = nn.CosineSimilarity(dim=2)

    def forward(self, embeddings):
        # embeddings形状: [batch, repr_dim]
        sim_matrix = self.cossim(embeddings.unsqueeze(1), embeddings.unsqueeze(0)) / self.temp
        exp_sim = torch.exp(sim_matrix)

        # 对角线为正向样本
        diag_mask = torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
        positive = exp_sim[diag_mask].sum()
        negative = exp_sim[~diag_mask].sum()

        return -torch.log(positive / negative)


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super(SimpleGCN, self).__init__()

        # 定义GCN卷积层
        self.convs = nn.ModuleList()

        # 第一层GCNConv (输入 -> 隐藏)
        self.convs.append(GCNConv(in_channels, out_channels))

        # 后续隐藏层的GCNConv (隐藏 -> 隐藏)
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(out_channels, out_channels))

    def forward(self, x, adjacency_matrix):
        # 将稠密邻接矩阵转为稀疏格式 (edge_index, edge_weight)
        edge_index, edge_weight = dense_to_sparse(adjacency_matrix)

        # 多层GCN卷积
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)

        return x


class GCNFeatureAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GCNFeatureAgent, self).__init__()

        self.args = args
        self.input_shape = input_shape
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.N = args.n_agents
        self.device = th.device("cuda" if args.use_cuda else "cpu")
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)

        # 替换为标准的torch_geometric的GCN
        self.gnn = SimpleGCN(args.rnn_hidden_dim, args.rnn_hidden_dim, num_layers=2)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim, device=self.device)
        self.adjacency_matrix = self.get_adjacency_matrix(type=args.cg_edges)

        # 新增对比学习组件-----------------------------
        if self.args.ifActionConstrast:
            self.action_encoder = ActionEncoder(self.args.n_actions, self.args.action_repr_dim)
            self.contrastive_loss = NTXentLoss(temperature=0.1)

    def init_hidden(self):
        # 初始化hidden state
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))

        if self.args.need_graphEmb:
            # 使用GNN前向传播 (GCN)
            x = self.gnn(x, self.adjacency_matrix)

        # GRU更新hidden state
        if hidden_state != None:
            hidden_state = hidden_state.reshape(hidden_state.size(0) * self.args.n_agents, -1)
        h = self.rnn(x, hidden_state)

        # 对比学习分支-------------------------------
        # 假设inputs包含动作信息（需根据实际数据结构调整）
        if self.args.ifActionConstrast:
            actions_onehot = inputs[..., -self.action_encoder.encoder[0].in_features:]
            action_repr = self.action_encoder(actions_onehot)
            contrast_loss = self.contrastive_loss(action_repr)
            return contrast_loss, h
        else:
            return None, h

    def get_adjacency_matrix(self, type="allstar"):
        adj = torch.zeros((self.N, self.N), dtype=torch.float32, device=self.device)

        if type == "line":
            for i in range(self.N - 1):
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1  # undirected
        elif type == "full":
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:
                        adj[i, j] = 1
        elif type == "cycle":
            for i in range(self.N):
                adj[i, (i + 1) % self.N] = 1
                adj[(i + 1) % self.N, i] = 1
        elif type == "star":
            for i in range(1, self.N):
                adj[0, i] = 1
                adj[i, 0] = 1
        elif type == "allstar":
            for k in range(self.N):
                for i in range(self.N):
                    if i != k:
                        adj[k, i] = 1
                        adj[i, k] = 1

        # 添加自连接（如有需要）
        adj.fill_diagonal_(1.0)

        return adj
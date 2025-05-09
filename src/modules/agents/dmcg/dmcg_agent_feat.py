import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from TRAMA_release_pymarl.src.utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

from TRAMA_release_pymarl.src.modules.agents.dmcg.gcn_lib import GTN
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch
import time  # 确保顶部导入time


class GNN(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1024, num_nodes=None, cg_edges=None, device="cpu"):
        super().__init__()
        self.N = num_nodes
        self.device = device
        # self.feature_norm = nn.LayerNorm(in_channels)
        self.A = self.get_adjacency_matrix(type=cg_edges)
        self.gnn = GTN(
            num_edge=self.N,
            num_channels=2,
            w_in=in_channels,
            w_out=out_channels,
            num_nodes=self.N,
            num_layers=2
        )

    def get_edge_index(self, type="star"):  # need an initial graph construction
        if type == "line":
            edges = [[i, i + 1] for i in range(self.N - 1)]  # # arrange agents in a line
        elif type == "full":
            edges = [[(j, i + j + 1) for i in range(self.N - j - 1)] for j in range(self.N - 1)]
            edges = [e for l in edges for e in l]
        elif type == 'cycle':  # arrange agents in a circle
            edges = [(i, i + 1) for i in range(self.N - 1)] + [(self.N - 1, 0)]
        elif type == 'star':  # arrange all agents in a star around agent 0
            edges = [(0, i + 1) for i in range(self.N - 1)]
        edge_index = th.tensor(edges).T  # # arrange agents in a line
        return edge_index.to(self.device)

    def get_adjacency_matrix(self, type="allstar"):
        A = []
        if type == "lcs":
            for t in ["line", "cycle", "star"]:
                edges = self.get_edge_index(type=t)
                value_tmp = th.ones(edges.shape[1]).type(th.FloatTensor).to(self.device)
                A.append((edges, value_tmp))
        elif type == "allstar":
            all_edges = [[(k, i) for i in range(self.N) if i != k] for k in range(self.N)]
            for e in all_edges:
                edges = th.tensor(e).T
                value_tmp = th.ones(edges.shape[1]).type(th.FloatTensor).to(self.device)
                A.append((edges, value_tmp))
        elif type == "line" or type == "cycle" or type == "star" or type == "full":
            for t in [type] * self.N:
                edges = self.get_edge_index(type=t)
                value_tmp = th.ones(edges.shape[1]).type(th.FloatTensor).to(self.device)
                A.append((edges, value_tmp))
        return A

    def forward(self, x):
        # x = self.feature_norm(x)
        eval = not self.training
        if eval and self.cached_results:
            return self.cached_results
        else:
            x, H, Ws = self.gnn(self.A, x, num_nodes=self.N, eval=eval)
            self.cached_results = x
            return x


class DMCGFeatureAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DMCGFeatureAgent, self).__init__()
        self.args = args
        self.agent = args.agent

        self.cg_edges = args.cg_edges
        device = th.device("cuda" if args.use_cuda else "cpu")

        print(f"Using {self.agent} Agent")
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.gnn = GNN(input_shape, args.rnn_hidden_dim, num_nodes=args.n_agents, cg_edges=self.cg_edges, device=device)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # x = F.relu(self.fc1(inputs), inplace=True)
        start_time = time.time()

        x = self.gnn(inputs)

        gnn_time = time.time()

        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        h = self.rnn(x, hidden_state)
        rnn_time = time.time()

        print(
            f"[Agent Forward] GNN time: {gnn_time - start_time:.6f}s, RNN time: {rnn_time - gnn_time:.6f}s, Total: {rnn_time - start_time:.6f}s")

        return None, h




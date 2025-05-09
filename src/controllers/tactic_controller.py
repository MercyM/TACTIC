from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from components.goal_selectors import REGISTRY as goal_REGISTRY
import torch
import torch as th
from torch import nn
import torch.optim as optim
from TRAMA_release_pymarl.src.modules.agents.dmcg.simpleGCN import GCNFeatureAgent
from .basic_controller import BasicMAC
import contextlib
import itertools
import torch_scatter
from math import factorial
from random import randrange
import numpy as np


# This multi-agent controller shares parameters between agents
class TACTICMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_agents = args.n_agents
        # self.n_goals    = args.n_max_code
        self.latent_dim = args.latent_dim
        self.bs = args.batch_size
        self.args = args

        # ============== 1) 构建 LAGMA 需要的模块 ==============
        self.n_clusters = args.n_cluster
        self.n_goals = self.n_clusters
        self.goal_repr_dim = args.latent_dim
        self.selected_goal_index = None
        # self.true_goal = -1*th.ones(args.n_agents,dtype=th.int32)
        self.true_goal = None
        input_shape = self._get_input_shape(scheme)

        # ============== 2) 构建 DCG (payoffs, utilities) 相关模块 ==============
        self.n_actions = args.n_actions
        self.payoff_rank = args.cg_payoff_rank
        self.payoff_decomposition = isinstance(self.payoff_rank, int) and self.payoff_rank > 0
        self.iterations = args.msg_iterations
        self.normalized = args.msg_normalized
        self.anytime = args.msg_anytime
        # Create neural networks for utilities and payoff functions
        self.utility_fun = self._mlp(self.args.rnn_hidden_dim, args.cg_utilities_hidden_dim, self.n_actions)
        payoff_out = 2 * self.payoff_rank * self.n_actions if self.payoff_decomposition else self.n_actions ** 2
        self.payoff_fun = self._mlp(2 * self.args.rnn_hidden_dim, args.cg_payoffs_hidden_dim, payoff_out)
        # Create neural network for the duelling option
        self.duelling = args.duelling
        if self.duelling:
            self.state_value = self._mlp(int(np.prod(args.state_shape)), [args.mixing_embed_dim], 1)
        # Create the edge information of the CG
        self.edges_from = None
        self.edges_to = None
        self.edges_n_in = None
        self._set_edges(self._edge_list(args.cg_edges))
        self.agent_dcg = GCNFeatureAgent(input_shape, self.args)

        # default: one-hot vector
        self.goal_onehot = th.eye(self.n_clusters).to(device=args.device)  # [n_goal, n_goal]

        self._build_agents(input_shape)

        if self.args.goal_grad:
            for param in self.goal_predictor.parameters():
                param.requires_grad = False

        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.goal_selector = goal_REGISTRY[args.goal_selector](args)

        self.hidden_states = None
        self.goal_hidden_states = None
        self.hidden_states_dcg = None

        self.softmax = nn.Softmax(dim=1)  # Add softmax layer
        # for initialization, this will be replaced by VQVAE's embedding
        # self.goal_latent = nn.Parameter(th.rand(self.args.latent_dim, self.args.n_max_code )) # emb~U[0,1)

        # 新增模块-----------------------------------------
        self.quantizer = Quantizer(
            num_embeddings=args.n_cluster,
            embedding_dim=64,
            commitment_cost=args.commitment_cost
        ).to(device=args.device)

        # 任务注意力层（修正维度问题）
        self.attention_layer = nn.Sequential(
            nn.Linear(64 * 2, args.attn_dim),
            nn.ReLU(),
            nn.Linear(args.attn_dim, 1)
        ).to(device=args.device)
        # -----------------------------------------

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), goal_latent=None, test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        # .. action selection without goal
        # agent_outputs = self.forward(ep_batch, t_ep, goal_latent=goal_latent, test_mode=test_mode)
        # chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        # .. action selection with goal selection first
        agent_outputs, _, _, _ = self.forward(ep_batch, t_ep, goal_latent=None, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)

        return chosen_actions, self.selected_goal_index

    def _task_aware_attention(self, h_states):
        """使用self.edges_from和self.edges_to中的边生成注意力权重"""
        # h_states形状: [batch, agents, dim]
        edges_from = self.edges_from  # 直接使用已有的边定义
        edges_to = self.edges_to

        # 拼接节点特征
        h_src = h_states[:, edges_from, :]  # [batch, n_edges, dim]
        h_dst = h_states[:, edges_to, :]  # [batch, n_edges, dim]
        h_concat = torch.cat([h_src, h_dst], dim=-1)  # [batch, n_edges, dim*2]

        # 计算注意力分数
        attn_scores = self.attention_layer(h_concat).squeeze(-1)  # [batch, n_edges]
        return torch.sigmoid(attn_scores)

    def forward(self, ep_batch, t, goal_latent=None, test_mode=False, selected_goal=None,
                compute_grads=False):

        selected_goal_latent = None
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]  # only used for "pi_logits"

        # 新增协调图处理
        if self.args.ifActionConstrast:
            f_i, f_ij, contrast_loss = self.annotations(ep_batch, t, compute_grads)
        else:
            f_i, f_ij = self.annotations(ep_batch, t, compute_grads)

        # actions = self.greedy(f_i, f_ij, available_actions=ep_batch['avail_actions'][:, t])
        # dcg_outs = self.q_values(f_i, f_ij, actions)

        dcg_outs = self.q_values_per_agent_action(f_i, f_ij)

        # .. step 1: select goal based on local obsevation
        goal_probs_logit = None

        goal_probs_logit, self.goal_hidden_states = self.goal_predictor(agent_inputs, self.goal_hidden_states)


        agent_inputs = th.cat([agent_inputs, dcg_outs.reshape(ep_batch.batch_size * self.n_agents, -1)], dim=-1)

        # .. step 2: selected goal-onehot
        if selected_goal is None:
            if (t % self.args.n_pred_step) == 0:
                self.selected_goal_index, _ = self.goal_selector.select_goal(goal_probs_logit, test_mode=test_mode)
        else:
            self.selected_goal_index = selected_goal

        selected_goal_input = self.goal_onehot[self.selected_goal_index]  # tensor: [bs*n,n_cluster]

        # .. step 3: action selection conditioned on selected goal
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, selected_goal_input)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        if self.args.ifActionConstrast:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), \
                   (None if goal_probs_logit is None else goal_probs_logit.view(ep_batch.batch_size, self.n_agents,
                                                                                -1)), \
                   (None if selected_goal_latent is None else selected_goal_latent.view(ep_batch.batch_size,
                                                                                        self.n_agents,
                                                                                        -1)), contrast_loss
        else:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), \
                   (None if goal_probs_logit is None else goal_probs_logit.view(ep_batch.batch_size, self.n_agents,
                                                                                -1)), \
                   (None if selected_goal_latent is None else selected_goal_latent.view(ep_batch.batch_size,
                                                                                        self.n_agents,
                                                                                        -1))

    def update_goal_predictor(self, goal_prob_logits_out_reshaped, labels_reshaped, mask_reshaped):

        masked_prediction_loss = self.criterion(goal_prob_logits_out_reshaped[mask_reshaped],
                                                labels_reshaped[mask_reshaped])

        self.optimizer.zero_grad()
        masked_prediction_loss.backward()
        th.nn.utils.clip_grad_norm_(self.goal_predictor_params, 1.)
        self.optimizer.step

        return masked_prediction_loss.detach()  # for logging

    def target_update_goal_predictor(self):
        self.target_goal_predictor.load_state_dict(
            self.goal_predictor.state_dict())

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.goal_hidden_states = self.goal_predictor.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents,
                                                                                        -1)  # bav

        self.hidden_states_dcg = self.agent_dcg.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        # return self.agent.parameters()
        params = list(self.agent.parameters())
        if self.args.goal_grad:
            params += list(self.goal_predictor.parameters())  # note that selector does not have any parameters

        param = itertools.chain(self.agent_dcg.parameters(), self.utility_fun.parameters(),
                                self.payoff_fun.parameters())
        if self.duelling:
            param = itertools.chain(param, self.state_value.parameters())
        params = params + list(param)

        return params

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.goal_predictor.load_state_dict(other_mac.goal_predictor.state_dict())

        self.agent_dcg.load_state_dict(other_mac.agent_dcg.state_dict())

        self.utility_fun.load_state_dict(other_mac.utility_fun.state_dict())
        self.payoff_fun.load_state_dict(other_mac.payoff_fun.state_dict())
        if self.duelling:
            self.state_value.load_state_dict(other_mac.state_value.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.goal_predictor.cuda()
        self.agent_dcg.cuda()

        self.utility_fun.cuda()
        self.payoff_fun.cuda()
        if self.edges_from is not None:
            self.edges_from = self.edges_from.cuda()
            self.edges_to = self.edges_to.cuda()
            self.edges_n_in = self.edges_n_in.cuda()
        if self.duelling:
            self.state_value.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.goal_predictor.state_dict(), "{}/goal_predictor.th".format(path))

        th.save(self.agent_dcg.state_dict(), "{}/agent_dcg.th".format(path))

        th.save(self.utility_fun.state_dict(), "{}/utilities.th".format(path))
        th.save(self.payoff_fun.state_dict(), "{}/payoffs.th".format(path))
        if self.duelling:
            th.save(self.state_value, "{}/state_value.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.goal_predictor.load_state_dict(
            th.load("{}/goal_predictor.th".format(path), map_location=lambda storage, loc: storage))

        self.agent_dcg.load_state_dict(
            th.load("{}/agent_dcg.th".format(path), map_location=lambda storage, loc: storage))

        self.utility_fun.load_state_dict(
            th.load("{}/utilities.th".format(path), map_location=lambda storage, loc: storage))
        self.payoff_fun.load_state_dict(
            th.load("{}/payoffs.th".format(path), map_location=lambda storage, loc: storage))
        if self.duelling:
            self.payoff_fun.load_state_dict(
                th.load("{}/state_value.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.goal_predictor = agent_REGISTRY[self.args.gp_agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape


    # ================== DCG Core Methods =============================================================================

    def q_values_per_agent_action(self, f_i, f_ij):
        """
        Args:
            f_i  : shape [batch_size, n_agents, n_actions]
                   Utilities for each agent i, each action a_i.
            f_ij : shape [batch_size, n_edges, n_actions, n_actions]
                   Payoff for each edge e, where e = (edge_from[e], edge_to[e]).
                   Usually DCG has symmetrical edges or will store both directions.

        Returns:
            Q : shape [batch_size, n_agents, n_actions]
                For each (batch, i, a_i), 返回一个近似Q值。

        做法：
        1) 初始 Q = f_i, 代表单体 utilities。
        2) 对每条边 e=(i->j)， payoff[:, e] 是形状 [batch, n_actions, n_actions]。
           我们对 j 的 action 做个 max (或者 sum / expect 也可)，
           得到 "对 i 来说 a_i 的最优 payoff"。
        3) 把这个 payoff 加到 Q[:, i, a_i] 里去。
        4) 同理，对 j 也加 payoff 中 “对 j 的各个 a_j 来说 i 动作 max over i 的 payoff”。
        这样让每个 agent i 在自己动作 a_i 上，假设对方选最优相应动作 a_j。
        5) 最终得到 Q[batch, i, a_i]。
        """

        batch_size = f_i.size(0)
        n_agents = f_i.size(1)
        n_actions = f_i.size(2)

        # 先复制一份 f_i 作为我们的 Q
        Q = f_i.clone()  # shape: [batch, n_agents, n_actions]

        edges_from = self.edges_from  # shape [E]
        edges_to = self.edges_to  # shape [E]

        # 向量化处理所有边
        payoff_ij = f_ij  # [batch, E, a_i, a_j]

        # 对每条边计算i和j的max payoff
        max_payoff_i = payoff_ij.max(dim=-1)[0]  # [batch, E, a_i]
        max_payoff_j = payoff_ij.max(dim=-2)[0]  # [batch, E, a_j]

        # 使用scatter_add一次性累加
        Qpayoff_i = torch.zeros_like(Q).scatter_add(1, edges_from.view(1, -1, 1).expand(batch_size, -1, n_actions),
                                                    max_payoff_i)
        Qpayoff_j = torch.zeros_like(Q).scatter_add(1, edges_to.view(1, -1, 1).expand(batch_size, -1, n_actions),
                                                    max_payoff_j)

        Q += (Qpayoff_i + Qpayoff_j)


        return Q

    def annotations(self, ep_batch, t, compute_grads=False):
        """ Returns all outputs of the utility and payoff functions (Algorithm 1 in Boehmer et al., 2020). """
        with th.no_grad() if not compute_grads else contextlib.suppress():
            agent_inputs = self._build_inputs(ep_batch, t)

            if self.args.ifActionConstrast:
                contrast_loss, self.hidden_states_dcg = self.agent_dcg(agent_inputs, self.hidden_states_dcg)
                self.hidden_states_dcg = self.hidden_states_dcg.view(ep_batch.batch_size, self.n_agents, -1)
            else:
                self.hidden_states_dcg = self.agent_dcg(agent_inputs, self.hidden_states_dcg)[1].view(
                    ep_batch.batch_size,
                    self.n_agents, -1)

            f_i = self.utilities(self.hidden_states_dcg)
            if self.args.ifAttGraph:
                # VQ-VAE量化--------------------------------
                quantized, encoding_indices, vq_loss = self.quantizer(self.hidden_states_dcg)
                self.hidden_states_dcg = quantized.view(ep_batch.batch_size, self.n_agents, -1)

                # 任务感知注意力------------------------------
                attention = self._task_aware_attention(quantized)
                f_ij = self.payoffs(quantized, attention)
                # 将VQ损失附加到计算图中（梯度系数调整）
                if compute_grads:
                    f_ij = f_ij + vq_loss.unsqueeze(-1).unsqueeze(-1) * 0.01  # 梯度传播系数
            else:
                f_ij = self.payoffs(self.hidden_states_dcg)
        if self.args.ifActionConstrast:
            return f_i, f_ij, contrast_loss
        else:
            return f_i, f_ij

    def utilities(self, hidden_states):
        """ Computes the utilities for a given batch of hidden states. """
        return self.utility_fun(hidden_states)

    def payoffs(self, hidden_states, attention=None):
        """修正后的收益函数（保持维度兼容性）"""
        # 原始代码结构
        inputs = torch.cat([
            hidden_states[:, self.edges_from],
            hidden_states[:, self.edges_to]
        ], dim=-1)  # [batch, E, 2*hidden]

        # 应用注意力权重（维度广播修正）
        if attention != None:
            assert inputs.shape[1] == attention.shape[1], "边数不匹配"
            attn_mask = attention.unsqueeze(-1)  # [batch, E, 1]
            inputs = inputs * attn_mask

        output = self.payoff_fun(inputs)

        # 保持原始后续处理逻辑
        if self.payoff_decomposition:
            batch, E = output.shape[:2]
            output = output.view(batch, E, self.payoff_rank, 2, self.n_actions)
            L = output[..., 0, :]
            R = output[..., 1, :]
            P = torch.einsum('beri,berj->berij', L, R).sum(dim=2)
        else:
            batch, E = output.shape[:2]
            P = output.view(batch, E, self.n_actions, self.n_actions)

        # 保持对称性处理
        P_sym = 0.5 * (P + P.transpose(-1, -2))
        return P_sym


    def q_values(self, f_i, f_ij, actions):
        """ Computes the Q-values for given utilities, payoffs and actions (Algorithm 2 in Boehmer et al., 2020). """
        n_batches = actions.shape[0]
        # Use the utilities for the chosen actions
        values = f_i.gather(dim=-1, index=actions).squeeze(dim=-1).mean(dim=-1)
        # Use the payoffs for the chosen actions (if the CG contains edges)
        if len(self.edges_from) > 0:
            f_ij = f_ij.view(n_batches, len(self.edges_from), self.n_actions * self.n_actions)
            edge_actions = actions.gather(dim=-2, index=self.edges_from.view(1, -1, 1).expand(n_batches, -1, 1)) \
                           * self.n_actions + actions.gather(dim=-2,
                                                             index=self.edges_to.view(1, -1, 1).expand(n_batches, -1,
                                                                                                       1))
            values = values + f_ij.gather(dim=-1, index=edge_actions).squeeze(dim=-1).mean(dim=-1)
        # Return the Q-values for the given actions
        return values

    # ================== Override methods of BasicMAC to integrate DCG into PyMARL ====================================

    # ================== Private methods to help the constructor ======================================================

    @staticmethod
    def _mlp(input, hidden_dims, output):
        """ Creates an MLP with the specified input and output dimensions and (optional) hidden layers. """
        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        dim = input
        layers = []
        for d in hidden_dims:
            layers.append(nn.Linear(dim, d))
            layers.append(nn.ReLU())
            dim = d
        layers.append(nn.Linear(dim, output))
        return nn.Sequential(*layers)

    def _edge_list(self, arg):
        """ Specifies edges for various topologies. """
        edges = []
        wrong_arg = "Parameter cg_edges must be either a string:{'vdn','line','cycle','star','full'}, " \
                    "an int for the number of random edges (<= n_agents!), " \
                    "or a list of either int-tuple or list-with-two-int-each for direct specification."
        # Parameter cg_edges must be either a string:{'vdn','line','cycle','star','full'}, ...
        if isinstance(arg, str):
            if arg == 'line':  # arrange agents in a line
                edges = [(i, i + 1) for i in range(self.n_agents - 1)]
            elif arg == 'cycle':  # arrange agents in a circle
                edges = [(i, i + 1) for i in range(self.n_agents - 1)] + [(self.n_agents - 1, 0)]
            elif arg == 'star':  # arrange all agents in a star around agent 0
                edges = [(0, i + 1) for i in range(self.n_agents - 1)]
            elif arg == 'full':  # fully connected CG
                edges = [[(j, i + j + 1) for i in range(self.n_agents - j - 1)] for j in range(self.n_agents - 1)]
                edges = [e for l in edges for e in l]
            else:
                assert False, wrong_arg
        # ... an int for the number of random edges (<= (n_agents-1)!), ...
        if isinstance(arg, int):
            assert 0 <= arg <= factorial(self.n_agents - 1), wrong_arg
            for i in range(arg):
                found = False
                while not found:
                    e = (randrange(self.n_agents), randrange(self.n_agents))
                    if e[0] != e[1] and e not in edges and (e[1], e[0]) not in edges:
                        edges.append(e)
                        found = True
        # ... or a list of either int-tuple or list-with-two-int-each for direct specification.
        if isinstance(arg, list):
            assert all([(isinstance(l, list) or isinstance(l, tuple))
                        and (len(l) == 2 and all([isinstance(i, int) for i in l])) for l in arg]), wrong_arg
            edges = arg
        return edges

    def _set_edges(self, edge_list):
        """ Takes a list of tuples [0..n_agents)^2 and constructs the internal CG edge representation. """
        self.edges_from = th.zeros(len(edge_list), dtype=th.long)
        self.edges_to = th.zeros(len(edge_list), dtype=th.long)
        for i, edge in enumerate(edge_list):
            self.edges_from[i] = edge[0]
            self.edges_to[i] = edge[1]
        self.edges_n_in = torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                    index=self.edges_to, dim=0, dim_size=self.n_agents) \
                          + torch_scatter.scatter_add(src=self.edges_to.new_ones(len(self.edges_to)),
                                                      index=self.edges_from, dim=0, dim_size=self.n_agents)
        self.edges_n_in = self.edges_n_in.float()


import torch
import torch.nn.functional as F


class Quantizer(nn.Module):
    """VQ-VAE量化层（修正版）"""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # 码本初始化
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # 确保输入维度正确
        assert inputs.shape[
                   -1] == self.embedding_dim, f"Input dim {inputs.shape[-1]} != embedding dim {self.embedding_dim}"

        if self.embeddings.weight.device != inputs.device:
            self.embeddings = self.embeddings.to(inputs.device)

        # 转换输入维度 [batch*agents, dim] -> [batch, agents, dim]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # 计算与码本的距离（修正括号问题）
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # 获取最近邻编码
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(input_shape)

        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 直通估计器
        quantized = inputs + (quantized - inputs).detach()

        return quantized, encoding_indices, loss

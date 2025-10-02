import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init

from src.modules.sge.smacv2contrastivemodel import SMACV2GraphContrastiveModel
from src.utils.smac_utils import build_graphs_from_state_batch, state_features, build_partial_graphs_from_obs_batch


class GEMANRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GEMANRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape+1, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # self.apply(weights_init)

        self.sge_model = SMACV2GraphContrastiveModel(device=args.device, d_node_in=9, enemy_feature_idx=[]).to(
            args.device)
        self.sge_model.load_state_dict(torch.load("/home/aamato/Documents/marl/pymarl2/src/modules/sge/model_final.pth",
                                                  map_location=torch.device(args.device)))
        self.sge_model.eval()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        obs = inputs.view(-1, e)

        graphs = build_partial_graphs_from_obs_batch(obs, device="cuda")

        with torch.no_grad():
            embeddings, final_embeddings, current_state, goal_state = self.sge_model(graphs)
            similarity = torch.nn.functional.cosine_similarity(current_state, goal_state, dim=-1)
            similarity = (similarity + 1) / 2

        inputs = torch.cat([inputs.view(-1, e), similarity.view(-1, 1)], dim=-1)

        x = F.relu(self.fc1(inputs.view(-1, e+1)), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        return q.view(b, a, -1), h.view(b, a, -1)

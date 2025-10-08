from .gema_n_rnn_agent import GEMANRNNAgent
from .lagma_agent import LAGMAAgent
from .rnn_agent_mappo import RNNAgentMappo

REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_ppo_agent import RNNPPOAgent
from .conv_agent import ConvAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .mlp_agent import MLPAgent
from .atten_rnn_agent import ATTRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_mappo"] = RNNAgentMappo
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["gema_n_rnn"] = GEMANRNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["lagma"] = LAGMAAgent

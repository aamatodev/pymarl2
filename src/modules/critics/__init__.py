from src.modules.critics.centralv import CentralVCritic
from .centralV_rnn_critic import CentralVRNNCritic

critic_REGISTRY = {}


critic_REGISTRY["central_v"] = CentralVCritic
critic_REGISTRY["centralV_rnn_critic"] = CentralVRNNCritic
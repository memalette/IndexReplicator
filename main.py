import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from environments.index_environment import Env
from utils.utils import init_weights, get_device, hyperparam_search, Exp
from agents.base import Base
from agents.actor_critic import ActorCritic
from agents.ppo import PPO
from agents.particle_filter import ParticleFilter


# TODO: run best models and compare

device = get_device()

# Set seed
seed = 550
n_seeds = 10
torch.manual_seed(seed)
np.random.seed(seed)

# Set hyperparams

# PPO
hyperparams = {'lr_rate': 0.0005,
               'gamma': 0.99,
               'lmbda': 0.95,
               'eps_clip': 0.2,
               'K_epoch': 3,
               'T_horizon': 20,
               'n_episode': 500,
               'hidden_size': 256,
               'device': device
               }

# Actor-critic
GAMMA = 0.99
N_EPISODES = 1000
LR_POL = 0.0001
LR_VALF = 0.0001
EXP = 3
N_HIDDEN_POL = 400
N_HIDDEN_VALF = 400

actor_critic_agent = ActorCritic(N_EPISODES, GAMMA, LR_VALF, LR_POL, N_HIDDEN_VALF, N_HIDDEN_POL)

# Experiment 1
print('Experiment 1 :\n')
env1 = Env(data_path='returns.csv', context='test', experiment=1)
model_PPO = PPO(env1.n_states, env1.n_assets, hyperparams).float().to(device)

TE_PPO = []
TE_AC = []

for seed in range(n_seeds):
    TE_PPO.append(model_PPO.predict.predict(env1, pred_id='_ppo' + str(seed)))
    TE_AC.append(actor_critic_agent.predict(env1, pred_id=seed))

print('mean Tracking Error for PPO: ', np.array(TE_PPO).mean(), '\n')
print('mean Tracking Error for Actor-critic: ', np.array(TE_AC).mean(), '\n')


# Experiment 2
print('Experiment 2 :\n')
env2 = Env(data_path='returns.csv', context='test', experiment=2)
model_PPO = PPO(env2.n_states, env2.n_assets, hyperparams).float().to(device)

TE_PPO = []
TE_AC = []

for seed in range(n_seeds):
    TE_PPO.append(model_PPO.predict.predict(env2, pred_id='_ppo' + str(seed)))
    TE_AC.append(actor_critic_agent.predict(env2, pred_id=seed))

print('mean Tracking Error for PPO: ', np.array(TE_PPO).mean(), '\n')
print('mean Tracking Error for Actor-critic: ', np.array(TE_AC).mean(), '\n')

# Experiment 3
print('Experiment 3 :')
env3 = Env(data_path='returns.csv', context='test', experiment=3)
model_PPO = PPO(env3.n_states, env3.n_assets, hyperparams).float().to(device)

TE_PPO = []
TE_AC = []

for seed in range(n_seeds):
    TE_PPO.append(model_PPO.predict.predict(env3, pred_id='_ppo' + str(seed)))
    TE_AC.append(actor_critic_agent.predict(env3, pred_id=seed))

print('mean Tracking Error for PPO: ', np.array(TE_PPO).mean())
print('mean Tracking Error for Actor-critic: ', np.array(TE_AC).mean())



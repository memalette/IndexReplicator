import argparse
import os
import sys
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
from utils.utils import get_device, read_config_file
from utils.utils_graphs import plot_returns, plot_values
from agents.base import Base
from agents.particle_filter import ParticleFilter
from agents.actor_critic import ActorCritic
from agents.ppo import PPO
from agents.reinforce import reinforce_agent

parser = argparse.ArgumentParser(description='Parser to run best models')
parser.add_argument('--save_dir', type=str, default='./figs/',
                    help='location where figures will be saved')
parser.add_argument('--config_path', type=str, default='./config.json',
                    help='location where figures will be saved')
parser.add_argument('--experience', type=int, default=0,
                    help='experience of environment')
parser.add_argument('--n_tests', type=int, default=2,
                    help='number of tests to compute TEs')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

# Use the flags passed to the script to make the
# name for the experimental directory
print("\n########## Setting Up Experiment ##########")
flags = [flag.lstrip('--').replace('/', '').replace('\\', '') for flag in sys.argv[1:]]

experiment_path = os.path.join(args.save_dir+'_'.join([str(argsdict['experience']),
                                         str(argsdict['seed']), '/']

                                         + flags))
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

# Increment a counter so that previous results with the same args will not
# be overwritten. Comment out the next four lines if you only want to keep
# the most recent results.
i = 0
while os.path.exists(experiment_path + "_" + str(i)):
    i += 1
experiment_path = experiment_path + "_" + str(i)

# Set experiment
config = read_config_file(args.config_path)
env = Env(data_path='./data/returns.csv', context='test', experiment=args.experience)
device = get_device()

# set seed for entire run
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Particle filter
# TODO: fill this part

# A2C
params_ac = config["A2C"]
agent_ac = ActorCritic(params_ac["n_episodes"], params_ac["gamma"], params_ac["lr_valf"],
                       params_ac["lr_valf"], params_ac["n_hidden_valf"], params_ac["n_hidden_pol"])

# PPO
params_ppo = config["PPO"]
agent_ppo = PPO(env.n_states, env.n_assets, params_ppo["hyperparams"]).float().to(device)

# REINFORCE
params_re = config['RE']
agent_reinforce = reinforce_agent(params_re['hyperparams'] , env)

# main loop for figures
n_figs = 9

for fig in range(n_figs):

    # use same start
    start = int(np.random.uniform(0, env.history_len-env.T))

    # compute predictions
    _, returns_ac, values_ac = \
        agent_ac.predict(env, start, pred_id='_ac' + str(fig), model_path=params_ac["best_model_path"])
    _, returns_ppo, values_ppo = \
        agent_ppo.predict(env, start, pred_id='_ac' + str(fig), model_path=params_ppo["best_model_path"])
    _, returns_reinforce, values_reinforce = \
        agent_reinforce.predict(env, start, pred_id='_ac' + str(fig), model_path=params_re["best_model_path"])

    # plot graphs
    returns_path = experiment_path + '_returns_all_' + str(fig) + '.png'
    values_path = experiment_path + '_values_all_' + str(fig) + '.png'

    plot_returns(env, returns_ac, returns_ppo, returns_path)
    plot_values(env, values_ac, values_ppo, values_path)


# main loop to compute average TE's

TE_PF = []
TE_PPO = []
TE_AC = []
TE_RE = []

for i in range(args.n_tests):

    start = int(np.random.uniform(0, env.history_len-env.T))

    # compute predictions
    te_ac, _, _ = \
        agent_ac.predict(env, start, pred_id='_ac' + str(fig), model_path=params_ac["best_model_path"])
    te_ppo, _, _ = \
        agent_ppo.predict(env, start, pred_id='_ac' + str(fig), model_path=params_ppo["best_model_path"])
    te_reinforce, _, _ = \
        agent_reinforce.predict(env, start, pred_id='_ac' + str(fig), model_path=params_re["best_model_path"])

    # append values
    TE_AC.append(te_ac)
    TE_PPO.append(te_ppo)
    TE_RE.append(te_reinforce)

#print('mean Tracking Error for PF: ', round(np.array(TE_PF).mean()*100000, 4))
print('mean Tracking Error for A2C: ', round(np.array(TE_AC).mean()*100000, 4))
print('mean Tracking Error for PPO: ', round(np.array(TE_PPO).mean()*100000, 4))
print('mean Tracking Error for PPO: ', round(np.array(TE_RE).mean()*100000, 4))




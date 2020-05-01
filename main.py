import argparse
import os
import sys
import numpy as np
import torch

from environments.index_environment import Env
from utils.utils import get_device, read_config_file
from utils.utils_graphs import plot_returns, plot_values

from agents.particle_filter import ParticleFilter
from agents.actor_critic import ActorCritic
from agents.ppo import PPO
from agents.reinforce import reinforce_agent


parser = argparse.ArgumentParser(description='Parser to run best models')
parser.add_argument('--save_dir', type=str, default='./figs/',
                    help='location where figures will be saved')
parser.add_argument('--config_path', type=str, default='./config.json',
                    help='location where config file is')
parser.add_argument('--experience', type=int, default=0,
                    help='experience of environment')
parser.add_argument('--n_tests', type=int, default=100,
                    help='number of tests to compute TEs')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

# Use the flags passed to the script to make the
# name for the experimental directory
print("\n########## Setting Up Experiment{} ##########".format(args.experience))
flags = [flag.lstrip('--').replace('/', '').replace('\\', '') for flag in sys.argv[1:]]

experiment_path = os.path.join(args.save_dir+'_'.join([str(argsdict['experience']),
                                         str(argsdict['seed']), '/']))

print('Figures will be saved in: {}'.format(experiment_path))
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

# set seed for entire run and instanciate models
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Particle filter
params_pf = config["PF"]
particle_filter_agent = ParticleFilter(n_particles=params_pf["n_particles"], n_assets=env.n_assets,
                                       vol=params_pf["vol"], likeli_scale=params_pf["likeli_scale"])

# A2C
params_ac = config["A2C"]
agent_ac = ActorCritic(params_ac["n_episodes"], params_ac["gamma"], params_ac["lr_valf"],
                       params_ac["lr_pol"], params_ac["n_hidden_valf"], params_ac["n_hidden_pol"])

# PPO
params_ppo = config["PPO"]
agent_ppo = PPO(env.n_states, env.n_assets, params_ppo["hyperparams"]).float().to(device)

# REINFORCE
params_re = config['RE']
agent_reinforce = reinforce_agent(params_re['hyperparams'], env)


# main loop for figures
n_figs = 3

for start in range(n_figs):

    # use same start/period
    start = int(np.random.uniform(0, env.history_len-env.T))

    # compute predictions
    _, _, returns_pf, values_pf = particle_filter_agent.learn(env, start)
    _, returns_ac, values_ac = \
        agent_ac.predict(env, start, pred_id='_ac' + str(start), model_path=params_ac["best_model_path"])
    _, returns_ppo, values_ppo = \
        agent_ppo.predict(env, start, pred_id='_ppo' + str(start), model_path=params_ppo["best_model_path"])
    _, returns_re, values_re = \
        agent_reinforce.predict(env, start, pred_id='_re' + str(start), model_path=params_re["best_model_path"])

    # plot graphs
    returns_path = experiment_path + '_returns_all_' + str(start) + '.png'
    values_path = experiment_path + '_values_all_' + str(start) + '.png'

    plot_returns(env, returns_pf, returns_ac, returns_ppo, returns_re, returns_path)
    plot_values(env, values_pf, values_ac, values_ppo, values_re, values_path)

print('Done plotting figures!')

# main loop to compute average TE's

TE_PF = []
TE_PPO = []
TE_AC = []
TE_RE = []

for i in range(args.n_tests):

    start = int(np.random.uniform(0, env.history_len-env.T))

    # compute predictions
    te_pf, _, _, _ = particle_filter_agent.learn(env, start)
    te_ac, _, _ = \
        agent_ac.predict(env, start, pred_id='_ac' + str(start), model_path=params_ac["best_model_path"])
    te_ppo, _, _ = \
        agent_ppo.predict(env, start, pred_id='_ac' + str(start), model_path=params_ppo["best_model_path"])
    te_re, _, _ = \
        agent_reinforce.predict(env, start, pred_id='_ac' + str(start), model_path=params_re["best_model_path"])

    # append values
    TE_PF.append(te_pf)
    TE_AC.append(te_ac)
    TE_PPO.append(te_ppo)
    TE_RE.append(te_re)


print('mean Tracking Error for PF: ', round(np.array(TE_PF).mean()*100000, 4))
print('mean Tracking Error for A2C: ', round(np.array(TE_AC).mean()*100000, 4))
print('mean Tracking Error for PPO: ', round(np.array(TE_PPO).mean()*100000, 4))
print('mean Tracking Error for REINFORCE: ', round(np.array(TE_RE).mean()*100000, 4))


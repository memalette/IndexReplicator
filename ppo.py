import numpy as np
#from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from sim_environment import SimEnv
from utils.utils import init_weights
from utils.utils import get_device


class PPO(nn.Module):
    def __init__(self, num_assets, hyperparams, std=0):
        super(PPO, self).__init__()
        self.data = []
        self.hyperparams = hyperparams
        self.device = hyperparams['device']

        self.fc1 = nn.Linear(1, 256)
        self.fc_pi = nn.Linear(256, num_assets)
        self.fc_v = nn.Linear(256, 1)

        self.apply(init_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=self.hyperparams['lr_rate'])
        self.log_std = nn.Parameter(torch.ones(num_assets) * std)

    def pi(self, x):
        x = x.view(-1, 1)
        mu = F.relu(self.fc1(x))
        mu = self.fc_pi(mu)

        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        return dist

    def v(self, x):
        x = x.view(-1, 1)
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def append_data(self, transition):
        self.data.append(transition)

    def select_action(self, dist):
        constraint_respected = False

        while not constraint_respected:
            a = dist.sample().squeeze(0)
            action = (a / a.sum())

            if (action.abs()).max() < 2:
                constraint_respected = True

        return action, a

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst = [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, prob_a = transition

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])

        s, a, r, s_prime, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.stack(a_lst), \
                                   torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                   torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, prob_a

    def update(self):
        s, a, r, s_prime, log_prob = self.make_batch()

        for i in range(self.hyperparams['K_epoch']):
            td_target = r.to(self.device) + self.hyperparams['gamma'] * self.v(s_prime.float().to(self.device))
            v_s = self.v(s.float().to(self.device))
            delta = td_target - v_s
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0

            for delta_t in delta[::-1]:
                advantage = self.hyperparams['gamma'] * self.hyperparams['lmbda'] * advantage + delta_t[0]
                advantage_lst.append([advantage])

            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            new_dist = self.pi(s.float().to(self.device))
            entropy = new_dist.entropy().mean()
            _, a = self.select_action(new_dist)
            log_prob_new = new_dist.log_prob(a).sum()
            ratio = torch.exp(log_prob_new - log_prob.to(self.device))  # a/b == exp(log(a)-log(b))

            surr1 = (ratio * advantage)
            surr2 = (torch.clamp(ratio, 1 - self.hyperparams['eps_clip'], 1 + self.hyperparams['eps_clip']) * advantage).to(self.device)

            # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s.float() , td_target.detach().float())
            loss = -torch.min(surr1, surr2) + 0.5 * (v_s.float() - td_target.detach().float()).pow(2).mean() - 0.001 * entropy
            loss = loss.mean().to(self.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == '__main__':

    def main(seed, N_ASSETS, S0, MUS, SIGMAS, T, EPS_PARAM, params_PPO):

        torch.manual_seed(seed)
        np.random.seed(seed)

        device = params_PPO['device']

        env = SimEnv(N_ASSETS, S0, MUS, SIGMAS, T, EPS_PARAM)
        model = PPO(N_ASSETS, params_PPO).float().to(device)

        replicator_returns = []
        index_returns = []

        for episode in range(5):

            # print('episode: ', n_epi)
            s = env.reset()
            for day in range(T):  # 252 days in an episode

                # print('day :', day)

                for t in range(params_PPO['T_horizon']):
                    dist = model.pi(torch.from_numpy(s).float().to(device))

                    action, a = model.select_action(dist)

                    s_prime, r, done = env.step(action.detach().cpu().numpy())

                    model.append_data((torch.from_numpy(s), a,
                                       torch.from_numpy(r),
                                       torch.from_numpy(s_prime),
                                       dist.log_prob(a).sum()))
                    s = s_prime

                model.update()

                replicator_returns.append(env.portfolio)
                index_returns.append(env.index)

        return replicator_returns, index_returns

    device = get_device()

    # Environment params
    N_ASSETS = 3
    EPS_PARAM = (0, 0.01)
    T = 252  # trading days in a year (this could be the end of an episode)
    S0 = np.array([10, 15, 20])
    MUS = np.array([-0.01, 0.02, 0.05])
    SIGMAS = np.array([0.01, 0.05, 0.10])

    # Model hyperparams
    hyperparams = {'lr_rate': 0.0005,
                   'gamma': 0.98,
                   'lmbda': 0.95,
                   'eps_clip': 0.2,
                   'K_epoch': 3,
                   'T_horizon': 20,
                   'device': device}

    n_seeds = 5
    rep_runs = []
    index_runs = []

    for experiment in range(n_seeds):
        rep, index = main(experiment, N_ASSETS, S0, MUS, SIGMAS, T, EPS_PARAM, hyperparams)
        rep_runs.append(np.cumprod(np.array(rep) + 1))
        index_runs.append(np.cumprod(np.array(index) + 1))

    plt.plot(np.array(rep_runs).mean(axis=0), label='Clone')
    plt.plot(np.array(index_runs).mean(axis=0), label='Index')
    plt.title('Cumulative Return with multiple seeds')
    plt.legend()
    plt.show()




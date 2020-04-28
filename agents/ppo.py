import numpy as np
import pandas as pd
#from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from environments.index_environment import Env
from utils.utils import init_weights
from utils.utils import get_device


class PPO(nn.Module):
    def __init__(self, num_states, num_assets, hyperparams, std=0):
        super(PPO, self).__init__()
        self.data = []
        self.hyperparams = hyperparams
        self.device = hyperparams['device']

        self.fc1 = nn.Linear(num_states, 256)
        self.fc_pi = nn.Linear(256, num_assets)
        self.fc_v = nn.Linear(256, 1)

        self.apply(init_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=self.hyperparams['lr_rate'])
        self.log_std = nn.Parameter(torch.ones(num_assets) * std)

    def pi(self, x):

        mu = F.relu(self.fc1(x))
        mu = self.fc_pi(mu)

        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        return dist

    def v(self, x):

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
        s_lst, a_lst, r_lst, next_s_lst, log_prob_lst, done_lst = \
            [], [], [], [], [], []

        for transition in self.data:
            s, a, r, next_s, log_prob, done = transition

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            next_s_lst.append(next_s)
            log_prob_lst.append([log_prob])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s = torch.stack(s_lst)
        a = torch.stack(a_lst)
        r = torch.tensor(r_lst)
        next_s = torch.stack(next_s_lst)
        log_prob = torch.tensor(log_prob_lst)
        done = torch.tensor(done_lst)

        self.data = []
        return s, a, r, next_s, log_prob, done

    def update(self):
        s, a, r, next_s, log_prob, done_mask = self.make_batch()

        for i in range(self.hyperparams['K_epoch']):
            td_target = r.to(self.device) + self.hyperparams['gamma'] * self.v(next_s.float().to(self.device)) * done_mask
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
            ratio = torch.exp(log_prob_new - log_prob.to(self.device))

            surr1 = (ratio * advantage)
            surr2 = (torch.clamp(ratio, 1 - self.hyperparams['eps_clip'], 1 + self.hyperparams['eps_clip']) * advantage).to(self.device)

            # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s.float() , td_target.detach().float())
            loss = -torch.min(surr1, surr2) + 0.5 * (v_s.float() - td_target.detach().float()).pow(2).mean() - 0.001 * entropy
            loss = loss.mean().to(self.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return -loss

    def learn(self, env):

        best_loss = 10000000

        losses = []
        replicator_returns = []
        index_returns = []

        action = 0

        for episode in range(self.hyperparams['n_episode']):

            print('episode: ', episode)

            state = env.reset()

            done = False
            loss_ep = []
            while not done:
                for t in range(self.hyperparams['T_horizon']):
                    dist = self.pi(torch.from_numpy(state).float().to(device))

                    next_action, a = self.select_action(dist)
                    delta = next_action - action

                    next_state, reward, done = env.step(next_action.detach().cpu().numpy(), delta.cpu().numpy())

                    self.append_data((torch.from_numpy(state).float(), a,
                                       torch.from_numpy(reward),
                                       torch.from_numpy(next_state),
                                       dist.log_prob(a).sum(),
                                       done))

                    state = next_state
                    action = next_action

                    if done:
                        break

                loss_ep.append(self.update().detach().cpu().numpy())

            # append last values of episode
            losses.append(np.array(loss_ep).mean())
            replicator_returns.append(env.portfolio)
            index_returns.append(env.index)

        if np.array(losses).mean() < best_loss:
            best_loss = np.array(losses).mean()
            torch.save(self.state_dict(), '../models/ppo/best_ppo.pt')

        return losses, replicator_returns, index_returns

    def predict(self, env, pred_id=None):

        # reset env
        state = env.reset()

        # instantiate variables
        action_logs = []
        portfolio_returns = []
        action = 0
        done = False
        T = 0

        # load best model
        self.load_state_dict(torch.load('../models/ppo/best_ppo.pt'))

        while not done:
            dist = self.pi(torch.from_numpy(state).float().to(device))

            next_action, _ = self.select_action(dist)
            delta = next_action - action

            action_logs.append(next_action.detach().cpu().numpy())

            # Take the action and observe reward and next state
            next_state, reward, done = env.step(next_action.detach().cpu().numpy(), delta.cpu().numpy())

            portfolio = np.dot(next_action.detach().cpu().numpy(), env.assets.reshape((env.n_assets, 1)))
            portfolio_returns.append(portfolio)

            # update the total length for this episode
            T += 1

            state = next_state
            action = next_action

        action_logs = pd.DataFrame(np.vstack(action_logs)[:, :2], index=env.dates)

        plt.plot(action_logs)
        plt.savefig('../figs/action_logs' + str(pred_id) + '.png')
        plt.close()

        returns_logs = pd.DataFrame({'index': np.array(env.index_returns).flatten(),
                                     'replicator': np.array(portfolio_returns).flatten()},
                                    index=env.dates)
        returns_logs.plot(marker='.')
        plt.savefig('../figs/returns' + str(pred_id) + '.png')
        plt.close()

        # unit values
        index_returns = np.array(env.index_returns).flatten()
        portfolio_returns = np.array(portfolio_returns).flatten()
        index_value = np.cumprod(1 + index_returns)
        portfolio_value = np.cumprod(1 + portfolio_returns)

        values_logs = pd.DataFrame({'index': index_value,
                                    'replicator': portfolio_value},
                                   index=env.dates)

        values_logs.plot(marker='.')
        plt.legend()
        plt.savefig('../figs/values' + str(pred_id) + '.png')
        plt.close()

        tracking_errors = (returns_logs['index'] - returns_logs['replicator']) ** 2

        return tracking_errors.mean()


if __name__ == '__main__':

    device = get_device()

    # Set seed
    seed = 550
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model hyperparams
    hyperparams = {'lr_rate': 0.0005,
                   'gamma': 0.99,
                   'lmbda': 0.95,
                   'eps_clip': 0.2,
                   'K_epoch': 3,
                   'T_horizon': 20,
                   'n_episode': 200,
                   'device': device}

    ##### TRAINING ####
    #env = Env(data_path='../returns.csv', context='train')
    #model = PPO(env.n_states, env.n_assets, hyperparams).float().to(device)
    #losses, rep_returns, index_returns = model.learn(env)

    # Plot cumulative return
    #f1 = plt.figure()
    #ax1 = f1.add_subplot(111)
    #ax1.plot(np.cumprod(np.array(rep_returns) + 1), label='Clone')
    #ax1.plot(np.cumprod(np.array(index_returns) + 1), label='Index')
    #ax1.title.set_text('Cumulative Return')
    #ax1.legend()
    #f1.savefig('cum_returns.pdf')

    # Plot loss evolution
    #f2 = plt.figure()
    #ax2 = f2.add_subplot(111)
    #ax2.plot(np.array(losses), label='Loss')
    #ax2.title.set_text('Loss over episodes')
    #f2.savefig('loss.pdf')

    #print('Done training!')

    ##### PREDICT #####

    ## compute predictions for multiple seeds

    env = Env(data_path='../returns.csv', context='test')
    model = PPO(env.n_states, env.n_assets, hyperparams).float().to(device)

    for experiment in range(10):

        print('SEED: ', experiment)

        torch.manual_seed(experiment)
        np.random.seed(experiment)

        tracking_error = model.predict(env, pred_id='_ppo' + str(experiment))

    print('Done predicting!')




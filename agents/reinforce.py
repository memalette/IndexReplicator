import sys
sys.path.append('../')

from environments.index_environment import *
from agents.base import Base
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable


import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm.auto import tqdm


class Exp(nn.Module):

    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        out = torch.exp(x)
        out = torch.clamp(out, min=0.0001)
        return out

class PolicyNN(nn.Module):
    def __init__(self,n_inputs, n_outputs, n_hidden = 128):
        super(PolicyNN,self).__init__()

        # Model definition
        # first layer
        self.l1 = nn.Sequential(
            nn.Linear(n_inputs,n_hidden),
            #nn.Dropout(0.6),
            nn.ReLU(),

        )

        self.alpha = nn.Sequential(
            nn.Linear(n_hidden, n_outputs),
            Exp()
        )

    def forward(self,state):
        l1_output = self.l1(state)
        return self.alpha(l1_output) 

		
class reinforce_agent():
	def __init__(self, hyperparams , environment ):
  
		self.env = environment
		self.state_dim = self.env.n_states
		self.n_actions = self.env.n_assets
		self.hyperparams = hyperparams


	def select_action(self , state):
    
		state = torch.tensor(state, dtype=torch.float)
		alpha = self.Policy(state)
		dirichlet_dist = torch.distributions.dirichlet.Dirichlet(alpha)

		action = dirichlet_dist.sample()

		log_prob = dirichlet_dist.log_prob(action)
    #print('in policy estimator, action log_prob', action, log_prob)
		return action, log_prob
  
  
	def train(self):
		self.Policy = PolicyNN(self.state_dim , self.n_actions , self.hyperparams['hidden_layer_neurons'])
		self.reinforce()
  
    
	def reinforce(self):
    # this is the function which trains the agent using the reinforce algorithm
    # insted of making updates at each timestep the returns are stored and used for update at the end of episode
    # this stabilizes training and helps with convergence

		num_episodes = self.hyperparams['n_episodes']
		results = np.zeros(num_episodes)
		optimizer = optim.Adam(self.Policy.parameters() , lr = self.hyperparams['learning_rate'])
		gamma = self.hyperparams['gamma'] 
		Average_reward = 0
		for index_episode in range(num_episodes):
      # at each episode the rewards and log_probs of actions are stored in a list to be used at the end
      # of episode for training
			k = 0
			rewards = []
			log_probs = []
			state = self.env.reset()
			running_reward = 0
			prev_action = 0
			while True:
				action,log_prob = self.select_action(state)
				delta = action - prev_action
				log_probs.append(log_prob)
				state , reward , done  = self.env.step(action.numpy(), delta.numpy())
				rewards.append(reward)
				running_reward += (gamma ** k ) * reward
        #print(reward)
				k += 1
				if done:
					break
      #print(k)
      #print(rewards)
      #print(log_probs)
      #print('running reward' , running_reward)
			results[index_episode] = running_reward
			Average_reward += running_reward
			if index_episode % 10 == 0:
				Average_reward /= (10 * k)
				print('Episode number :', index_episode , 'Average reward in the last 10 episodes:' , Average_reward)
				Average_reward = 0
			ret = 0
			returns = [i for i in range(k)]
      # returns for each timestep are computed 
			for j in range(k-1,-1,-1):
				ret = ret * gamma + rewards[j][0]
				returns[j] = ret * (gamma ** j)
      #print(returns)
			loss = []
      #print(returns)
      #print(log_probs)
			for j in range(k):
				loss.append( -returns[j] * log_probs[j].unsqueeze(0))
			optimizer.zero_grad()
      #print(loss)
			loss = torch.cat(loss).sum()
			loss.backward()
			optimizer.step()
    
    #print(results)
    #return results
		torch.save(self.Policy.state_dict(), '../models/reinforce/reinforce.pt')
		print('done')


	def predict(self, env , start = None , save = False , model_path='../models/reinforce/',pred_id=None):
    
		if start is None:
			state = env.reset()
		else:
			state = env.reset(start)
    
		action_logs = []
		portfolio_returns = []
		prev_action = 0
		terminal = False
		T = 0
		#print(model_path + 'reinforce.pt')
		self.Policy = PolicyNN(env.n_states , env.n_assets , self.hyperparams['hidden_layer_neurons'])
		self.Policy.load_state_dict(torch.load(model_path + 'reinforce.pt'))

		while not terminal:
			action,_ = self.select_action(state)
			delta = action - prev_action
			action_logs.append(action.numpy())

			next_state, reward, terminal = env.step(action.numpy(), delta.numpy())
			prev_action = action

			portfolio = np.dot(action.numpy(), env.assets.reshape((env.n_assets,1)))
			portfolio_returns.append(portfolio)

			T += 1

			state = next_state
    
	
		action_logs = pd.DataFrame(np.vstack(action_logs)[:,:2], index=env.dates)

		plt.plot(action_logs)
		if save:
			plt.savefig('../figs/reinforce_action_logs'+str(pred_id)+'.png')
		plt.close()



		returns_logs = pd.DataFrame({'index': np.array(env.index_returns).flatten(), 
                                     'replicator': np.array(portfolio_returns).flatten()},
                                      index=env.dates)
		returns_logs.plot(marker='.')
		if save:
			plt.savefig('../figs/reinforce_returns'+str(pred_id)+'.png')
		plt.close()
		index_returns = np.array(env.index_returns).flatten()
		portfolio_returns = np.array(portfolio_returns).flatten()
		index_value = np.cumprod(1 + index_returns)
		portfolio_value = np.cumprod(1 + portfolio_returns)


		values_logs = pd.DataFrame({'index': index_value, 
                                     'replicator': portfolio_value},
                                      index=env.dates)
		values_logs.plot(marker='.')
		plt.legend()
		if save:
			plt.savefig('../figs/reinforce_values'+str(pred_id)+'.png')
		plt.close()

		tracking_errors = (returns_logs['index'] - returns_logs['replicator'])**2
    
		return tracking_errors.mean(), portfolio_returns, portfolio_value
	
	
if __name__ == '__main__':

	hyperparams = {
    'n_episodes' : 100 ,
    'gamma' : 0.99,
    'learning_rate' : 1e-3 ,
    'hidden_layer_neurons' : 200,
    'Exp_num' : 0
	}
	
    # train
	env = Env(context='train', experiment=hyperparams['Exp_num'])
	agent = reinforce_agent(hyperparams , env)
	agent.train()

    # test
	env = Env(context='test', experiment=hyperparams['Exp_num'])
	
	
	TE = []
	for i in range(10):
		te, _, _ = agent.predict(env, pred_id=i , save= True)
		TE.append(te)

	TE = np.array(TE).mean()
	print('AVERAGE TE: '+str(TE))

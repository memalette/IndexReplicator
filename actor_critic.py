# Necassary import statments
from index_environment import *
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from tqdm.auto import tqdm
import glob



class ValueEstimator(nn.Module):

    def __init__(self,n_inputs, n_hidden = 128,lr = 0.001):
        
        super(ValueEstimator,self).__init__()
        
        # Model definition
        self.model = nn.Sequential(
            nn.Linear(n_inputs,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,1)
        )
       
        # Model optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr)

        # Loss criterion
        self.criterion = torch.nn.MSELoss()


    def predict(self,state):
        """
        Compute the probability for each action corresponding to state s
        """
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            return self.model(state)

    def update(self,states,returns):
        """
        Update the weights of the network based on 
            states     : input states for the value estimator
            returns    : actual monte-carlo return for the states
        """
        pred_returns = self.model(states)
        loss = self.criterion(pred_returns,Variable(returns))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



class PolicyNN(nn.Module):
    def __init__(self,n_inputs, n_outputs, n_hidden = 128):
        super(PolicyNN,self).__init__()

        # Model definition
        # first layer
        self.l1 = nn.Sequential(
            nn.Linear(n_inputs,n_hidden),
            #nn.Dropout(0.6),
            nn.ReLU(),

            nn.Linear(n_hidden,n_hidden),
            #nn.Dropout(0.6),
            nn.ReLU()
        )

        self.alpha = nn.Sequential(
            nn.Linear(n_hidden, n_outputs),
            nn.ReLU()
        )

    def forward(self,state):
        """
        Compute the probability for each action corresponding to state s
        """
        l1_output = self.l1(state)
        return self.alpha(l1_output)



class PolicyEstimator(nn.Module):
    def __init__(self, policy_nn,lr = 0.001):
        super(PolicyEstimator,self).__init__()

        self.policy_nn = policy_nn
       
        # Model optimizer
        self.optimizer = torch.optim.Adam(self.policy_nn.parameters(),lr)


    def predict(self,state):

        """
        Compute the probability for each action corresponding to state s
        """
        state = torch.tensor(state, dtype=torch.float)
        
        return self.policy_nn(state)

    def update(self,advantages,log_probs):
        """
        Update the weights of the network based on 
            advantages : advantage for each step in the episode
            log_probs  : log of probability of each action
        """

        policy_gradients = []

        for log_prob,adv in zip(log_probs,advantages):
            policy_gradients.append(-log_prob * adv)

        # Computing gradient ascent using negative loss 
        loss = torch.stack(policy_gradients).sum()

        # Use backprop to train
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

    def sample_action(self, dirichlet_dist):
        return dirichlet_dist.sample()


    def select_action(self,state):
        alpha = self.predict(state)

        dirichlet_dist = torch.distributions.dirichlet.Dirichlet(alpha)

        action = self.sample_action(dirichlet_dist)
        
        log_prob = dirichlet_dist.log_prob(action)

        return action, log_prob



class ActorCritic:
    def __init__(self,n_episodes, gamma):

        self.n_episodes = n_episodes
        self.gamma = gamma

    def learn(self, env):
        # stats for the episode
        episodic_return = []
        episodic_length = []

        # Define value function approximator
        valf_est = ValueEstimator(env.n_states) 

        # Define policy estimator
        policy_nn = PolicyNN(n_inputs=env.n_states, n_outputs=env.n_assets)
        policy_est = PolicyEstimator(policy_nn) 

        for eps in tqdm(range(self.n_episodes)):
          
            eps_buffer = []
            log_probs = []
            cum_reward = 0
            T = 0 # episode length

            state = env.reset()
            action_logs = []
            portfolio_returns = []
            prev_action = 0

            i = 0
            terminal = False
            while not terminal:
            
                # Choose action based on the state and current parameter to generate a full episode
                action, log_prob = policy_est.select_action(state)
                # print(action)
                action_logs.append(action.numpy())
                log_probs.append(log_prob)

                # print(cost)

                # Take the action and observe reward and next state
                next_state, reward, terminal = env.step(action.numpy())
                next_state = None if terminal else next_state
                prev_action = action

                # update the total reward for this episode
                cum_reward += reward

                # update the total length for this episode
                T += 1
                portfolio = np.dot(action.numpy(), env.assets.reshape((env.n_assets,1)))
                # print(portfolio - env.index)
                portfolio_returns.append(portfolio)
                action_logs.append(action.numpy())

                # storing the episode in buffer 
                eps_buffer.append(Transition(state, action, reward, next_state))

                i+=1

                if not terminal:
                    state = next_state
                  
                else:

                    # Storing the episode length and total reward 
                    episodic_length.append(T)
                    episodic_return.append(cum_reward)

                    # Extract the states and rewards from the stored episodic history
                    states,_,rewards,next_states = zip(*eps_buffer) 

                    # Calculate the returns for each state using bootstrapped values
                    Gt = []
                  
                    for reward,next_state in zip(rewards,next_states):
                        if next_state is not None:
                            R = torch.tensor(reward) + self.gamma * valf_est.predict(next_state)
                    else:
                        R = torch.tensor(reward)

                    Gt.append(R)

                    # Moving the data to correct device 
                    returns = torch.tensor(Gt, dtype=torch.float).reshape(-1,1)
                    states  = torch.tensor(states, dtype=torch.float)  

                    #Calculate the baseline
                    baseline_values = valf_est.predict(states)

                    # Calculate advanatge using baseline  
                    advantages = returns - baseline_values

                    # Update the function approximators
                    valf_est.update(states,returns)   
                    policy_est.update(advantages,log_probs)

        action_logs = np.vstack(action_logs)

        return action_logs, env.index_returns, portfolio_returns





if __name__ == '__main__':


    GAMMA = 0.9
    N_EPISODES = 1000

    # instantiate index environment
    env = Env()
    Transition = namedtuple('Transition',('state','action','reward','next_state'))

    actor_critic_agent = ActorCritic(N_EPISODES, GAMMA)
    action_logs, index_returns, portfolio_returns = actor_critic_agent.learn(env)


    plt.close()
    plt.plot(np.array(index_returns).flatten())
    plt.plot(np.array(portfolio_returns).flatten())
    plt.show()


    # unit values
    index_returns = np.array(index_returns).flatten()
    portfolio_returns = np.array(portfolio_returns).flatten()
    index_value = np.cumprod(1 + index_returns)
    portfolio_value = np.cumprod(1 + portfolio_returns)


    plt.close()
    plt.plot(index_value, label='index')
    plt.plot(portfolio_value, label='portfolio')
    plt.legend()
    plt.show()
# Necassary import statments
from environments.index_environment import *
from agents.base import Base
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm.auto import tqdm


class ValueEstimator(nn.Module):

    def __init__(self,n_inputs, n_hidden = 128,lr = 0.00001):
        
        super(ValueEstimator,self).__init__()
        
        # Model definition
        self.model = nn.Sequential(
            nn.Linear(n_inputs,n_hidden),
            nn.ReLU(),
            # nn.Linear(n_hidden,n_hidden),
            # nn.ReLU(),
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
        # print('loss value estimator: '+str(loss))

        return float(loss.detach().numpy())
        

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



class PolicyEstimator(nn.Module):
    def __init__(self, policy_nn,lr = 0.0001):
        super(PolicyEstimator,self).__init__()

        self.policy_nn = policy_nn
       
        # Model optimizer
        self.optimizer = torch.optim.Adam(self.policy_nn.parameters(),lr)


    def predict(self,state):

        state = torch.tensor(state, dtype=torch.float)
        
        return self.policy_nn(state)

    def update(self,advantages,log_probs):
        """
        Update the weights of the network based on 
            advantages : advantage for each step in the episode
            log_probs  : log of probability 
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
        # print('loss policy estimator: '+ str(loss))

        return float(loss.detach().numpy())

    def sample_action(self, dirichlet_dist):
        return dirichlet_dist.sample()


    def select_action(self,state):

        alpha = self.predict(state)
        # print(alpha)

        dirichlet_dist = torch.distributions.dirichlet.Dirichlet(alpha)

        action = self.sample_action(dirichlet_dist)

        log_prob = dirichlet_dist.log_prob(action)
        # print(log_prob)

        return action, log_prob


class ActorCritic(Base):
    def __init__(self,n_episodes, gamma, lr_valf, lr_pol, n_hidden_valf, n_hidden_pol):

        self.n_episodes = n_episodes
        self.gamma = gamma
        self.lr_valf = lr_valf
        self.lr_pol = lr_pol
        self.n_hidden_valf = n_hidden_valf
        self.n_hidden_pol = n_hidden_pol

    def learn(self, env):
        # stats for the episode
        episodic_return = []
        episodic_length = []

        # Define value function approximator
        valf_est = ValueEstimator(n_inputs=env.n_states, lr=self.lr_valf, n_hidden=self.n_hidden_valf) 

        # Define policy estimator
        policy_nn = PolicyNN(n_inputs=env.n_states, n_outputs=env.n_assets, n_hidden=self.n_hidden_pol)
        policy_est = PolicyEstimator(policy_nn, lr=self.lr_pol) 
        loss_valf = []
        loss_pol = []
        best_loss_valf = 10000000
        best_loss_pol = 10000000

        for eps in tqdm(range(self.n_episodes)):
          
            eps_buffer = []
            log_probs = []
            cum_reward = 0
            T = 0 # episode length

            state = env.reset()
            action_logs = []
            portfolio_returns = []
            prev_action = 0


            # i = 0
            terminal = False
            while not terminal:
            
                # Choose action based on the state and current parameter to generate a full episode
                action, log_prob = policy_est.select_action(state)
                delta = action - prev_action
                # print(action)
                action_logs.append(action.numpy())
                log_probs.append(log_prob)

                # print(cost)

                # Take the action and observe reward and next state
                next_state, reward, terminal = env.step(action.numpy(), delta.numpy())
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

                # i+=1

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
                    loss_valf.append(valf_est.update(states,returns))
                    loss_pol.append(policy_est.update(advantages,log_probs))

                    if len(loss_valf) > 20: 
                        current_mean_valf = np.array(loss_valf).mean()
                        current_mean_pol = np.array(loss_pol).mean()
                        print('rolling mean value loss: '+str(current_mean_valf/10))
                        print('rolling mean policy loss: '+str(current_mean_pol/1000))
                        loss_valf.pop(0)
                        loss_pol.pop(0)

                        if current_mean_valf < best_loss_valf:
                            torch.save(valf_est.state_dict(), 'models/best_valf_est.pt')
                            best_loss_valf = current_mean_valf

                        if current_mean_pol < best_loss_pol: 
                            last_best = eps
                            torch.save(policy_est.state_dict(), 'models/best_pol_est.pt')
                            best_loss_pol = current_mean_pol

                        if last_best - eps > 20:
                            break



    def predict(self, env, pred_id=None):

        # reset env
        state = env.reset()

        # instantiate variables
        action_logs = []
        portfolio_returns = []
        prev_action = 0
        terminal = False
        T = 0

        # load best models 

        # Define value function approximator
        valf_est = ValueEstimator(env.n_states) 
        valf_est.load_state_dict(torch.load('models/best_valf_est.pt'))

        # Define policy estimator
        policy_nn = PolicyNN(n_inputs=env.n_states, n_outputs=env.n_assets)
        policy_est = PolicyEstimator(policy_nn) 
        policy_est.load_state_dict(torch.load('models/best_pol_est.pt'))


        while not terminal:
        
            action,_ = policy_est.select_action(state)
            delta = action - prev_action

            action_logs.append(action.numpy())

            # Take the action and observe reward and next state
            next_state, reward, terminal = env.step(action.numpy(), delta.numpy())
            next_state = None if terminal else next_state
            prev_action = action

            portfolio = np.dot(action.numpy(), env.assets.reshape((env.n_assets,1)))
            portfolio_returns.append(portfolio)

            # update the total length for this episode
            T += 1

            state = next_state
              

        action_logs = pd.DataFrame(np.vstack(action_logs)[:,:2], index=env.dates)

        plt.plot(action_logs)
        plt.savefig('figs/action_logs'+str(pred_id)+'.png')
        plt.close()


        returns_logs = pd.DataFrame({'index': np.array(env.index_returns).flatten(), 
                                     'replicator': np.array(portfolio_returns).flatten()},
                                      index=env.dates)
        returns_logs.plot(marker='.')
        plt.savefig('figs/returns'+str(pred_id)+'.png')
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
        plt.savefig('figs/values'+str(pred_id)+'.png')
        plt.close()


        tracking_errors = (returns_logs['index'] - returns_logs['replicator'])**2

        return tracking_errors.mean()



if __name__ == '__main__':


    GAMMA = 0.99
    N_EPISODES = 1000
    LR_POL = 0.0001
    LR_VALF = 0.0001
    EXP = 3
    N_HIDDEN_POL = 400
    N_HIDDEN_VALF = 400
    
    Transition = namedtuple('Transition',('state','action','reward','next_state'))

    # train
    env = Env(context='train', experiment=EXP)
    actor_critic_agent = ActorCritic(N_EPISODES, GAMMA, LR_VALF, LR_POL, N_HIDDEN_VALF, N_HIDDEN_POL)
    actor_critic_agent.learn(env)


    # test
    env = Env(context='test', experiment=EXP)
    actor_critic_agent = ActorCritic(N_EPISODES, GAMMA, LR_VALF, LR_POL, N_HIDDEN_VALF, N_HIDDEN_POL)

    TE = []
    for i in range(10):
        TE.append(actor_critic_agent.predict(env, pred_id=i))

    TE = np.array(TE).mean()
    print('AVERAGE TE: '+str(TE))



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Env:
	def __init__(self, context='train'):

		self.returns = pd.read_csv('returns.csv', index_col=0)
		half = int(self.returns.shape[0]/2)

		if context == 'train':
			self.returns.iloc[:half,:]
		elif context == 'test':
			self.returns.iloc[half:,:]

		self.index_col = self.returns.columns[0]
		self.assets_col = self.returns.columns[1:]
		self.n_assets = len(self.assets_col)

		self.t = 0
		self.T = 100
		self.history_len = self.returns.shape[0]  


		self.reset()


	def reset(self):
		# reset values
		self.start = int(np.random.uniform(0, self.history_len-self.T))
		self.t = 0
		self.state = np.array([0]*(self.n_assets+2))
		self.n_states = self.state.shape[0]
		self.prev_action = 0
		self.IV = 1
		self.PV = 1
		self.dates = self.returns.index[self.start:(self.start+self.T)]

		# logging lists
		self.portfolio_returns = []
		self.index_returns = []
		
		return self.state

	def step_index(self):
		index = self.returns[self.index_col].values[self.t+self.start]
		self.IV = self.IV * (1+index)
		return index

	def step_assets(self):
		return self.returns[self.assets_col].values[self.t+self.start,:]


	def get_reward(self, state):
		total_delta = np.sum(np.abs(state[2:]))
		reward = -np.abs(self.state[0]) -np.abs(self.state[1])*5 #- total_delta*0.02
		return reward.flatten()


	def step(self, action=None, delta=None):
		# step index - get index return
		self.index = self.step_index()
		self.index_returns.append(float(self.index))

		# step assets - get assets returns
		self.assets = self.step_assets()
		self.t += 1

		# if for particle filter
		if (action is not None):
			# get portfolio return
			self.portfolio = np.dot(action, self.assets.reshape((self.n_assets,1)))
			self.PV = self.PV *(1+self.portfolio)

			# state 0 - difference between index and portfolio
			difference_indicator1 = float(self.index - self.portfolio)
			difference_indicator2 = float(self.IV - self.PV)

			# store in new state
			self.state = np.concatenate((np.array([difference_indicator1, difference_indicator2]), delta))

			# get reward
			reward = self.get_reward(self.state)
			# print(reward)

			# store values for next iteration
			self.prev_action = action

			rule2 = (self.t == self.T) 

			if rule2:
				done = True
			else:
				done = False

			return self.state.flatten(), reward, done

		else: 

			done = True if (self.t == self.T) else False
			return done
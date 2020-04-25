import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Env:
	def __init__(self):

		self.reset()

		self.returns = pd.read_csv('returns.csv', index_col=0)
		self.index_col = self.returns.columns[0]
		self.assets_col = self.returns.columns[1:]
		self.n_assets = len(self.assets_col)

		self.t = 0
		self.T = 500 #self.returns.shape[0]  # 500

		self.reset()

	def reset(self):
		# reset values
		self.t = 0
		self.state = np.array([0])
		self.n_states = self.state.shape[0]
		self.prev_action = 0

		# logging lists
		self.portfolio_returns = []
		self.index_returns = []
		
		return self.state

	def step_index(self):
		return self.returns[self.index_col].values[self.t]

	def step_assets(self):
		return self.returns[self.assets_col].values[self.t,:]


	def get_reward(self, state):
		reward = -np.abs(self.state[0])
		return reward.flatten()


	def step(self, action=None):
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

			# state 0 - difference between index and portfolio
			difference_indicator = float(self.index - self.portfolio)

			# store in new state
			self.state = np.array([difference_indicator])

			# get reward
			reward = self.get_reward(self.state)

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Env:
	def __init__(self, max_diff=None, leverage_cost=None, trans_cost=None, 
					max_leverage=None, cutoff_leverage=None):
		self.max_diff = max_diff
		self.max_leverage = max_leverage
		self.leverage_cost = leverage_cost
		self.trans_cost = trans_cost
		self.cutoff_leverage = cutoff_leverage

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
		self.state = np.array([0, 0, 0])
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
		leverage_cost = -self.leverage_cost if self.state[2] > self.cutoff_leverage else 0
		leverage_cost += -0.05 if self.state[2] > self.max_leverage else 0
		reward = -(self.state[0])**2 - self.state[1]*self.trans_cost - leverage_cost
		return reward.flatten()


	def step(self, action=None, delta=None):
		# step index - get index return
		self.index = self.step_index()
		self.index_returns.append(float(self.index))

		# step assets - get assets returns
		self.assets = self.step_assets()
		self.t += 1

		# if for particle filter
		if (action is not None) and (delta is not None):
			# get portfolio return
			self.portfolio = np.dot(action, self.assets.reshape((self.n_assets,1)))

			# state 0 - difference between index and portfolio
			difference_indicator = float(self.index - self.portfolio)

			# state 1 - transaction cost indicator
			transaction_indicator = float(np.sum(np.abs(delta)))

			# state 2 - gross leverage indicator 
			leverage_indicator = float(np.sum(np.abs(action)))

			# store in new state
			self.state = np.array([difference_indicator, transaction_indicator, leverage_indicator])

			# get reward
			reward = self.get_reward(self.state)

			# store values for next iteration
			self.portfolio_returns.append(float(self.portfolio))
			self.prev_action = action

			# stopping rule
			rule1 = (np.abs(self.state[0]) > self.max_diff)
			rule2 = (self.t == self.T) 
			rule3 = (self.state[2] > self.max_leverage)

			if rule1 or rule2 or rule3:
				done = True
			else:
				done = False

			return self.state.flatten(), reward, done

		else: 

			done = True if (self.t == self.T) else False
			return done
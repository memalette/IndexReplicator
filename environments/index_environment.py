import numpy as np
import pandas as pd


class Env:
	def __init__(self, data_path='../data/returns.csv', context='train', experiment=0):

		self.returns = pd.read_csv(data_path, index_col=0)
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
		self.experiment = experiment

		self.reset()

	def reset(self, start=None):
		# reset values
		if start is None:
			self.start = int(np.random.uniform(0, self.history_len-self.T))
		else:
			self.start = start

		self.t = 0
		
		self.prev_action = 0
		self.IV = 1
		self.PV = 1
		self.AV = np.array([1]*self.n_assets)
		self.dates = self.returns.index[self.start:(self.start+self.T)]

		# logging lists
		self.portfolio_returns = []
		self.index_returns = []

		if self.experiment == 0:
			self.state = np.array([0]*(self.n_assets+1))
		elif (self.experiment == 1):
			self.state = np.array([0]*(self.n_assets+2))
		elif (self.experiment == 2):
			self.state = np.array([0]*(3*self.n_assets+2))
		elif (self.experiment == 3):
			self.state = np.array([0]*(3*self.n_assets+2))

		self.n_states = self.state.shape[0]

		
		return self.state

	def step_index(self):
		index = self.returns[self.index_col].values[self.t+self.start]
		self.IV = self.IV * (1+index)
		return index

	def step_assets(self):
		assets = self.returns[self.assets_col].values[self.t+self.start,:]
		self.AV = self.AV * (1+assets)
		return assets


	def get_reward(self, state):
	
		base_penalty = -np.abs(self.state[0])

		if self.experiment == 0:
			reward = base_penalty
		elif (self.experiment == 1) or (self.experiment == 2):
			cumul_penalty = -np.abs(self.state[1])*5
			reward = base_penalty + cumul_penalty
		elif (self.experiment == 3): 
			cumul_penalty = -np.abs(self.state[1])*5
			turnover_penalty = -np.sum(np.abs(state[2:17]))*0.7
			reward = base_penalty + cumul_penalty + turnover_penalty
		else: 
			print('Specify experiment 0-4')

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
			cumulative_returns = self.AV - 1
			past_returns = self.assets

			# store in new state
			if self.experiment == 0:
				self.state = np.concatenate((np.array([difference_indicator1]), delta))
			elif (self.experiment == 1):
				self.state = np.concatenate((np.array([difference_indicator1, difference_indicator2]), delta))
			elif (self.experiment == 2):
				self.state = np.concatenate((np.array([difference_indicator1, difference_indicator2]), delta, 
										cumulative_returns, past_returns))
			elif (self.experiment == 3):
				self.state = np.concatenate((np.array([difference_indicator1, difference_indicator2]), delta, 
										cumulative_returns, past_returns))

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
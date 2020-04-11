import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Env:
	def __init__(self, n_assets, s0, mus, sigmas, T, epsilon_param):
		self.n_assets = n_assets
		self.S = s0
		self.mus = mus
		self.sigmas = sigmas
		self.T = T
		self.epsilon_param = epsilon_param

		self.reset()


	def step_betas(self):
		constraint_respected = False
		while not constraint_respected:
			epsilon = np.random.normal(self.epsilon_param[0], self.epsilon_param[1], size=self.n_assets)
			next_betas = self.betas + epsilon
			next_betas = next_betas/np.sum(next_betas)

			if max(np.abs(next_betas)) < 2: 
				constraint_respected = True

		self.betas = next_betas


	def init_betas(self):
		constraint_respected = False
		while not constraint_respected:
			betas = np.random.uniform(-1, 1, size=self.n_assets)
			betas = betas /np.sum(betas)

			if max(np.abs(betas)) < 2:
				constraint_respected = True

		self.betas = betas


	def step_index(self):
		self.step_betas()
		self.step_assets()

		return np.dot(self.assets.T, self.betas)


	def step_assets(self):
		dT = 1
		Z = np.random.normal(0,1, size = self.n_assets)
		prev_S = self.S
		self.S = prev_S + (self.mus-0.5*self.sigmas**2)*dT + self.sigmas*np.sqrt(dT)*Z
		self.assets = (self.S-prev_S)/prev_S


	def reset(self):
		self.init_betas()
		self.state = np.random.normal(0, 0.2, size=1)
		return self.state

	def get_reward(self):
		return (self.state)**2

	def step(self, action):
		index = self.step_index()
		self.portfolio = np.dot(self.assets.T, action)
		self.state = index - self.portfolio

		reward = self.get_reward()
		done = False # continuous task

		return self.state, reward, done



if __name__ == '__main__':

    N_ASSETS = 3 
    EPSILON_PARAM = (0, 0.01)
    T = 252 # trading days in a year (this could be the end of an episode)
    S0 = np.array([10, 15, 20])
    MUS = np.array([-0.01, 0.02, 0.05])
    SIGMAS = np.array([0.01, 0.05, 0.10])

    env = Env(N_ASSETS, S0, MUS, SIGMAS, T, EPSILON_PARAM)


    # example of a simulation for a portfolio that stays constant
    action = np.array([1./3., 1./3., 1./3.])
    portfolio_returns = []
    for step in range(T):
        env.step(action)
        portfolio_returns.append(env.portfolio)


    # transform returns into cumulative values
    portfolio_value = np.cumprod(np.array(portfolio_returns)+1)


    plt.plot(portfolio_value)
    plt.show()



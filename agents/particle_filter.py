from scipy.stats import *
from numpy.random import random
from environments.index_environment import *

class ParticleFilter:
	def __init__(self, n_particles, n_assets, vol, likeli_scale):
		self.n_particles = n_particles
		self.n_assets = n_assets
		self.vol = vol
		self.likeli_scale = likeli_scale

		self.q = np.zeros((self.n_assets, self.n_assets))
		np.fill_diagonal(self.q, self.vol)

	def initialize_action(self):
		action = np.random.uniform(0, 1, (self.n_particles, self.n_assets))
		sums = np.sum(action, axis=1).reshape(action.shape[0],1)
		div =  np.tile(sums, self.n_assets) 
		action = action/div

		return action


	def multinomial_resample(self, weights):
		cumul_sum = np.cumsum(weights)
		cumul_sum[-1] = 1.
		return np.searchsorted(cumul_sum, random(len(weights)))


	def learn(self, env):

		state = env.reset()
		action_logs = []
		portfolio_returns = []

		action = self.initialize_action()
		terminal = False
		i=0
		while not terminal:
			epsilon = np.random.multivariate_normal(mean=np.array([0.0]*self.n_assets), 
													cov=self.q, 
													size=self.n_particles
												)
			action_pres = action + epsilon
			sums = np.sum(action_pres, axis=1).reshape(action_pres.shape[0],1)
			div =  np.tile(sums, self.n_assets)  
			action_pres = action_pres/div
			

			# get state
			terminal = env.step()
			states = np.sum(action_pres * env.assets, axis=1)


			likelihood = norm.pdf(x=states-env.index, loc=0, scale=self.likeli_scale)

			idx = self.multinomial_resample(likelihood/np.sum(likelihood))
			action_post = action_pres[idx]

			action = action_post.mean(0)
			portfolio = np.dot(action.T, env.assets.reshape((self.n_assets,1)))
			print(portfolio - env.index)
			portfolio_returns.append(portfolio)
			action_logs.append(action)

			i+=1

		return action_logs, env.index_returns, portfolio_returns



if __name__ == '__main__':

	N_PARTICLES = 1000000 # the more the merrier, but it gets slow
	VOL = 0.1 # standard deviation for the epsilon 
	LIKELI_SCALE = 0.005 # the smaller, the better it will track the index

	# instantiate index environment
	env = Env(context='test')

	particle_filter_agent = ParticleFilter(
								n_particles=N_PARTICLES, 
								n_assets=env.n_assets, 
								vol=VOL, 
								likeli_scale = LIKELI_SCALE
								)

	action_logs, index_returns, portfolio_returns = particle_filter_agent.learn(env)

	action_logs = np.vstack(action_logs)
	gross_levergage = np.abs(action_logs).sum(1)

	plt.plot(gross_levergage)
	plt.title('Replication Portfolio gross leverage')
	plt.show()

	plt.close()
	plt.plot(np.array(index_returns).flatten())
	plt.plot(np.array(portfolio_returns).flatten())
	plt.title('Index vs Replication Portfolio returns')
	plt.show()


	# unit values
	index_returns = np.array(index_returns).flatten()
	portfolio_returns = np.array(portfolio_returns).flatten()
	index_value = np.cumprod(1 + index_returns)
	portfolio_value = np.cumprod(1 + portfolio_returns)


	plt.close()
	plt.plot(index_value, label='index')
	plt.plot(portfolio_value, label='portfolio')
	plt.title('Index vs Replication Portfolio values')
	plt.legend()
	plt.show()

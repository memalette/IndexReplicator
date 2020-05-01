import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_returns(env, a, b, c, d, save_path):

    returns_logs = pd.DataFrame({'index': np.array(env.index_returns).flatten(),
                                 'PF': np.array(a).flatten(),
                                 'A2C': np.array(b).flatten(),
                                 'PPO': np.array(c).flatten(),
                                 'RE': np.array(d).flatten()},
                                 index=env.dates)

    returns_logs.plot(marker='.')
    plt.title('Portfolio Returns')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_values(env, a, b, c, d, save_path):

    index_returns = np.array(env.index_returns).flatten()
    index_value = np.cumprod(1 + index_returns)

    returns_logs = pd.DataFrame({'index': index_value,
                                 'PF': np.array(a).flatten(),
                                 'A2C': np.array(b).flatten(),
                                 'PPO': np.array(c).flatten(),
                                 'RE': np.array(d).flatten()},
                                 index=env.dates)

    returns_logs.plot(marker='.')
    plt.title('Portfolio Values')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


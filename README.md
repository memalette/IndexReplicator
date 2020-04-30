# Reinforcement Learning Approaches to Index Replication

In this project, you will find various implementations of reinforcement learning methods in order to perform index 
replication. The index used for replication is the S&P 500 extracted from yahoo finance. We compare these methods to 
particle filters similar to what is proposed by [Roncalli and Weisang 2008](http://www.thierry-roncalli.com/download/particle-filter.pdf?fbclid=IwAR1Fp6eq2rfYsAcNx8RQtdifWx1ykko8TtQqD38H9_3AK386WH26zLGyD7E). Algorithms implemented thus include: 

## Algorithms
* [Particle filter](https://github.com/memalette/IndexReplicator/blob/master/agents/particle_filter.py)
* [A2C](https://github.com/memalette/IndexReplicator/blob/master/agents/actor_critic.py)
* [PPO](https://github.com/memalette/IndexReplicator/blob/master/agents/ppo.py)
* [REINFORCE]() TODO: add link

## Dependencies
1. Python 3
2. PyTorch
3. Hyperopt 

command to install Hyperopt: 
```
pip install hyperopt
``` 

## Arguments in parser of main.py

For a specific experience, main.py runs predictions of the best models and produces
graphs comparing returns and values. It also computes the mean tracking error, used as a 
 performance metric, of each agent with `n_tests` predictions.

<pre>
--save_dir SAVE_DIR   path where figures of the experiment will be saved
--config_path CONFIG_PATH 
                      path where the configuration file is. The config
                      file contains the best hyperparameters of each model
                      for this specific experimence
--experience EXPERIENCE
                      type of experience used in the index environment
--n_tests N_TESTS     number of test predictions done to compute the mean tracking
                      error of each model
--seed SEED           random seed to reproduce the same experiments
</pre>

## Example of a command to run main.py

* Running main.py for the third experiment:

```
python main.py --save_dir="/location/where/figures/are/saved" --experience=3 --n_tests=100
```


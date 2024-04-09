import training_functions as tf
from game import Game
from agent import Agent

version = 'version_1'
# env, agent, eps = tf.load_state(version)

env = Game(4, reward_mode='log2', negative_reward = -2, tile_move_penalty = 0.1)
eps = 0.5
# Create the agent, duplicating default values for visibility
agent = Agent(state_size=env.state_size * 18, action_size=env.action_size,
              seed=42, fc1_units=1024, fc2_units=1024, fc3_units = 1024,
              buffer_size = 100000, batch_size = 1024, lr = 0.004, 
              use_expected_rewards=True, predict_steps = 2, gamma = 0., 
              tau = 0.001)

tf.dqn(agent=agent, 
       version=version, 
       env=env,
       n_episodes=200000,
       eps_start=eps or 0.05,
       eps_end=0.00001,
       eps_decay=0.999,
       step_penalty=0,
       sample_mode='random',
       start_learn_iterations=10,
       bootstrap_iterations=0,
       bootstrap_every=50)
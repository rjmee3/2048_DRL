from DQNAgent import DQNAgent
from env import Env
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import logging
from keras.optimizers import Adam

'''----------------------------HYPERPARAMETERS----------------------------'''

BOARD_SIZE         = 4                                              # length of one size of the board
BATCH_SIZE         = 256                                            # num of examples used in one iteration
STATE_SHAPE        = (BOARD_SIZE, BOARD_SIZE, 1)                    # size of a one-hot encoded board
ACTION_SIZE        = 4                                              # will always be 4 (up, down, left, right)
LEARNING_RATE      = 0.001                                          # learning rate for minimizing loss function
GAMMA              = 0.95                                           # discount factor to determine importance of future reward
MEMORY_SIZE        = 10000                                          # how many experiences are stored in the replay buffer
TARGET_UPDATE_FREQ = 5                                              # how many episodes it takes until the target network is updated
MAX_EPISODES       = 10000                                          # maximum number of episodes to train on
EPSILON            = 1.0                                            # initial value for epsilon in epsilon-greedy strategy
EPSILON_DECAY      = 0.99                                           # the rate at which epsilon decays
EPSILON_MIN        = 0.00                                           # the lowest epsilon is allowed to go
MODEL_WEIGHTS_PATH = 'model_weights.pth'                            # file path for saved model weights

'''-----------------------------------------------------------------------'''
if __name__ == '__main__':
    # suppress excessive logging from tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel(logging.ERROR)
    
    
    agent = DQNAgent(STATE_SHAPE, ACTION_SIZE, MEMORY_SIZE, GAMMA, 
                    EPSILON, EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE, 
                    BATCH_SIZE, Adam(lr=LEARNING_RATE), 'mean_squared_error')
    env = Env(BOARD_SIZE)
    agent.model.summary()

    # establish matplotlib for interactive mode
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(12, 9))

    # Plot initial data
    reward_plot, = ax[0].plot([], [], label='Total Reward', color='blue')
    ax[0].set_title('Total Reward per Episode')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Total Reward')
    ax[0].legend()

    scatter_plot, = ax[1].plot([], [], label='Total Reward vs Total Moves', color='red', marker='o', linestyle='')
    ax[1].set_title('Total Reward vs Total Moves')
    ax[1].set_xlabel('# of Moves')
    ax[1].set_ylabel('Total Reward')
    ax[1].legend()

    # initialize empty lists for dynamic plotting
    episode_list = []
    reward_list = []
    move_list = []

    for episode in range(MAX_EPISODES):
        state = env.reset()
        total_reward = 0
        moves = 0
        
        while not env.game_over:
            action = agent.select_action(state)
            
            next_state, reward, done = env.step(action)
            reward = reward if not done else -10
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            moves += 1
            if done:
                print("Episode: {}/{}, Total Reward: {}, Epsilon: {:.2}".format(
                    episode + 1, MAX_EPISODES, total_reward, agent.epsilon))
                break
            
        episode_list.append(episode)
        reward_list.append(total_reward)
        move_list.append(moves)
        reward_plot.set_data(episode_list, reward_list)
        scatter_plot.set_data(move_list, reward_list)
        
        ax[0].relim()
        ax[0].autoscale_view()
        ax[1].relim()
        ax[1].autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
            
        env.render()
        
        if len(agent.memory) > BATCH_SIZE and ((episode+1) % TARGET_UPDATE_FREQ) == 0:
            agent.replay()

    plt.ioff()
    plt.show()
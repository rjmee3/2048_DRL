import numpy as np
from agent import Agent
import time
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os

base_dir = './data/'
os.makedirs(base_dir, exist_ok=True)

def transform_state(state):
    '''
    Reshapes state into a one-hot encoded vector. '''
    state = np.reshape(state, -1)
    state[state==0] = 1
    state = np.log2(state)
    state = state.astype(int)
    new_state = np.reshape(np.eye(18)[state], -1)
    return new_state
    
def dqn(agent, env, version, n_episodes=100, eps_start=0.05, eps_end=0.001, eps_decay=0.995,
        start_learn_iterations = 20, print_moves=False, stats_interval=100, save_interval=100,
        saving_model=True):
    '''
    Handles bulk of the training logic for Deep Q Learning. '''
    
    # initialize epsilon value and learning iterations
    eps = eps_start
    learn_iterations = start_learn_iterations
    
    # run training loop for n_episodes
    for _ in range(1, n_episodes + 1):
        # increment current iteration and get starting time for episode run
        agent.current_iteration = agent.current_iteration + 1
        time_start = time.time()
        
        # reset the environment
        env.reset(2)
        
        # initialize variables pre-episode
        state = transform_state(env.current_state())
        reward = env.reward
        total_rewards = reward
        score = env.score
        agent.total_steps = 0
        
        # play an entire game until game over
        while not env.done:
            # default reward to negative reward value
            reward = env.negative_reward

            # get action values by passing state through neural network
            action_values = agent.act(state)
            actions_sorted = [(index, value) for index, value in enumerate(action_values[0])]
            actions_sorted = sorted(actions_sorted, key=lambda x: x[1], reverse=True)
            random_action = random.choice(np.arange(agent.action_size))
            action_index = 0
            env.moved = False

            # loop while a move does not change the board
            while not env.moved:
                # choose a random action based on epsilon value
                if random.random() < eps:
                    action_elem = actions_sorted[random_action]
                else:
                    action_elem = actions_sorted[action_index]
                    action_index += 1
                    
                # pass action to environment and get next state as a result of action
                action = np.int64(action_elem[0])
                env.step(action, action_values)
                next_state = transform_state(env.current_state())

                # get reward as a result of the move
                reward = env.reward
                
                # calculate error
                error = np.abs(reward - action_elem[1]) ** 2

                # record game score and game over indicator
                score = env.score
                done = env.done
                
                # save step data to replay buffer
                agent.step(state, action, reward, next_state, done, error)
                
                # set current state to next state
                state = next_state
                
                # increment step and reward trackers
                agent.total_steps += 1
                total_rewards += reward

                # used for debug
                if print_moves:
                    env.draw_board()

                # break out of game loop if game over
                if done:
                    break
        
        # allow the agent to learn based on num of learn iterations
        agent.learn(learn_iterations, save_loss=True)
            
        # update max score board
        if score >= agent.max_score:
            agent.max_score = score
            agent.best_score_board = env.game_board.copy()
        
        # update max tile board
        if env.game_board.max() >= agent.max_val:
            agent.max_val = env.game_board.max()
            agent.best_val_board = env.game_board.copy()
        
        # append various data tracking metrics to their respective lists
        agent.max_vals_list.append(env.game_board.max())
        agent.max_steps_list.append(env.steps)
        agent.last_n_scores.append(score)
        agent.last_n_steps.append(env.steps)
        agent.last_n_total_rewards.append(total_rewards)
        agent.mean_steps.append(np.mean(agent.last_n_steps))
        agent.mean_total_rewards.append(np.mean(agent.last_n_total_rewards))
        
        # record end episode time
        time_end = time.time()
        
        # decay epsilon value
        eps = max(eps_end, eps_decay*eps)
        
        # print training metrics based on set interval
        if agent.current_iteration % stats_interval == 0:
            clear_output()
            
            # Training metrics
            
            # combined plot of highest tile and mean steps
            fig1, max_val_mean_steps_plot = plt.subplots()
            fig1.set_size_inches(16,6)
            max_val_mean_steps_plot.plot(agent.max_vals_list, 
                                        label='Max cell value seen on board', 
                                        alpha=0.5,
                                        color='tab:orange')
            max_val_mean_steps_plot.plot(agent.mean_steps, 
                                        label='Mean steps over last 50 episodes', 
                                        color='b')            
            fig1.tight_layout()
            plt.legend()
            plt.xlabel('Episode #')
            plt.title('Max Tile Reached Each Episode & Mean Steps Over Last 50 Episodes')
            plt.show()
            
            # mean reward plot
            fig2, mean_reward_plot = plt.subplots()
            fig2.set_size_inches(16, 6)
            mean_reward_plot.plot(agent.mean_total_rewards, 
                                  label='Mean total rewards over last 50 episodes', 
                                  color='r')
            fig2.tight_layout()
            plt.legend()
            plt.xlabel('Episode #')
            plt.title('Mean Reward Over Last 50 Episodes')
            plt.show()
            
            # loss plot
            fig3, loss_plot = plt.subplots()
            fig3.set_size_inches(16, 6)
            loss_plot.plot(agent.losses,
                           label='Loss',
                           color='c')
            fig3.tight_layout()
            plt.legend()
            plt.yscale('log')
            plt.xlabel('Episode #')
            plt.title('Loss')
            plt.show()

            # printing the best scoring board
            env.draw_board(agent.best_score_board, 'Best Scoring Board')

        # save model weights based on set save interval AND if choosing to save model
        if agent.current_iteration % save_interval == 0 and saving_model:
            agent.save(version)

        # formate episode string for output  
        s = '%d/%d | %0.2fs | Score:%d | Average Score:%d | Total Rewards:%d | Average Total Rewards:%d | Max Tile Reached:%d' %\
              (agent.current_iteration, n_episodes, time_end - time_start, score, np.mean(agent.last_n_scores), 
               total_rewards, np.mean(agent.last_n_total_rewards), np.max(agent.max_vals_list))
        
        # print episode string
        s = s + ' ' * (150-len(s))
        print(s, end='\r')
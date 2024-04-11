import numpy as np
from agent import Agent
import time
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pickle
import os
import torch
import math
import torch.nn.functional as F

base_dir = './data/'
os.makedirs(base_dir, exist_ok=True)

def save_state(name, eps, env, agent):
    with open(base_dir+'game_%s.pkl' % name, 'wb') as f:
        state = {
            'env': env,
            'last_eps': eps
        }
        pickle.dump(state, f)
    agent.save(name)

def save_game_history(name, best_history, worst_history):
    with open(base_dir+'/best_game_history_%s.pkl' % name, 'wb') as f:
        pickle.dump(best_history, f)
    with open(base_dir+'/worst_game_history_%s.pkl' % name, 'wb') as f:
        pickle.dump(worst_history, f)
    
def load_game_history(name):
    best_history = []
    worst_history = []
    if os.path.exists(base_dir+'/best_game_history_%s.pkl' % name):
        with open(base_dir+'/best_game_history_%s.pkl' % name, 'rb') as f:
            best_history = pickle.load(f)
    
    if os.path.exists(base_dir+'/worst_game_history_%s.pkl' % name):
        with open(base_dir+'/worst_game_history_%s.pkl' % name, 'rb') as f:
            worst_history = pickle.load(f)
    
    return best_history, worst_history
    
def load_state(name):
    with open(base_dir+'game_%s.pkl' % name, 'rb') as f:
        state = pickle.load(f)
        g = state['env']
        eps = state['last_eps']
    a = Agent(state_size=4 * 4 * 18, fc1_units=1024, fc2_units=1024, fc3_units = 1024)
    a.load(name)
    return g, a, eps

def transform_state(state, mode='plain'):
    if mode == 'plain':
        return np.reshape(state, -1)
    elif mode == 'plain_hw':
        return np.concatenate([np.reshape(state, -1), np.reshape(np.transpose(state), -1)])
    elif mode == 'log2':
        state = np.reshape(state, -1)
        state[state==0] = 1
        return np.log2(state) / 17
    elif mode == 'one_hot':
        state = np.reshape(state, -1)
        state[state==0] = 1
        state = np.log2(state)
        state = state.astype(int)
        new_state = np.reshape(np.eye(18)[state], -1)
        return new_state
    else:
        return state
    
def dqn(agent, version, env, n_episodes=100, eps_start=0.05, eps_end=0.001, eps_decay=0.995, 
        sample_mode='error', start_learn_iterations = 20):
    
    eps = eps_start
    eps_initial = eps_start
    eps_list = []
    starting_iteration = agent.current_iteration
    best_game_history, worst_game_history = load_game_history(version)
    learn_iterations = start_learn_iterations
    
    for i_episode in range(1, n_episodes+1):
        agent.current_iteration = agent.current_iteration + 1
        time_start = time.time()
        
        actions = np.array([0, 0, 0, 0])
        
        env.reset(2)
        
        state = transform_state(env.current_state(), mode = 'one_hot')
        reward = env.reward
        total_rewards = reward
        score = env.score
        agent.total_steps = 0
        eps_list.append(eps)
        
        while not env.done:
            reward = env.negative_reward
            
            action_values = agent.act(state)
            
            actions_sorted = [(index, value) for index, value in enumerate(action_values[0])]
            actions_sorted = sorted(actions_sorted, key=lambda x: x[1], reverse=True)
            
            random_action = random.choice(np.arange(agent.action_size))
            action_index = 0
            env.moved = False
            while not env.moved:
                if random.random() < eps:
                    action_elem = actions_sorted[random_action]
                else:
                    action_elem = actions_sorted[action_index]
                    action_index += 1
                    
                action = np.int64(action_elem[0])
                actions[action] += 1
                env.step(action, action_values)
                next_state = transform_state(env.current_state(), mode='one_hot')
                reward = env.reward
                
                error = np.abs(reward - action_elem[1]) ** 2
                score = env.score
                done = env.done
                
                if len(agent.actions_avg_list) > 0:
                    actions_dist = [np.mean(agent.actions_deque[i]) for i in range(4)][action]
                else:
                    actions_dist = (actions / np.sum(actions))[action]
                    
                agent.step(state, action, reward, next_state, done, error, actions_dist)
                
                state = next_state
                
                agent.total_steps += 1
                total_rewards += reward
                
                if done:
                    break
        
        agent.learn(learn_iterations, mode=sample_mode, save_loss=True, weight=env.game_board.max())
        
        actions = actions /env.steps
        
        agent.actions_deque[0].append(actions[0])
        agent.actions_deque[1].append(actions[1])
        agent.actions_deque[2].append(actions[2])
        agent.actions_deque[3].append(actions[3])
        
        agent.actions_avg_list.append([np.mean(agent.actions_deque[i]) for i in range(4)])
        
        if total_rewards > agent.max_total_reward:
            agent.max_total_reward = total_rewards
            
        if score >= agent.max_score:
            agent.max_score = score
            agent.best_score_board = env.game_board.copy()
            best_game_history = env.history.copy()
        
        if score < agent.min_score:
            agent.min_score = score
            worst_game_history = env.history.copy()
        
        if env.game_board.max() > agent.max_val:
            agent.max_val = env.game_board.max()
            agent.best_val_board = env.game_board.copy()
        
        if env.steps > agent.max_steps:
            agent.max_steps = env.steps
            agent.best_steps_board = env.game_board.copy()
            
        
        agent.total_rewards_list.append(total_rewards)
        agent.scores_list.append(score)
        agent.max_vals_list.append(env.game_board.max())
        agent.max_steps_list.append(env.steps)
        
        agent.last_n_scores.append(score)
        agent.last_n_steps.append(env.steps)
        agent.last_n_vals.append(env.game_board.max())
        agent.last_n_total_rewards.append(total_rewards)
        
        agent.mean_scores.append(np.mean(agent.last_n_scores))
        agent.mean_steps.append(np.mean(agent.last_n_steps))
        agent.mean_vals.append(np.mean(agent.last_n_vals))
        agent.mean_total_rewards.append(np.mean(agent.last_n_total_rewards))
        
        time_end = time.time()
              
        eps = max(eps_end, eps_decay*eps)
            
        if agent.current_iteration % 500 == 0:
            clear_output()
            
            # Training metrics
            
            # combined plot of highest tile and mean steps
            fig1, max_val_mean_steps_plot = plt.subplots()
            fig1.set_size_inches(16,6)
            max_val_mean_steps_plot.plot(agent.max_vals_list + [None for _ in range(10000 - len(agent.scores_list))], 
                                        label='Max cell value seen on board', 
                                        alpha=0.5,
                                        color='tab:orange')
            max_val_mean_steps_plot.plot(agent.mean_steps + [None for _ in range(10000 - len(agent.scores_list))], 
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
            mean_reward_plot.plot(agent.mean_total_rewards + [None for _ in range(10000 - len(agent.scores_list))], 
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
            
            # epsilon plot
            fig4, eps_plot = plt.subplots()
            fig4.set_size_inches(16, 6)
            eps_plot.plot(eps_list,
                          label='Epsilon per Episode',
                          color='k')
            fig4.tight_layout()
            plt.legend()
            plt.title('Epsilon Value Per Episode')
            plt.xlabel('Episode #')
            plt.show()

            env.draw_board(agent.best_score_board, 'Best score board')
            
            save_state(version, eps, env, agent)
            save_game_history(version, best_game_history, worst_game_history)
            
        s = '%d/%d | %0.2fs | Score:%d | Average Score:%d | Total Rewards:%d | Average Total Rewards:%d | Max Tile Reached:%d' %\
              (agent.current_iteration, starting_iteration + n_episodes, time_end - time_start, score, np.mean(agent.last_n_scores), 
               total_rewards, np.mean(agent.last_n_total_rewards), np.max(agent.max_vals_list))
        
        s = s + ' ' * (150-len(s))
        print(s, end='\r')
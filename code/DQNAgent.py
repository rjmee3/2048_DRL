import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import TensorBoard
from keras import callbacks
from env import get_valid_actions, preprocess_state
import os
import tensorflow as tf
import datetime

class DQNAgent:
    def __init__(self, state_shape, action_size, memory_size, 
                 gamma, epsilon, epsilon_min, epsilon_decay, 
                 learning_rate, batch_size, optimizer, 
                 loss_function):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.earlystopping_callback = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
        self.update_target_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=self.state_shape, padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.action_size, activation='linear'))  # Using linear activation for Q-values
        model.compile(optimizer=self.optimizer, loss=self.loss_function)
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def select_action(self, state):
        valid_actions = get_valid_actions(state)
        invalid_actions = list(set([0, 1, 2, 3]) - set(valid_actions))
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_actions)
        action_values = self.model.predict(preprocess_state(state).reshape(self.state_shape), verbose=0)
        if len(invalid_actions) > 0:
            action_values[0, invalid_actions] = float('-inf')
        return np.argmax(action_values[0])
    
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        # minibatch = self.memory
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_pred = self.target_model.predict(preprocess_state(next_state).reshape(self.state_shape), verbose=0)
                target = reward + self.gamma * np.amax(next_state_pred)
            state_pred = self.model.predict(preprocess_state(state).reshape(self.state_shape), verbose=0)
            
            # valid_actions_mask = np.zeros_like(state_pred)
            # valid_actions = get_valid_actions(state)
            # valid_actions_mask[0, valid_actions] = 1
            
            # masked_state_pred = state_pred * valid_actions_mask
            
            
            # masked_state_pred[0][action] = target
            
            target_f = state_pred
            target_f[0][action] = target
            
            states.append(preprocess_state(state).reshape(self.state_shape))
            targets.append(target_f)
            
        states = np.vstack(states)
        targets = np.vstack(targets)
        
        # print(states)
        # print(targets)
        
        self.model.fit(states, targets, epochs=100, callbacks=[self.tensorboard_callback, self.earlystopping_callback], verbose=1)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_target_model()
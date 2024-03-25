'''
    DOES NOT WORK
    Pulled from Github: https://github.com/tjwei/2048-NN
    in order to study a network that WORKS
'''



from random import randint, shuffle, seed
import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from IPython.display import clear_output
from c2048 import Game, push
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, InputLayer, batch_norm, DropoutLayer
from lasagne.layers import  MergeLayer, ReshapeLayer, FlattenLayer, ConcatLayer
floatX = theano.config.floatX
from lasagne.nonlinearities import rectify, elu, softmax, sigmoid
from lasagne.init import Constant, Sparse
floatX = theano.config.floatX

from lasagne.layers.dnn import Conv2DDNNLayer
from lasagne.regularization import regularize_network_params, l1, l2, regularize_layer_params_weighted

def Winit(shape):
    rtn = np.random.normal(size=shape).astype(floatX)
    rtn[np.random.uniform(size=shape) < 0.9] *= 0.01
    return rtn

input_var = T.tensor4()
target_var = T.vector()
N_FILTERS = 512
N_FILTERS2 = 4096

_ = InputLayer(shape=(None, 16, 4, 4), input_var=input_var)

conv_a =  Conv2DDNNLayer(_, N_FILTERS, (2,1), pad='valid')#, W=Winit((N_FILTERS, 16, 2, 1)))
conv_b =  Conv2DDNNLayer(_, N_FILTERS, (1,2), pad='valid')#, W=Winit((N_FILTERS, 16, 1, 2)))

conv_aa =  Conv2DDNNLayer(conv_a, N_FILTERS2, (2,1), pad='valid')#, W=Winit((N_FILTERS2, N_FILTERS, 2, 1)))
conv_ab =  Conv2DDNNLayer(conv_a, N_FILTERS2, (1,2), pad='valid')#, W=Winit((N_FILTERS2, N_FILTERS, 1, 2)))

conv_ba =  Conv2DDNNLayer(conv_b, N_FILTERS2, (2,1), pad='valid')#, W=Winit((N_FILTERS2, N_FILTERS, 2, 1)))
conv_bb =  Conv2DDNNLayer(conv_b, N_FILTERS2, (1,2), pad='valid')#, W=Winit((N_FILTERS2, N_FILTERS, 1, 2)))

_ = ConcatLayer([FlattenLayer(x) for x in [conv_aa, conv_ab, conv_ba, conv_bb, conv_a, conv_b]])
l_out = DenseLayer(_, num_units=1,  nonlinearity=None)

prediction = lasagne.layers.get_output(l_out)
P = theano.function([input_var], prediction)
loss = lasagne.objectives.squared_error(prediction, target_var).mean()/2
#layers = {conv1: 0.5, conv2: 0.5}
#l1_penalty = regularize_layer_params_weighted(layers, l1)
#loss = loss + 1e-4 * l1_penalty
accuracy = lasagne.objectives.squared_error(prediction, target_var).mean()
params = lasagne.layers.get_all_params(l_out, trainable=True)
#params = [l_out.W]
updates = lasagne.updates.adam(loss, params, beta1=0.5)
#updates = lasagne.updates.sgd(loss, params, learning_rate=α)
#updates = lasagne.updates.adamax(loss, params)


train_fn = theano.function([input_var, target_var], loss, updates=updates)
loss_fn = theano.function([input_var, target_var], loss)
accuracy_fn =theano.function([input_var, target_var], accuracy)

from random import randint
table ={2**i:i for i in range(1,16)}
table[0]=0
def make_input(grid):
    g0 = grid
    r = np.zeros(shape=(16, 4, 4), dtype=floatX)
    for i in range(4):
        for j in range(4):
            v = g0[i, j]
            r[table[v],i, j]=1
    return r

logf=open("logf-rl-theano-n-tuple-6", "w")
def printx(*a, **kw):
    print(*a, file=logf, flush=True, **kw)
    print(*a, flush=True, **kw)
    
from random import random, randint

def get_grid(driver):
    grid = np.zeros(shape=(4,4), dtype='uint16')
    for x in driver.find_elements_by_class_name('tile'):
        cl = x.get_attribute('class').split()
        for t in cl:
            if t.startswith('tile-position-'):
                pos = int(t[14])-1, int(t[16])-1
            elif t.startswith('tile-') and t[5].isdigit():
                v = int(t[5:])
        grid[pos[1], pos[0]] = v
    return grid

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (12.0, 8.0)

import math
import time
from collections import defaultdict
  
def Vchange(grid, v):
    g0 = grid
    g1 = g0[:,::-1,:]
    g2 = g0[:,:,::-1]
    g3 = g2[:,::-1,:]
    r0 = grid.swapaxes(1,2)
    r1 = r0[:,::-1,:]
    r2 = r0[:,:,::-1]
    r3 = r2[:,::-1,:]
    xtrain = np.array([g0,g1,g2,g3,r0,r1,r2,r3], dtype=floatX)
    ytrain = np.array([v]*8, dtype=floatX)
    train_fn(xtrain, ytrain)

arrow=[Keys.ARROW_LEFT, Keys.ARROW_UP, Keys.ARROW_RIGHT, Keys.ARROW_DOWN]
def gen_sample_and_learn(driver):
    body = driver.find_element_by_tag_name('body')
    game_len = 0
    game_score = 0
    last_grid = None
    keep_playing =False
    while True:
        try:
            grid_array = get_grid(driver)
        except:
            grid_array = None
        board_list = []
        if grid_array is not None:
            if not keep_playing and grid_array.max()==2048:
                driver.find_element_by_class_name('keep-playing-button').click()
                keep_playing = True
                time.sleep(1)
            for m in range(4):
                g = grid_array.copy()
                s = push(g, m%4)
                if s >= 0:
                    board_list.append( (g, m, s) )
        if board_list:
            boards = np.array([make_input(g) for g,m,s in board_list], dtype=floatX)
            p = P(boards).flatten()        
            game_len+=1
            best_move = -1
            best_v = None
            for i, (g,m,s) in enumerate(board_list):
                v = 2*s + p[i]
                if best_v is None or v > best_v:
                    best_v = v
                    best_move = m
                    best_score = 2*s
                    best_grid = boards[i]
            body.send_keys(arrow[best_move])
            game_score += best_score
        else:
            best_v = 0
            best_grid = None
        if last_grid is not None:
            Vchange(last_grid, best_v)       
        last_grid = best_grid
        if not board_list:
            break
        plt.pause(0.05)
    return game_len, grid_array.max(), game_score

results = []
driver = webdriver.Firefox()
graph = plt.plot([], [], 'b')[0]
dots256 = plt.plot([],[], 'ro')[0]
dots512 = plt.plot([],[], 'yo')[0]
dots1024 = plt.plot([],[], 'go')[0]
plt.xlim((0,100))
plt.ylim((0,25000))
for j in range(200):
    driver.get("https://gabrielecirulli.github.io/2048/")
    time.sleep(2)
    result = gen_sample_and_learn(driver)
    print(j, result)
    results.append(result)
    graph.set_data(np.arange(len(results)), np.array(results)[:, 2])
    dots_data =[[],[],[]]
    for i, d in enumerate(results):
        c = 0 if d[1]<=256 else (1 if d[1]==512 else 2)
        dots_data[c].append([i, d[2]])
    dots_graph = [dots256, dots512, dots1024]
    for i in range(3):
        if dots_data[i]:
            xy = np.array(dots_data[i])
            dots_graph[i].set_data(xy[:, 0], xy[:,1])
    plt.title("Game #%d"%j, fontsize=64)
    plt.draw()
    plt.pause(3)
    if result[1] >= 2048:
        break
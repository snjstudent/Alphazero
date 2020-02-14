from __future__ import absolute_import, division, print_function
import os
import glob
import cv2
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Conv2D, Input, Layer, Dense, Flatten, Conv2DTranspose, BatchNormalization, ReLU, Dropout, MaxPool2D, AveragePooling2D,Softmax
from tensorflow.keras.models import Model,Sequential
import tensorflow_addons as tfa
import numpy as np
import random
import sys
import time
from operator import attrgetter
import math
import itertools
global model
from datetime import datetime
import pickle
C_PUCT = 1.0


class Residual_Block(Model):
    def __init__(self, filters_num: int, dropout_rate: float = 0.1, stride: int = 1, *args, **kwargs):
        super(Residual_Block, self).__init__(*args, **kwargs)
        self.filters_num = filters_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        
        self.conv = Conv2D(filters=self.filters_num,kernel_size=1, strides=self.stride, padding='same')
        self.conv_1 = Conv2D(filters=self.filters_num,kernel_size=3, strides=self.stride, padding='same')
        self.conv_2 = Conv2D(filters=self.filters_num * 4, kernel_size=1, strides=self.stride, padding='same')
        self.conv_3 = Conv2D(filters=self.filters_num * 4, kernel_size=1, strides=self.stride, padding='same')
        self.batch_norm = BatchNormalization()
        self.batch_norm_1 = BatchNormalization()
        self.batch_norm_2 = BatchNormalization()
        self.relu = ReLU()
        self.dropout = Dropout(rate=self.dropout_rate)
        self.conv_3.trainable = False
    
    def call(self, inputs):
        batch_normed_1 = self.batch_norm(inputs)
        relued_1 = self.relu(batch_normed_1)
        conv_1 = self.conv(relued_1)
        batch_normed_2 = self.batch_norm_1(conv_1)
        relued_2 = self.relu(batch_normed_2)
        conv_2 = self.conv_1(relued_2)
        batch_normed_3 = self.batch_norm_2(conv_2)
        relued_3 = self.relu(batch_normed_3)
        conv_3 = self.conv_2(relued_3)
        conv_3 = self.conv_3(inputs) + conv_3
        return conv_3

class ResNet50(Model):
    def __init__(self, stride: int = 1, *args, **kwargs):
        super(ResNet50, self).__init__(*args, **kwargs)
        self.stride = stride
        self.avgpool = AveragePooling2D(padding='same',strides=1)
        self.maxpool = MaxPool2D(padding='same',strides=1)
        self.ResBlocks: List[Layers] = []  #= [Residual_Block(filters_num=4 * 2 ** (i)) for i in range(4)]
        layers_num = [3, 4, 6, 3]
        for i in range(len(layers_num)):
            for _ in range(layers_num[i]):
                self.ResBlocks.append(Residual_Block(filters_num=4 * 2 ** (i)))
        self.conv = Conv2D(filters=16, kernel_size=7, strides=self.stride, padding='same')
        self.softmax = Softmax()
        self.dense = Dense(1, activation='sigmoid')
        self.dense_1 = Dense(81, activation='softmax')
        self.flat = Flatten()


    def call(self, inputs):
        conv_1 = self.conv(inputs)
        maxpooled = self.maxpool(conv_1)
        for layer in self.ResBlocks:
            maxpooled = layer(maxpooled)
        avgpooled = self.avgpool(maxpooled)
        avgpooled = self.flat(avgpooled)
        value = self.dense(avgpooled)
        avgpooled = self.dense_1(avgpooled)
        return avgpooled, value

class AlphaZero_NN:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        input_shape = input_dim.shape
        self.input_layer = Input(shape=(input_shape[0], input_shape[1], 1))
        self.resnet = ResNet50()
        avgpool, value = self.resnet(self.input_layer)
        self.model = Model(inputs=[self.input_layer], outputs=[avgpool, value])

                

class node:
    def __init__(self, state, p):
        """
        p : policy
        w : value
        n : iterations
        """
        self.state = state
        self.p = p
        self.w = 0
        self.n = 0
        self.child_nodes = None
    
    
    def evaluate(self):
        if self.state.end:
            value = -1 if self.state.lose else 0
            self.w += value
            self.n += 1
            
            return value
        
        if not self.child_nodes:
            policies, value = model.model.predict(np.array([[self.state.board_1]], dtype=np.float32).reshape(1, 9, 9, 1))
            self.w += value
            self.n += 1
            self.child_nodes = tuple(node(self.state.next(action), policy) for action, policy in zip(self.state.legal_actions, policies[0]))
            
            return value
        
        else:
            value = -self.next_child_node().evaluate()
            self.w += value
            self.n += 1
            
            return value
    
    def next_child_node(self):
        def pucb_values():
            t = sum(map(attrgetter('n'), self.child_nodes))
            return tuple((-child_node.w / child_node.n if child_node.n else 0.0) + C_PUCT * child_node.p * math.sqrt(t) / (1 + child_node.n) for child_node in self.child_nodes)
        return self.child_nodes[np.argmax(pucb_values())]


    


class GameBoard:
    def __init__(self,
    board_1=[[0 for i in range(9)] for u in range(9)],
    board_2=[[0 for i in range(9)] for u in range(9)]):
        global model
        self.board_1 = board_1
        self.board_2 = board_2
        self.count = 0
        self.action_count = 0


    @property
    def lose(self):
        board_tmp = np.array(self.board_1).reshape(9, 9)
        for i in range(9):
            for u in range(9):
                self.depth_first_search(i, u, board_tmp)
                if self.count >= 5 or (self.board_1[i][u] * self.board_2[i][u]):
                    return True
                self.count = 0
        return False
    
    @property
    def end(self):
        return self.lose or self.draw
    
    @property
    def draw(self):
        return self.action_count == 81

    @property
    def legal_actions(self):
        tmp = []
        for i in range(len(self.board_1)):
            for u in range(len(self.board_1[i])):
                if self.board_1[i][u] == 0 and self.board_2[i][u] == 0:
                    tmp.append((i + 1) * (u * 1))
                else:
                    tmp.append(0)
        return tuple(tmp)

    def next(self, action):
        q, mod = divmod(len(self.board_1[0]), len(self.board_1[0]))
        self.board_1[q - 1 if q > 0 else q][mod - 1 if mod > 0 else mod] = 1
        return GameBoard(self.board_2, self.board_1)

    def depth_first_search(self, x, y, board):
        board[x][y] = 0
        for dx in range(-1, 1):
            for dy in range(-1, 1):
                nx, ny = x + dx, y + dy
                if (nx >= 0 and nx <= 9 and ny >= 0 and ny <= 9 and board[nx][ny] > 0):
                    self.depth_first_search(nx, ny, board)
                    count += 1
        return
    
                
        
class Train:
    def __init__(self, state):
        #self.state, self.policy = state.board_1
        global model
        #input_state = np.array([self.state.board_1], dtype=np.float32).reshape(9, 9, 1)
        #self.state, self.policy = model.model.predict([[input_state], [input_state]], batch_size=1)
        #self.mc_tree = node(self.state, self.policy)
    
    def save_data(self, states, ys):
        y_policies, y_values = ys
        now = datetime.now()

        for i in range(len(states)):
            with open('./data/{:04}-{:02}-{:02}-{:02}-{:02}-{:02}-{:04}-{:02}.pickle'.format(now.year, now.month, now.day, now.hour, now.minute, now.second, states[i].action_count, i), mode='wb') as f:
                pickle.dump((states[i], y_policies[i], y_values[i]), f)

    def first_player_value(self,state):
        if state.lose:
            return 1 if state.action_count == 81 else - 1
        return 0
            
    def play(self, epoch):
        for i in range(epoch):
            print("epoch", i)
            states = []
            self.state = GameBoard()
            ys = [[],None]
            while (True):
                if self.state.end:
                    break
                rootnode = node(self.state, 0)
                for i in range(20):
                    rootnode.evaluate()
                score = tuple(map(attrgetter('n'), rootnode.child_nodes))
                print(score)
                print(len(boltzman(score, 1.0)))
                policies = [0 for i in range(9 * 9)]
                for action, policy in zip(self.state.legal_actions, boltzman(score, 1.0)):
                    policies[action] = policy
                states.append(state)
                ys[0].append(policies)
                self.state = self.state.next(np.random.choice(self.state.legal_actions, p=boltzman(score, 1.0)))
            value = 0
            if self.state.lose:
                value = -1 if self.state.action_count % 2 == 0 else - 1
            value = self.first_player_value(self.state)
            ys[1] = tuple([value if i % 2 == 0 else - value for i in range(len(ys[0]))])
            self.save_data(states, ys)
    
    def load_data(self):
        played_lists = glob.glob("data/*")
        states, values, policies = [], [], []
        for i in played_lists:
            state, policy, value = pickle.load(i)
            states.append(state)
            values.append(value)
            policies.append(policy)
        return states, values, policies
    
    def compile_model(self, model_, model_name="tmp"):
        model_.compile(loss=['mean_squared_error', 'mean_squared_error'], optimizer=tfk.optimizers.Adam(lr=0.0002))
        if model_name != "tmp":
            model_.load_weights(model_name)
        return model_
        
    def match_model(self, play_count: int, model_camp, model_callenger):
        global model
        total_point: int = 0
        is_camp: bool = False
        
        for i in range(play_count):
            state = GameBoard()
            while (True):
                model = model_camp if is_camp else model_callenger
                if state.end:
                    break
                rootnode = node(self.state, 0)
                for i in range(20):
                    rootnode.evaluate()
                score = tuple(map(attrgetter('n'), rootnode.child_nodes))
                state = state.next(np.random.choice(state.legal_actions, p=boltzman(score, 1.0)))
                is_camp = ~is_camp
                
            total_point += self.first_player_value(state) if i % 2 == 0 else 1 - self.first_player_value
            print("end game: challenger win percentage is ", total_point / play_count)
        model = model_callenger if total_point / play_count > 0.5 else model_camp
        

def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / (sum(xs) + 1e-30) for x in xs]
 

if __name__ == "__main__":
    play_count = 2
    cycle_count = 3
    global model
    model = AlphaZero_NN(np.array([[0 for i in range(9)] for u in range(9)]))
    model_camp = AlphaZero_NN(np.array([[0 for i in range(9)] for u in range(9)]))
    for u in range(cycle_count):
        state = GameBoard()
        train = Train(state)
        train.play(1)
        states, values, policies = train.load_data()
        model_challenger = AlphaZero_NN(np.array([[0 for i in range(9)] for u in range(9)]))
        model_challenger.model.fit(states, [policies, values], epochs=100)
        train.match_model(100, model_camp.model if u == 0 else model, model_challenger.model)
    model.save_weights('champ_model.h5')

        
        
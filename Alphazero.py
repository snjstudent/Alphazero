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
global model

class Residual_Block(Model):
    def __init__(self, filters_num: int, dropout_rate: float = 0.1 ,stride: int = 1):
        
        self.filters_num = filters_num
        self.dropout_rate = dropout_rate
        self.conv = Conv2D(filters=self.filters_num,kernel_size=1, strides=self.stride, padding='same')
        self.conv_1 = Conv2D(filters=self.filters_num * 4,kernel_size=3, strides=self.stride, padding='same')
        self.conv_2 = Conv2D(filters=self.filters_num * 4, kernel_size=1, strides=self.stride, padding='same')
        self.batch_norm = BatchNormalization()
        self.relu = ReLU()
        self.dropout = Dropout(rate=self.dropout_rate)
    
    def call(self, inputs):
        batch_normed_1 = self.batch_norm(inputs)
        relued_1 = self.relu(batch_normed_1)
        conv_1 = self.conv(relued_1)
        batch_normed_2 = self.batch_norm(conv_1)
        relued_2 = self.relu(batch_normed_2)
        conv_2 = self.conv(relued_2)
        batch_normed_3 = self.batch_norm(conv_2)
        relued_3 = self.relu(batch_normed_3)
        conv_3 = self.conv(relued_3)
        conv_3 = inputs + conv_3
        return conv_3

class ResNet50(Model):
    def __init__(self):
        self.avgpool = AveragePooling2D()
        self.maxpool = MaxPool2D()
        self.ResBlocks: List[Layers] = [Residual_Block(filters_num=4 * 2 ** (i + 1)) for i in range(4)]
        self.conv = Conv2D(filters=16, kernel_size=7, strides=self.stride, padding='same')
        self.softmax = Softmax()

    def call(self, inputs):
        conv_1 = self.conv(inputs)
        maxpooled = self.maxpool(conv_1)
        for _ in range(3):
            maxpooled = self.ResBlocks[0](maxpooled)
        for _ in range(4):
            maxpooled = self.ResBlocks[1](maxpooled)
        for _ in range(6):
            maxpooled = self.ResBlocks[2](maxpooled)
        for _ in range(3):
            maxpooled = self.ResBlocks[3](maxpooled)
        avgpooled = self.avgpool(maxpooled)
        return self.softmax(avgpooled)


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
            policies, value = model.predict(self.state)
            self.w += value
            self.child_nodes = tuple(node(self.state.next(action), policy) for action, policy in zip(self.state.legal_actions, policies))
            
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
        return self.child_nodes[np.argmax(pucb_values)]


def pv_mcts_scores(self, model, evaluate_count, state):
    root_node = node(state, 0)
    for _ in range(evaluate_count):
        root_node.evaluate()
    return tuple(map(attrgetter('n'), root_node.child_nodes))
    
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

class GameBoard:
    def __init__(self,
    board_1=[[0 for i in range(9)] for u in range(9)],
    board_2=[[0 for i in range(9)] for u in range(9)]):
        self.board_1 = board_1
        self.board_2 = board_2
        self.count = 0
        self.action_count = 0


    @property
    def lose(self):
        board_tmp = self.board_2
        for i in range(9):
            for u in range(9):
                self.depth_first_search(i, u, board_tmp)
                if self.count >= 5 or (self.board_1[i][u] & self.board_2[i][u]):
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
        pass

    def next(self, action):
        return GameBoard(self.board_2, self.board_1)

    def depth_first_search(self, x, y, board):
        board[x][y] = 0
        for dx in range(-1, 1):
            for dy in range(-1, 1):
                nx, ny = x + dx, y + dy
                if (nx >= 0 and nx <= 9 and ny >= 0 and ny <= 9 and board[nx][ny] == 1):
                    self.depth_first_search(nx, ny, board)
                    count += 1
        return
    
                
        
class Train:
    def __init__(self, state):
        self.state = state
        model = ResNet50()
        self.state, self.policy = model.predict(self.state)
        self.mc_tree = node(self.state, self.policy)

    def train(self, epoch):
        for i in range(epoch):
            nowtime = time.time()
            while (time.time() - nowtime < 1):
                self.mc_tree.evaluate()
            mc_tree = mc_tree.next_child_node()

if __name__ == "__main__":
    state = GameBoard()
    train = Train(state)
       
            
                 



        
        
import torch
import random
import numpy as np
from collections import deque
from snake import SnakeGame, Direction, Point

MAX_MEMORY = 100_000 #Can store 100k items in mem
BATCH_SIZE = 1000
LEARNING_RATE = 0.01

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # Controls randomness
        self.gamma = 0 # Discount rate
        self.in_memory = deque(maxlen=MAX_MEMORY)

        # TODO: model, trainer

    def get_state(self, game):
        pass

    def memory(self, state, action, reward, next_state, game_over):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, game_over):
        pass

    def get_action(self, state):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0

    agent = Agent()
    game = SnakeGame()

    while True:
        # get current state
        current_state = agent.get_state(game)

        # get move
        move = agent.get_action(current_state)

        # perform move and get new state
        reward, game_over, score = game.play_step(move)
        new_state = agent.get_state(game)

        #train short memory (only 1 step)
        agent.train_short_memory(current_state, move, reward, new_state, game_over)

        #Store in memory
        agent.memory(current_state, move, reward, new_state, game_over)

        if game_over:
            # train long memory and plot result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                #TODO: save model

            print('Game ', agent.num_games, 'Score: ', score, 'Best: ', best_score)

            #TODO: Plot



if __name__ == '__main__':
    train()

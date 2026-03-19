import torch
import random
import numpy as np
from collections import deque
from snake import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from plot import plot
import os

MAX_MEMORY = 100_000 #Can store 100k items in mem
BATCH_SIZE = 1000
LEARNING_RATE = 0.01

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # Controls randomness
        self.gamma = 0.9 # Discount rate 
        self.in_memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, g=self.gamma)

        # Load saved model if it exists
        if os.path.exists('./model/model.pth'):
            self.model.load_state_dict(torch.load('./model/model.pth'))
            print('Loaded saved model')

    def get_state(self, game):
        head = game.snake[0]
        pl = Point(head.x - 20, head.y)
        pr = Point(head.x + 20, head.y)
        pu = Point(head.x, head.y - 20)
        pd = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(pr)) or
            (dir_l and game.is_collision(pl)) or
            (dir_u and game.is_collision(pu)) or
            (dir_d and game.is_collision(pd)),

            # Danger right
            (dir_u and game.is_collision(pr)) or
            (dir_d and game.is_collision(pl)) or
            (dir_l and game.is_collision(pu)) or
            (dir_r and game.is_collision(pd)),

            # Danger left
            (dir_d and game.is_collision(pr)) or
            (dir_u and game.is_collision(pl)) or
            (dir_r and game.is_collision(pu)) or
            (dir_l and game.is_collision(pd)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # Food is left
            game.food.x > game.head.x, # Food is right
            game.food.y < game.head.y, # Food is up
            game.food.y > game.head.y # Food is down
        ]

        return np.array(state, dtype=int)

    def memory(self, state, action, reward, next_state, game_over):
        self.in_memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.in_memory) > BATCH_SIZE:
            rand_sample = random.sample(self.in_memory, BATCH_SIZE) # Tuple list

        else:
            rand_sample = self.in_memory

        states, actions, rewards, next_states, game_over = zip(*rand_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_over)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # Random moves
        self.epsilon = max(0, 200 - self.num_games)
        move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            i = random.randint(0, 2)
            move[i] = 1
        else:
            this_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(this_state)
            i = torch.argmax(prediction).item()
            move[i] = 1

        return move
        
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
                agent.model.save()

            print('Game ', agent.num_games, 'Score: ', score, 'Best: ', best_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()

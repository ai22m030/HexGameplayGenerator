import torch
import numpy as np


class EpsilonGreedyActionSelector:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, q_values):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(len(q_values))
        else:
            return np.argmax(q_values)

    def decay_epsilon(self, decay_factor):
        self.epsilon *= decay_factor


class QLearningAgent:
    def __init__(self, board_size, lr, gamma, action_selector):
        self.lr = lr
        self.gamma = gamma
        self.action_selector = action_selector
        self.q_table = torch.zeros(board_size, board_size)

    def choose_action(self, action_space):
        q_values = [self.q_table[i][j] for i, j in action_space]
        action_index = self.action_selector.select_action(q_values)
        return action_space[action_index]

    def best_action(self, valid_actions):
        q_values = [self.q_table[i][j] for i, j in valid_actions]
        max_q_value = max(q_values)
        best_action_index = q_values.index(max_q_value)
        return valid_actions[best_action_index]

    def update(self, game, action, reward):
        i, j = action
        current_q = self.q_table[i][j]

        valid_actions = game.get_action_space()

        next_q = 0
        if valid_actions:
            next_action = self.best_action(valid_actions)
            next_i, next_j = next_action
            next_q = self.q_table[next_i][next_j]

        self.q_table[i][j] = current_q + self.lr * (reward + self.gamma * next_q - current_q)

    def save(self, filename):
        torch.save(self.q_table, filename + ".pt")

    def load(self, filename):
        self.q_table = torch.load(filename + ".pt")

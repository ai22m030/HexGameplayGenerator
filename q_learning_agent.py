import torch
import numpy as np


def get_valid_actions(action_space):
    valid_actions = [(i, j, k) for i, j in action_space for k in range(3)]
    return valid_actions


class QLearningAgent:
    def __init__(self, board_size, lr, gamma, epsilon):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = torch.zeros(board_size, board_size, 3)

    def choose_action(self, action_space):
        valid_actions = get_valid_actions(action_space)

        if np.random.uniform(0, 1) < self.epsilon:
            random_index = np.random.randint(len(valid_actions))
            action = valid_actions[random_index]
        else:
            action = self.best_action(valid_actions)

        return action

    def best_action(self, valid_actions):
        q_values = [self.q_table[i][j][k] for i, j, k in valid_actions]
        max_q_value = max(q_values)
        best_action_index = q_values.index(max_q_value)
        return valid_actions[best_action_index]

    def update(self, game, action, reward):
        i, j, k = action
        current_q = self.q_table[i][j][k]

        valid_actions = get_valid_actions(game.get_action_space())

        next_q = 0
        if valid_actions:
            next_action = self.best_action(valid_actions)
            next_i, next_j, next_k = next_action
            next_q = self.q_table[next_i][next_j][next_k]

        self.q_table[i][j][k] = current_q + self.lr * (reward + self.gamma * next_q - current_q)

    def save(self, filename):
        torch.save(self.q_table, filename + ".pt")

    def load(self, filename):
        self.q_table = torch.load(filename + ".pt")

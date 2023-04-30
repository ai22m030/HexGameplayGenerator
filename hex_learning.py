import torch.nn.functional as F
import torch.nn as nn
import torch
import pickle
import datetime
import numpy as np


def get_winning_path(board, player):
    size = len(board)
    visited = set()

    def dfs(x, y):
        if x < 0 or x >= size or y < 0 or y >= size or board[x][y] != player or (x, y) in visited:
            return []
        if (player == 1 and y == size - 1) or (player == -1 and x == size - 1):
            return [(x, y)]

        visited.add((x, y))
        adjacents = [(x - 1, y), (x - 1, y + 1), (x, y - 1), (x, y + 1), (x + 1, y), (x + 1, y - 1)]

        for a, b in adjacents:
            dfs_path = dfs(a, b)
            if dfs_path:
                return [(x, y)] + dfs_path

        return []

    if player == 1:
        for i in range(size):
            path = dfs(i, 0)
            if path:
                return path
    elif player == -1:
        for j in range(size):
            path = dfs(0, j)
            if path:
                return path
    return []


def get_average_winning_path(plays, player):
    winning_path_lengths = [len(get_winning_path(gameplay[player], player)) for gameplay, winner in plays if
                            winner == player]
    return np.mean(winning_path_lengths)


def write_plays(name, plays):
    with open(f'{name}_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.pkl', 'wb') as f:
        pickle.dump(plays, f)


def load_plays(name):
    with open(f'{name}_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.pkl', 'rb') as f:
        return pickle.load(f)


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class HexCNN(nn.Module):
    def __init__(self, size):
        super(HexCNN, self).__init__()
        self.board_size = size
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        fc_input_size = self.calculate_fc_input_size()
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, size * size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def calculate_fc_input_size(self):
        size = self.board_size - 4  # Adjust for conv1 and conv2 without padding
        size = size - 2  # Adjust for conv3 with padding=1
        size = size - 2  # Adjust for conv4 with padding=1
        return 64 * size * size

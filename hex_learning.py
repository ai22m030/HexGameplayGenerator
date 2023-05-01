import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import datetime
import numpy as np
from collections import deque
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class HexDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_winning_path(board, player):
    size = len(board)
    visited = set()

    def bfs(start_x, start_y):
        queue = deque([(start_x, start_y, [])])

        while queue:
            x, y, path = queue.popleft()
            if x < 0 or x >= size or y < 0 or y >= size or board[x][y] != player or (x, y) in visited:
                continue

            visited.add((x, y))
            new_path = path + [(x, y)]

            if (player == 1 and y == size - 1) or (player == -1 and x == size - 1):
                return new_path

            adjacents = [(x - 1, y), (x - 1, y + 1), (x, y - 1), (x, y + 1), (x + 1, y), (x + 1, y - 1)]
            for a, b in adjacents:
                queue.append((a, b, new_path))

        return []

    if player == 1:
        for i in range(size):
            path = bfs(i, 0)
            if path:
                return path
    elif player == -1:
        for j in range(size):
            path = bfs(0, j)
            if path:
                return path
    return []


def get_average_winning_path(plays, player):
    total_length = 0
    total_games = 0

    for gameplay, moves, winner in plays:
        if winner == player:
            total_length += len(gameplay)
            total_games += 1

    if total_games == 0:
        return 0

    return total_length / total_games


def plot_win_rates(rates):
    white_wins = np.array([x['white'] for x in rates])
    black_wins = np.array([x['black'] for x in rates])
    games_played = np.arange(1, len(rates) + 1)  # Change this line

    white_win_rate = white_wins / games_played
    black_win_rate = black_wins / games_played

    plt.plot(games_played, white_win_rate, label='White')
    plt.plot(games_played, black_win_rate, label='Black')

    plt.xlabel('Game plays')
    plt.ylabel('Cumulative Win Rate')
    plt.title('Cumulative Win Rate Over Time')
    plt.legend()

    plt.show()


def plot_losses(train_losses, val_losses, epochs):
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()

    plt.show()


def plot_cumulative_rewards(cumulative_rewards, num_games):
    plt.plot(range(1, num_games + 1), cumulative_rewards, label='Cumulative Rewards')

    plt.xlabel('Game plays')
    plt.ylabel('Cumulative Rewards')
    plt.title('Q-learning Agent Improvement')
    plt.legend()

    plt.show()


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
    def __init__(self, board_size):
        super(HexCNN, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Add this line
        self.fc1 = nn.Linear(16 * (board_size // 2) * (board_size // 2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, board_size * board_size)

    def forward(self, x):
        # Reshape the input data to have a single channel
        x = x.view(-1, 1, self.board_size, self.board_size)

        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        # Calculate the output size of the final pooling layer
        pool_output_size = x.size(1) * x.size(2) * x.size(3)

        # Flatten the tensor and pass it through the fully connected layers
        x = x.view(-1, pool_output_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
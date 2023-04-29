import random
import numpy as np
import pickle
from hex_engine import HexPosition
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from mcts import MCTS


def get_winning_path(board):
    size = len(board)
    visited = set()
    q = [(i, 0) for i in range(size) if board[i][0] == 1]
    while q:
        curr = q.pop(0)
        if curr[1] == size - 1:
            return [curr]
        visited.add(curr)
        adjacents = [(curr[0] - 1, curr[1]), (curr[0] - 1, curr[1] + 1), (curr[0], curr[1] - 1),
                     (curr[0], curr[1] + 1), (curr[0] + 1, curr[1]), (curr[0] + 1, curr[1] - 1)]
        adjacents = [a for a in adjacents if a[0] >= 0 and a[0] < size and a[1] >= 0 and a[1] < size
                     and board[a[0]][a[1]] == 1 and a not in visited]
        q += adjacents
    return []


def get_average_winning_path(gameplays, player):
    winning_path_lengths = [len(get_winning_path(gameplay[player])) for gameplay, winner in gameplays]
    return np.mean(winning_path_lengths)


def generate_gameplays(num_games, size, mcts_iterations=10, mcts_max_iterations=100, mcts_timeout=0.1):
    gameplays = []
    switch_players_every = 10
    for i in range(num_games):
        print(f"Generating gameplay {i + 1} out of {num_games}")
        hex_position = HexPosition(size=size)
        mcts = MCTS(hex_position, n_simulations=mcts_iterations, max_iterations=mcts_max_iterations)
        gameplay = [deepcopy(hex_position.board)]

        first_player = 1 if (i // switch_players_every) % 2 == 0 else -1
        mcts_player = first_player

        while hex_position.winner == 0:
            if mcts_player == hex_position.player:
                action = mcts.run(timeout=mcts_timeout)  # Pass the timeout here
                hex_position.move(action)
                gameplay.append(deepcopy(hex_position.board))
            else:
                hex_position.move(random.choice(hex_position.get_action_space()))
                gameplay.append(deepcopy(hex_position.board))

            if hex_position.winner != 0:
                break

        gameplays.append((deepcopy(gameplay), hex_position.winner))
        hex_position.reset()

    return gameplays


def prepare_data(gameplays, board_size, player):
    X = []
    Y = []

    for gameplay in gameplays:
        hex_position = HexPosition(size=board_size)
        for i in range(0 if player == 1 else 1, len(gameplay) - 1, 2):
            if i == 0:
                current_board = np.zeros((board_size, board_size))
            else:
                current_board = np.array(gameplay[i - 1])

            next_board = np.array(gameplay[i])

            hex_position.board = current_board
            if hex_position.winner == 0:
                X.append(current_board)

                diff_board = next_board - current_board
                move_coordinates = np.unravel_index(np.argmax(diff_board, axis=None), diff_board.shape)

                y = np.zeros(board_size * board_size)
                y[hex_position.coordinate_to_scalar(tuple(move_coordinates))] = 1
                Y.append(y)

    X = np.array(X).reshape(-1, 1, board_size, board_size)
    Y = np.array(Y)

    if player == -1:
        X_reversed = []
        for x in X:
            flipped_board = HexPosition(size=board_size)
            flipped_board.board = x[0]
            X_reversed.append(np.array(flipped_board.recode_black_as_white()))
        X = np.array(X_reversed).reshape(-1, 1, board_size, board_size)

    return X, Y


class HexCNN(nn.Module):
    def __init__(self, board_size):
        super(HexCNN, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32, board_size * board_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.sigmoid(self.fc(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train_model(X, Y, model, epochs=10, batch_size=16):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_samples = len(X)

    if num_samples < batch_size:
        batch_size = num_samples

    while batch_size > 1 and num_samples % batch_size != 0:
        batch_size = batch_size - 1

    if batch_size <= 1 and num_samples > 1:
        print("Warning: Using a batch size of 1.")
        batch_size = 1

    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            x_batch, y_batch = X[start_idx:end_idx], Y[start_idx:end_idx]

            # Convert numpy arrays to tensors
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            y_batch = torch.tensor(y_batch, dtype=torch.float32)

            optimizer.zero_grad()

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / num_batches
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")


def select_gameplays_based_on_threshold(gameplays, threshold_ratio):
    avg_winning_path_white = get_average_winning_path(gameplays, 1)
    avg_winning_path_black = get_average_winning_path(gameplays, -1)
    threshold_white = threshold_ratio * avg_winning_path_white
    threshold_black = threshold_ratio * avg_winning_path_black

    selected_gameplays_white = [gameplay for gameplay in gameplays if len(get_winning_path(gameplay[1])) >= 6 and len(get_winning_path(gameplay[1])) <= threshold_white]
    selected_gameplays_black = [gameplay for gameplay in gameplays if len(get_winning_path(gameplay[-1])) >= 6 and len(get_winning_path(gameplay[-1])) <= threshold_black]

    return selected_gameplays_white, selected_gameplays_black


if __name__ == "__main__":
    board_size = 7
    gameplay_count = 20

    # Generate and save gameplays to file
    print("Generating gameplays...")
    with open('gameplays.pkl', 'wb') as f:
        pickle.dump(generate_gameplays(gameplay_count, board_size, mcts_max_iterations=100, mcts_timeout=0.1), f)

    # Load gameplays from file
    print("Loading gameplays...")
    with open('gameplays.pkl', 'rb') as f:
        gameplays = pickle.load(f)

    # Calculate average winning paths for white and black players
    print("Calculating average winning paths...")
    avg_winning_path_white = get_average_winning_path(gameplays, 1)
    avg_winning_path_black = get_average_winning_path(gameplays, -1)

    # Define a threshold for selecting significantly better gameplays (e.g., 0.75 = 75% of the average)
    threshold = 0.75

    # Separate gameplays won by white and black players
    gameplays_won_by_white = [gameplay for gameplay, winner in gameplays if winner == 1]
    gameplays_won_by_black = [gameplay for gameplay, winner in gameplays if winner == -1]

    # Select gameplays with winning paths significantly better than the average
    print("Selecting gameplays based on threshold...")
    selected_gameplays_white = [gameplay for gameplay in gameplays_won_by_white if
                                threshold * avg_winning_path_white >= len(get_winning_path(gameplay[-1]))]

    selected_gameplays_black = [gameplay for gameplay in gameplays_won_by_black if
                                threshold * avg_winning_path_black >= len(get_winning_path(gameplay[1]))]

    hex_cnn = HexCNN(board_size)

    print("Preparing data for white player...")
    if len(selected_gameplays_white) > 0:
        X_white, Y_white = prepare_data(selected_gameplays_white, board_size, 1)
        print(f"Number of selected gameplays for white player: {len(selected_gameplays_white)}")
        if len(X_white) > 0:
            print("Training model for white player...")
            train_model(X_white, Y_white, hex_cnn)
        else:
            print("No data available for training the white player.")
    else:
        print("No gameplays were selected for the white player.")

    print("Preparing data for black player...")
    if len(selected_gameplays_black) > 0:
        X_black, Y_black = prepare_data(selected_gameplays_black, board_size, -1)
        print(f"Number of selected gameplays for black player: {len(selected_gameplays_black)}")
        if len(X_black) > 0:
            print("Training model for black player...")
            train_model(X_black, Y_black, hex_cnn)
        else:
            print("No data available for training the black player.")
    else:
        print("No gameplays were selected for the black player.")

    print("Saving the trained model...")
    torch.save(hex_cnn.state_dict(), "hex_cnn.pth")
    print("Model saved!")

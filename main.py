import datetime
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from hex_engine import HexPosition
from q_learning_agent import QLearningAgent
from mcts import MCTS
from copy import deepcopy
from sklearn.model_selection import train_test_split
from pathlib import Path


class HexAgent:
    def __init__(self, board_size):
        self.board_size = board_size
        self.hex_cnn = HexCNN(board_size)

    def generate_gameplays(self, num_games, mcts_iterations=10, mcts_max_iterations=100, mcts_timeout=0.1):
        return generate_gameplays(num_games, self.board_size, mcts_iterations, mcts_max_iterations, mcts_timeout)

    def prepare_data(self, gameplays, player):
        return prepare_data(gameplays, self.board_size, player)

    def train_model(self, X, Y, epochs=10, batch_size=16):
        train_model(X, Y, self.hex_cnn, epochs, batch_size)

    def save_model(self, model_path):
        torch.save(self.hex_cnn.state_dict(), model_path)

    def load_model(self, model_path):
        self.hex_cnn.load_state_dict(torch.load(model_path))


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
    q_agent = QLearningAgent(board_size=size ** 2, lr=0.1, gamma=0.99, epsilon=0.1)

    # Update the load path to check if the file exists before loading
    q_agent_file = Path("q_agent")
    if q_agent_file.is_file():
        q_agent.load("q_agent")

    gameplays = []

    for i in range(num_games):
        print(f"Generating gameplay {i + 1} out of {num_games}")
        hex_position = HexPosition(size=size)
        mcts = MCTS(hex_position, n_simulations=mcts_iterations, max_iterations=mcts_max_iterations)
        gameplay = [deepcopy(hex_position.board)]

        # Change player order every 5 episodes
        if i % 5 == 0:
            player_order = -1
        else:
            player_order = 1

        while hex_position.winner == 0:
            if hex_position.player == player_order:  # Use MCTS for the current player
                action = mcts.run(timeout=mcts_timeout)  # Pass the timeout here
                hex_position.move(action)
                gameplay.append(deepcopy(hex_position.board))
                if hex_position.winner != 0:
                    break
            else:  # Use Q-Learning agent for the other player
                q_action = q_agent.choose_action(hex_position.get_action_space())
                hex_position.move(q_action)
                gameplay.append(deepcopy(hex_position.board))

        if hex_position.winner == player_order:
            q_agent.update(hex_position, q_action, -10)
        elif hex_position.winner == -player_order:
            q_agent.update(hex_position, q_action, 10)

        gameplays.append((deepcopy(gameplay), hex_position.winner))
        hex_position.reset()

    q_agent.save("q_agent")

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


def train_model(X, Y, model, epochs=10, batch_size=6, early_stopping_rounds=5, validation_split=0.2):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split, random_state=42)

    num_samples = len(X_train)

    if num_samples < batch_size:
        batch_size = num_samples

    while batch_size > 1 and num_samples % batch_size != 0:
        batch_size = batch_size - 1

    if batch_size <= 1 and num_samples > 1:
        print("Warning: Using a batch size of 1.")
        batch_size = 1

    num_batches = num_samples // batch_size

    best_val_loss = float("inf")
    rounds_since_best_val_loss = 0

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            x_batch, y_batch = X_train[start_idx:end_idx], Y_train[start_idx:end_idx]

            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            y_batch = torch.tensor(y_batch, dtype=torch.float32)

            optimizer.zero_grad()

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / num_batches

        model.eval()
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, Y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            rounds_since_best_val_loss = 0
        else:
            rounds_since_best_val_loss += 1

        print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if rounds_since_best_val_loss >= early_stopping_rounds:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break


if __name__ == "__main__":
    # Board size
    board_size = 7
    # Samples to generate
    gameplay_count = 10000
    # Define a threshold for selecting significantly better gameplays (e.g., 0.75 = 75% of the average)
    threshold = 0.60

    hex_agent = HexAgent(board_size)

    print("Generating gameplays...")
    gameplays = hex_agent.generate_gameplays(gameplay_count, board_size, mcts_max_iterations=100, mcts_timeout=0.1)

    # Calculate average winning paths for white and black players
    avg_winning_path_white = get_average_winning_path(gameplays, 1)
    avg_winning_path_black = get_average_winning_path(gameplays, -1)

    # Separate gameplays won by white and black players
    gameplays_won_by_white = [gameplay for gameplay, winner in gameplays if winner == 1]
    print(f"Gameplays won by white: {len(gameplays_won_by_white)}")

    gameplays_won_by_black = [gameplay for gameplay, winner in gameplays if winner == -1]
    print(f"Gameplays won by black: {len(gameplays_won_by_black)}")

    selected_gameplays_white = [gameplay for gameplay in gameplays_won_by_white if
                                threshold * avg_winning_path_white >= len(get_winning_path(gameplay[-1]))]

    selected_gameplays_black = [gameplay for gameplay in gameplays_won_by_black if
                                threshold * avg_winning_path_black >= len(get_winning_path(gameplay[1]))]

    now = datetime.datetime.now()

    # Format the date and time as a string
    timestamp = now.strftime("%Y%m%d%H%M%S")

    # Save gameplays to file
    with open(f'gameplays_white_{timestamp}.pkl', 'wb') as f:
        pickle.dump(selected_gameplays_white, f)

    with open(f'gameplays_black_{timestamp}.pkl', 'wb') as f:
        pickle.dump(selected_gameplays_black, f)

    hex_cnn = HexCNN(board_size)

    hex_cnn_file = Path("hex_cnn")
    if hex_cnn_file.is_file():
        print("CNN model loading...")
        hex_cnn = torch.load("hex_cnn.pth")

    # Load gameplays from file
    with open(f'gameplays_white_{timestamp}.pkl', 'rb') as f:
        selected_gameplays_white = pickle.load(f)

    with open(f'gameplays_black_{timestamp}.pkl', 'rb') as f:
        selected_gameplays_black = pickle.load(f)

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

    torch.save(hex_cnn.state_dict(), "hex_cnn.pth")
    print("CNN model saved!")

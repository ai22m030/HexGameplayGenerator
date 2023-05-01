import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from hex_engine import HexPosition
from hex_learning import HexCNN, get_average_winning_path, get_winning_path, write_plays, load_plays
from q_learning_agent import QLearningAgent, EpsilonGreedyActionSelector
from mcts import MCTS
from copy import deepcopy
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt


class HexAgent:
    def __init__(self, size):
        self.board_size = size
        self.hex_cnn = HexCNN(size)

    def generate_game_plays(self, num_games, mcts_iterations=10, mcts_max_iterations=100, mcts_timeout=0.1):
        return generate_plays(num_games, self.board_size, mcts_iterations, mcts_max_iterations, mcts_timeout)

    def prepare_data(self, plays, player):
        return prepare_data(plays, self.board_size, player)

    def train_model(self, X, Y, epochs=10, batch_size=16):
        train_model(X, Y, self.hex_cnn, epochs, batch_size)

    def save_model(self, model_path):
        torch.save(self.hex_cnn.state_dict(), model_path)

    def load_model(self, model_path):
        self.hex_cnn.load_state_dict(torch.load(model_path))


def generate_plays(num_games, size, mcts_iterations=10, mcts_max_iterations=100, mcts_timeout=0.1):
    selector = EpsilonGreedyActionSelector(0.1)
    q_agent = QLearningAgent(board_size=size ** 2, lr=0.1, gamma=0.99, action_selector=selector)

    q_agent_file = Path("q_agent")
    if q_agent_file.is_file():
        q_agent.load("q_agent")

    unique_plays = {}
    win_counts = {'white': 0, 'black': 0}
    rates = []
    cumulative_rewards = []
    total_reward = 0

    while len(unique_plays) < num_games:
        print(f"Generating gameplay {len(unique_plays) + 1} out of {num_games}")
        hex_position = HexPosition(size=size)
        mcts = MCTS(hex_position, n_simulations=mcts_iterations, max_iterations=mcts_max_iterations)
        gameplay = [deepcopy(hex_position.board)]

        if len(unique_plays) % 5 == 0:
            player_order = -1
        else:
            player_order = 1

        q_action = None  # Initialize q_action with a default value before the loop starts
        while hex_position.winner == 0:
            if hex_position.player == player_order:
                action = mcts.run(timeout=mcts_timeout)
                hex_position.move(action)
                gameplay.append(deepcopy(hex_position.board))
                if hex_position.winner != 0:
                    break
            else:
                q_action = q_agent.choose_action(hex_position.get_action_space())
                hex_position.move(q_action)
                gameplay.append(deepcopy(hex_position.board))

            player_order *= -1

        if hex_position.winner == player_order:
            if q_action is not None:
                reward = -10
                q_agent.update(hex_position, q_action, reward)
            win_counts['black'] += 1
        elif hex_position.winner == -player_order:
            if q_action is not None:
                reward = 10
                q_agent.update(hex_position, q_action, reward)
            win_counts['white'] += 1
        total_reward += reward
        cumulative_rewards.append(total_reward)
        hashable_gameplay = pickle.dumps((deepcopy(gameplay), hex_position.winner))
        unique_plays[hashable_gameplay] = (deepcopy(gameplay), hex_position.winner)
        rates.append(win_counts.copy())
        hex_position.reset()
    q_agent.save("q_agent")
    plays = list(unique_plays.values())
    return plays, rates, cumulative_rewards


def prepare_data(hex_position, plays, board_size, player):
    X = []
    Y = []

    for play in plays:
        gameplay = play  # Don't unpack the tuple

        for i in range(len(gameplay) - 1):
            current_board = np.zeros((board_size, board_size))
            next_board = np.zeros((board_size, board_size))

            for j in range(i + 1):
                row, col = gameplay[j]
                current_board[row][col] = player if j % 2 == 0 else -player

            row, col = gameplay[i + 1]
            next_board[row][col] = player if (i + 1) % 2 == 0 else -player

            if player == -1:
                current_board = hex_position.recode_black_as_white(invert_colors=True)
                next_board = hex_position.recode_black_as_white(invert_colors=True)

            X.append(current_board)
            Y.append(next_board)

    return np.array(X), np.array(Y)


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

    train_losses = []
    val_losses = []

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
        train_losses.append(epoch_loss)

        model.eval()
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, Y_val_tensor).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            rounds_since_best_val_loss = 0
        else:
            rounds_since_best_val_loss += 1

        print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if rounds_since_best_val_loss >= early_stopping_rounds:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    plot_losses(train_losses, val_losses, epoch + 1)


def plot_win_rates(rates, num_games):
    white_wins = np.array([x['white'] for x in rates])
    black_wins = np.array([x['black'] for x in rates])
    games_played = np.arange(1, num_games + 1)

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


if __name__ == "__main__":
    board_size = 7
    gameplay_count = 2
    threshold = 0.8

    hex_agent = HexAgent(board_size)

    print("Generating game plays...")
    game_plays, win_rates, cumulative_rewards = hex_agent.generate_game_plays(gameplay_count, board_size,
                                                                              mcts_max_iterations=100, mcts_timeout=0.3)

    plot_win_rates(win_rates, gameplay_count)
    plot_cumulative_rewards(cumulative_rewards, gameplay_count)

    # Calculate average winning paths for white and black players
    avg_winning_path_white = get_average_winning_path(game_plays, 1)
    avg_winning_path_black = get_average_winning_path(game_plays, -1)

    # Separate game plays won by white and black players
    plays_won_by_white = [gameplay for gameplay, winner in game_plays if winner == 1]
    print(f"Game plays won by white: {len(plays_won_by_white)}")

    plays_won_by_black = [gameplay for gameplay, winner in game_plays if winner == -1]
    print(f"Game plays won by black: {len(plays_won_by_black)}")

    selected_gameplays_white = [gameplay for gameplay in plays_won_by_white if
                                threshold * avg_winning_path_white >= len(get_winning_path(gameplay[-1], 1))]
    selected_gameplays_black = [gameplay for gameplay in plays_won_by_black if
                                threshold * avg_winning_path_black >= len(get_winning_path(gameplay[-1], -1))]

    # Save gameplays to file
    write_plays("gameplays_white", selected_gameplays_white)
    write_plays("gameplays_black", selected_gameplays_black)

    hex_cnn = HexCNN(board_size)

    hex_cnn_file = Path("hex_cnn")
    if hex_cnn_file.is_file():
        print("CNN model loading...")
        hex_cnn = torch.load("hex_cnn.pth")

    # Load gameplays from file
    # selected_gameplays_white = load_plays('gameplays_white')
    # selected_gameplays_black = load_plays('gameplays_black')

    if len(selected_gameplays_white) > 0:
        hex_position = HexPosition(board_size)
        X_white, Y_white = prepare_data(HexPosition(board_size), selected_gameplays_white, board_size, 1)
        if len(X_white) > 0:
            print(f"Training model for white player: {len(selected_gameplays_white)}")
            train_model(X_white, Y_white, hex_cnn)
        else:
            print("No data available for training the white player.")

    if len(selected_gameplays_black) > 0:
        X_black, Y_black = prepare_data(HexPosition(board_size), selected_gameplays_black, board_size, -1)
        if len(X_black) > 0:
            print(f"Training model for black player: {len(selected_gameplays_black)}")
            train_model(X_black, Y_black, hex_cnn)
        else:
            print("No data available for training the black player.")

    torch.save(hex_cnn.state_dict(), "hex_cnn.pth")
    print("CNN model saved!")

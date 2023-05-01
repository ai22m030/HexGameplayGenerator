import os
import random

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from hex_engine import HexPosition
from hex_learning import HexCNN, get_average_winning_path, get_winning_path, write_plays, load_plays, plot_win_rates, \
    plot_losses, plot_cumulative_rewards, HexDataset
from q_learning_agent import QLearningAgent, EpsilonGreedyActionSelector
from mcts import MCTS
from copy import deepcopy
from sklearn.model_selection import train_test_split
from pathlib import Path


def custom_score(gameplay, size, tolerance=1.5):
    winning_path_length = len(get_winning_path(gameplay[0][-1], gameplay[2]))
    num_moves = len(gameplay[0])
    score = winning_path_length / num_moves
    return score if winning_path_length <= size * tolerance else 0


def generate_plays(num_games, size, random_count, mcts_iterations=10, mcts_max_iterations=100, mcts_timeout=0.1):
    q_agent = QLearningAgent(board_size=size ** 2, lr=0.1, gamma=0.99, action_selector=EpsilonGreedyActionSelector(0.1))
    mcts = MCTS(None, n_simulations=mcts_iterations, max_iterations=mcts_max_iterations)

    q_agent_file = Path("q_agent")
    if q_agent_file.is_file():
        q_agent.load("q_agent")

    unique_plays = {}
    win_counts = {'white': 0, 'black': 0}
    rates = []
    cum_rewards = []
    total_reward = 0
    current_unique_plays = 0
    reward = None
    q_action = None

    current_white = "mcts"
    current_black = "q"

    while len(unique_plays) < num_games:
        print(f"Generating gameplay {len(unique_plays) + 1} out of {num_games}")
        hex_position = HexPosition(size=size)
        mcts.update_root(hex_position)
        gameplay = [deepcopy(hex_position.board)]
        moves = []

        if len(unique_plays) % 5 == 0:
            player_order = -1
        else:
            player_order = 1

        while hex_position.winner == 0:
            if (current_white == "mcts" and hex_position.player == 1) or \
                    current_black == "mcts" and hex_position.player == -1:
                action = mcts.run(timeout=mcts_timeout, agent=lambda x: q_agent.choose_action(x))
                hex_position.move(action)
                gameplay.append(deepcopy(hex_position.board))
                moves.append(action)
                if hex_position.winner != 0:
                    break
            elif (current_white == "q" and hex_position.player == 1) or \
                    (current_black == "q" and hex_position.player == -1):
                q_action = q_agent.choose_action(hex_position.get_action_space())
                hex_position.move(q_action)
                q_agent.update(hex_position, q_action, 0)

                gameplay.append(deepcopy(hex_position.board))
                moves.append(q_action)
                if hex_position.winner != 0:
                    break

        if hex_position.winner == player_order:
            reward = -10
        elif hex_position.winner == -player_order:
            reward = 10

        if (reward is not None) and (q_action is not None):
            q_agent.update(hex_position, q_action, reward)

        if reward is not None:
            total_reward += reward
            cum_rewards.append(total_reward)

        hashable_gameplay = pickle.dumps((deepcopy(gameplay), deepcopy(moves), hex_position.winner))
        unique_plays[hashable_gameplay] = (deepcopy(gameplay), deepcopy(moves), hex_position.winner)

        if current_unique_plays != len(unique_plays):
            win_counts['white' if hex_position.winner == 1 else 'black'] += 1
            rates.append(win_counts.copy())

        current_unique_plays = len(unique_plays)

        hex_position.reset()
        current_white, current_black = current_black, current_white

    hex_position = HexPosition(size=size)
    while len(unique_plays) < num_games + random_count:
        print(f"Generating random gameplay {len(unique_plays) + 1} out of {num_games + random_count}")

        if len(unique_plays) >= random_count:
            print(f"Still not enough: white-{win_counts['white']}/black-{win_counts['black']}")

        gameplay = [deepcopy(hex_position.board)]
        moves = []

        while hex_position.winner == 0 and len(unique_plays) < num_games + random_count:
            move = random.choice(hex_position.get_action_space())
            gameplay.append(deepcopy(hex_position.board))
            hex_position.move(move)
            moves.append(move)

        hashable_gameplay = pickle.dumps((deepcopy(gameplay), deepcopy(moves), hex_position.winner))
        unique_plays[hashable_gameplay] = (deepcopy(gameplay), deepcopy(moves), hex_position.winner)
        win_counts['white' if hex_position.winner == 1 else 'black'] += 1
        rates.append(win_counts.copy())
        hex_position.reset()

    q_agent.save("q_agent")

    return list(unique_plays.values()), rates, cum_rewards


def prepare_data(plays, size, player):
    game = HexPosition(size)
    X = []
    Y = []

    for gameplay, moves, winner in plays:
        for i in range(len(gameplay) - 1):

            if player == -1 and (i % 2 == 0):
                game.reset()
                game.board = gameplay[i]
                X.append(game.recode_black_as_white(invert_colors=True))
                game.reset()
                game.move(game.recode_coordinates(moves[i]))
                Y.append(game.board)
            elif player == 1 and (i % 2 != 0):
                X.append(gameplay[i])
                game.reset()
                game.move(moves[i])
                Y.append(game.board)

    return X, Y


def custom_loss(outputs, targets, valid_moves):
    outputs = outputs.masked_fill(valid_moves == 0, float('-inf'))
    outputs = F.softmax(outputs.view(-1, outputs.size(-1)), dim=-1)
    targets = targets.view(-1)
    return F.cross_entropy(outputs, targets)


def train_model(X, Y, model, board_size, epochs=10, batch_size=6, early_stopping_rounds=5, validation_split=0.2):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split, random_state=42)

    # Create custom dataset
    train_dataset = HexDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(Y_train, dtype=torch.float))
    val_dataset = HexDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(Y_val, dtype=torch.float))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    best_val_loss = float("inf")
    rounds_since_best_val_loss = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(x_batch)

            # Mask invalid moves
            hex_pos = HexPosition(board_size)
            for i, board_state in enumerate(x_batch):
                hex_pos.board = board_state.squeeze(0).tolist()
                valid_moves = hex_pos.get_action_space()
                valid_move_mask = torch.zeros_like(outputs[i])
                for move in valid_moves:
                    valid_move_mask[hex_pos.coordinate_to_scalar(move)] = 1
                outputs[i] = outputs[i] * valid_move_mask

            # Calculate loss
            print("x_batch shape:", x_batch.shape)
            print("y_batch shape:", y_batch.shape)

            loss = criterion(outputs.view(-1, board_size, board_size), y_batch.view(-1, board_size, board_size))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.unsqueeze(1).float()
            y_batch = y_batch.view(-1)

            val_outputs = model(x_batch)
            val_outputs_flat = val_outputs.view(-1, board_size * board_size)
            y_batch_flat = y_batch.view(-1, board_size * board_size)
            val_loss = criterion(val_outputs_flat, y_batch_flat)

            val_running_loss += val_loss.item()

        if len(val_loader) > 0:
            val_loss = val_running_loss / len(val_loader)
        else:
            val_loss = 0

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


if __name__ == "__main__":
    isRunningInPyCharm = "PYCHARM_HOSTED" in os.environ

    if isRunningInPyCharm:
        print("verbose")
    else:
        print("not verbose")

    board_size = 7
    gameplay_count = 10
    random_gameplay_count = 190
    threshold = 2

    game = HexPosition(board_size)

    print("Generating game plays...")
    game_plays, win_rates, cumulative_rewards = generate_plays(gameplay_count, board_size, mcts_max_iterations=100,
                                                               mcts_timeout=0.3, random_count=random_gameplay_count)

    plot_win_rates(win_rates)
    plot_cumulative_rewards(cumulative_rewards, gameplay_count)

    # Calculate average winning paths for white and black players
    avg_winning_path_white = get_average_winning_path(game_plays, 1)
    avg_winning_path_black = get_average_winning_path(game_plays, -1)

    # Separate game plays won by white and black players
    plays_won_by_white = [(gameplay, moves, winner) for gameplay, moves, winner in game_plays if winner == 1]
    print(f"Game plays won by white: {len(plays_won_by_white)}")

    plays_won_by_black = [(gameplay, moves, winner) for gameplay, moves, winner in game_plays if winner == -1]
    print(f"Game plays won by black: {len(plays_won_by_black)}")

    selected_gameplays_white = list(filter(lambda x: custom_score(x, board_size, threshold) != 0, plays_won_by_white))
    selected_gameplays_black = list(filter(lambda x: custom_score(x, board_size, threshold) != 0, plays_won_by_black))

    # Save gameplays to file
    write_plays("gameplays_white", selected_gameplays_white)
    write_plays("gameplays_black", selected_gameplays_black)

    hex_cnn = HexCNN(board_size)

    hex_cnn_file = Path("hex_cnn")
    if hex_cnn_file.is_file():
        print("CNN model loading...")
        hex_cnn = torch.load("hex_cnn.pth")

    # Load gameplays from file
    if selected_gameplays_white is None:
        selected_gameplays_white = load_plays('gameplays_white')

    if selected_gameplays_black is None:
        selected_gameplays_black = load_plays('gameplays_black')

    print(f"Selected white: {len(selected_gameplays_white)} / black: {len(selected_gameplays_black)}")

    if len(selected_gameplays_white) > 0:
        X_white, Y_white = prepare_data(selected_gameplays_white, board_size, 1)
        print(f"Training model for white player: {len(selected_gameplays_white)}")
        train_model(X_white, Y_white, hex_cnn, board_size)

    if len(selected_gameplays_black) > 0:
        X_black, Y_black = prepare_data(selected_gameplays_black, board_size, -1)
        print(f"Training model for black player: {len(selected_gameplays_white)}")
        train_model(X_black, Y_black, hex_cnn, board_size)

    torch.save(hex_cnn.state_dict(), "hex_cnn.pth")

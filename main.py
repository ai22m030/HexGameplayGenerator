import random
import numpy as np
import pickle
from hex_engine import HexPosition


def get_winning_path(board):
    size = len(board)
    visited = set()
    q = [(i, 0) for i in range(size) if board[i][0] == 1]
    while q:
        curr = q.pop(0)
        if curr[1] == size - 1:
            return [curr]
        visited.add(curr)
        adjacents = [(curr[0]-1, curr[1]), (curr[0]-1, curr[1]+1), (curr[0], curr[1]-1),
                     (curr[0], curr[1]+1), (curr[0]+1, curr[1]), (curr[0]+1, curr[1]-1)]
        adjacents = [a for a in adjacents if a[0]>=0 and a[0]<size and a[1]>=0 and a[1]<size
                     and board[a[0]][a[1]] == 1 and a not in visited]
        q += adjacents
    return []


def generate_gameplays(num_games, hex_position):
    gameplays = []

    for _ in range(num_games):
        while hex_position.winner == 0:
            hex_position.move(random.choice(hex_position.get_action_space()))
            if hex_position.winner != 0:
                break
            hex_position.move(random.choice(hex_position.get_action_space()))
        gameplay = hex_position.history
        gameplays.append(gameplay)
        hex_position.reset()

    return gameplays

def get_average_winning_path(gameplays, player):
    winning_path_lengths = [len(get_winning_path(gameplay[player])) for gameplay in gameplays]
    return np.mean(winning_path_lengths)


def filter_unique_gameplays(gameplays):
    unique_gameplays = []
    final_states = set()

    for gameplay in gameplays:
        final_state = tuple(map(tuple, gameplay[-1]))
        if final_state not in final_states:
            unique_gameplays.append(gameplay)
            final_states.add(final_state)

    return unique_gameplays


if __name__ == '__main__':
    gameplays = filter_unique_gameplays(generate_gameplays(1000, HexPosition()))

    # Calculate average winning paths for white and black players
    avg_winning_path_white = get_average_winning_path(gameplays, -1)
    avg_winning_path_black = get_average_winning_path(gameplays, 1)

    # Define a threshold for selecting significantly better gameplays (e.g., 0.75 = 75% of the average)
    threshold = 0.75

    # Select gameplays with winning paths significantly better than the average
    selected_gameplays_white = [gameplay for gameplay in gameplays if len(get_winning_path(gameplay[-1])) <= threshold * avg_winning_path_white]
    selected_gameplays_black = [gameplay for gameplay in gameplays if len(get_winning_path(gameplay[1])) <= threshold * avg_winning_path_black]

    # Save the unique gameplays to files
    with open('unique_gameplays_white.pkl', 'wb') as f:
        pickle.dump(selected_gameplays_white, f)

    with open('unique_gameplays_black.pkl', 'wb') as f:
        pickle.dump(selected_gameplays_black, f)

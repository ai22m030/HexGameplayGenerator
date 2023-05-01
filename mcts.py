import math
from copy import deepcopy
from random import choice
import time


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.children = []
        self.parent = parent
        self.visits = 0
        self.wins = 0
        self.last_move = action

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def uct(self, exploration_coefficient=1):
        return (self.wins / self.visits) + exploration_coefficient * math.sqrt(
            2 * math.log(self.parent.visits) / self.visits)

    def fully_expanded(self):
        unexpanded_moves = set(self.state.get_action_space()) - set(child.last_move for child in self.children)
        return len(unexpanded_moves) == 0

    def is_leaf(self):
        return len(self.children) == 0

    def best_child(self, exploration_coefficient=1):
        if not self.children:
            return None

        best_child = max(self.children, key=lambda child: child.uct(exploration_coefficient))
        return best_child


class MCTS:
    def __init__(self, hex_position, n_simulations, exploration_weight=1, max_iterations=100):
        self.hex_position = hex_position
        self.n_simulations = n_simulations
        self.exploration_weight = exploration_weight
        self.root = Node(state=self.hex_position)
        self.max_iterations = max_iterations

    def run(self, timeout=1, agent=lambda x: choice(x)):
        start_time = time.time()
        iterations = 0
        while time.time() - start_time < timeout and iterations < self.max_iterations:
            iterations += 1
            selected_node = self.select()
            if selected_node.state.winner == 0:
                expanded_node = self.expand(selected_node)
                reward = self.simulate(expanded_node, agent)
                self.backpropagate(expanded_node, reward)
            else:
                self.backpropagate(selected_node, selected_node.state.winner)

        best_child_node = self.root.best_child(0)
        while best_child_node is not None and best_child_node.last_move not in self.root.state.get_action_space():
            best_child_node = best_child_node.best_child(0)

        if best_child_node is None or best_child_node.last_move is None:
            return choice(self.root.state.get_action_space())

        return best_child_node.last_move  # Return the action (last_move) of the best child node

    def select(self):
        current_node = self.root
        while not current_node.is_leaf() and current_node.fully_expanded():
            current_node = current_node.best_child(self.exploration_weight)
        return current_node

    def expand(self, node):
        unoccupied_positions = set(node.state.get_action_space()) - set(child.last_move for child in node.children)
        random_action = choice(list(unoccupied_positions))
        new_state = deepcopy(node.state)
        new_state.move(random_action)
        new_node = Node(state=new_state, parent=node, action=random_action)
        node.children.append(new_node)
        return new_node

    def simulate(self, node, agent):
        simulation_state = deepcopy(node.state)
        while simulation_state.winner == 0:
            random_action = choice(simulation_state.get_action_space())
            simulation_state.move(random_action)
            if simulation_state.winner != 0:
                break
            # Now the opponent makes a random move
            random_action = agent(simulation_state.get_action_space())
            simulation_state.move(random_action)

        return 1 if simulation_state.winner == self.hex_position.player else -1

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            if node.state.player == self.hex_position.player and reward == 1:
                node.wins += 1
            elif node.state.player != self.hex_position.player and reward == -1:
                node.wins += 1
            node = node.parent

    def update_root(self, new_hex_position):
        self.hex_position = new_hex_position
        self.root = Node(state=self.hex_position)

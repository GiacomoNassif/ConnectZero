from __future__ import annotations

from copy import deepcopy

import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod

from ConnectGame import __Connect__


class MonteCarloTreeSearchNode(ABC):

    def __init__(self, state, parent=None):
        """
        Parameters
        ----------
        state : mctspy.games.common.TwoPlayersAbstractGameState
        parent : MonteCarloTreeSearchNode
        """
        self.state = state
        self.parent = parent
        self.children = []

    @property
    @abstractmethod
    def untried_actions(self):
        """
        Returns
        -------
        list of mctspy.games.common.AbstractGameAction
        """
        pass

    @property
    @abstractmethod
    def q(self):
        pass

    @property
    @abstractmethod
    def n(self):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def is_terminal_node(self):
        pass

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def choose_move(self, possible_moves):
        return possible_moves[np.random.choice(len(possible_moves))]


class MCTSNode(MonteCarloTreeSearchNode):

    def __init__(self, state: __Connect__, parent: MCTSNode = None):
        super().__init__(state, parent)
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._untried_actions = None

    @property
    def untried_actions(self) -> list:
        if self._untried_actions is None:
            self._untried_actions = list(self.state.get_legal_actions())
        return self._untried_actions

    @property
    def q(self):
        """Defined in Connect games as wins-losses"""
        current_turn = self.state.current_players_turn
        next_turn = self.state.next_players_turn
        wins = self._results[current_turn]
        losses = self._results[next_turn]
        # My coding sucks reverse this. TODO: Fix this black magic.
        return -(wins - losses)

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        # Take the final action and pop it.
        action = self.untried_actions.pop()

        next_state = deepcopy(self.state)
        next_state.take_action(action)

        child_node = MCTSNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self) -> bool:
        return self.state.is_game_over

    def simulate(self):
        """Do a simulation of the rest of the game"""
        current_simulated_state = deepcopy(self.state)

        while not current_simulated_state.is_game_over:
            # Get all possible moves
            possible_moves = list(current_simulated_state.get_legal_actions())

            # Select an action according to the function
            # With a policy network, we'd change this
            action = self.choose_move(possible_moves)

            # Take the action
            current_simulated_state.take_action(action)

        return current_simulated_state.winner

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

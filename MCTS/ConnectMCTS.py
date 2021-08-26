from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import List

from Connect.ConnectGame import __Connect__, __ConnectAction__, __PlayerTurn__
from MCTS.MCTSNode import MonteCarloTreeSearchNode


class ConnectMCTS(MonteCarloTreeSearchNode):

    def __init__(self, state: __Connect__, parent: ConnectMCTS = None):
        super().__init__(state, parent)
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._untried_actions = None

    @property
    def untried_actions(self) -> List[__ConnectAction__]:
        if self._untried_actions is None:
            self._untried_actions = list(self.state.get_legal_actions())
        return self._untried_actions

    @property
    def q(self) -> float:
        """Defined in Connect games as wins-losses from the perspective of the parent player"""

        wins = self._results[self.parent.state.current_players_turn]
        losses = self._results[self.parent.state.next_players_turn]

        return wins - losses

    @property
    def n(self) -> int:
        return self._number_of_visits

    def expand(self) -> ConnectMCTS:
        # Take the final action and pop it.
        action = self.untried_actions.pop()

        next_state = deepcopy(self.state)
        next_state.take_action(action)

        child_node = ConnectMCTS(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self) -> bool:
        return self.state.is_game_over

    def simulate(self) -> __PlayerTurn__:
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

    def backpropagate(self, result: __PlayerTurn__) -> None:
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

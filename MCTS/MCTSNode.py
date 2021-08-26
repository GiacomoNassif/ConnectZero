from __future__ import annotations

from typing import List

import numpy as np
from abc import ABC, abstractmethod

from Connect.ConnectGame import __Connect__


class MonteCarloTreeSearchNode(ABC):

    def __init__(self, state: __Connect__, parent: MonteCarloTreeSearchNode = None):
        self.state = state
        self.parent = parent
        self.children: List[MonteCarloTreeSearchNode] = []

    @property
    @abstractmethod
    def untried_actions(self) -> List:
        pass

    @property
    @abstractmethod
    def q(self) -> float:
        pass

    @property
    @abstractmethod
    def n(self) -> int:
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def is_terminal_node(self) -> bool:
        pass

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def choose_move(self, possible_moves):
        return possible_moves[np.random.choice(len(possible_moves))]

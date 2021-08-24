from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.signal import convolve2d


class __PlayerTurn__(IntEnum):
    EMPTY = 0
    HERO = 1
    VILLAIN = -1


@dataclass
class __ConnectState__:
    """Class for keeping track of current board state and player move"""
    board: np.ndarray
    current_turn: __PlayerTurn__
    required_to_win: int
    winner: __PlayerTurn__ = __PlayerTurn__.EMPTY

    def get_next_player(self):
        if self.current_turn is __PlayerTurn__.HERO:
            return __PlayerTurn__.VILLAIN
        return __PlayerTurn__.HERO

    def update(self, action):
        self.board[action] = self.current_turn
        self.current_turn = self.get_next_player()

    @property
    def is_game_over(self):
        return self.winner is not __PlayerTurn__.EMPTY or 0 not in self.board


class __Connect__(ABC):
    """
    This is the underlying class.
    Each method returns a new instance of the class. Doesn't actually change base class.
    """

    def __init__(self, state: __ConnectState__):
        self.state = state

    @abstractmethod
    def legal_action(self, action):
        """Implement this method to determine what game we play."""
        pass

    # @staticmethod # For some reason this doesn't work...
    def _assert_legal_action(take_action):
        """A function decorate to take_action to ensure the move is legal."""

        def wrapper(self, action):
            assert self.legal_action(action), f'The action {action} is illegal.'
            take_action(self, action)

        return wrapper

    def _assert_game_not_over(take_action):
        """Assert the game isn't over"""

        def wrapper(self, action):
            assert self.state.winner is __PlayerTurn__.EMPTY, f'The game has finished!'
            take_action(self, action)

        return wrapper

    # TODO: This is pretty trash too.
    def _check_winner(take_action):
        """Move, then check if someone has won the game"""

        def wrapper(self, action):
            take_action(self, action)

            if self.check_winning_move(self.state.get_next_player()):
                self.state.winner = self.state.get_next_player()

        return wrapper

    @_assert_game_not_over
    @_assert_legal_action
    @_check_winner
    def take_action(self, action):
        self.state.update(action)

    def get_legal_actions(self):
        """Return all legal actions."""
        for action, value in np.ndenumerate(self.state.board):
            if self.legal_action(action):
                yield action

    @property
    def current_players_turn(self):
        return self.state.current_turn

    @property
    def next_players_turn(self):
        return self.state.get_next_player()

    @property
    def is_game_over(self):
        return self.state.is_game_over

    @property
    def winner(self):
        return self.state.winner

    # TODO: Improve. This method is trash.
    def check_winning_move(self, player) -> bool:
        """Check if player x has won"""
        horizontal_kernel = np.array([[1 for _ in range(self.state.required_to_win)]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(self.state.required_to_win, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

        for kernel in detection_kernels:
            if (convolve2d(self.state.board == player, kernel, mode='valid') == self.state.required_to_win).any():
                return True
        return False


class XesAndOes(__Connect__):
    base_state = __ConnectState__(np.full((3, 3), __PlayerTurn__.EMPTY),
                                  __PlayerTurn__.HERO,
                                  3)

    def __init__(self, state=base_state):
        super().__init__(state)

    def legal_action(self, action):
        return self.state.board[action] == __PlayerTurn__.EMPTY

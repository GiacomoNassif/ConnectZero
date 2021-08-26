from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, List

import numpy as np
from scipy.signal import convolve2d

# Name our connect action as a tuple of two int.
# TODO: Will have to generalise this for larger dimensions later!
__ConnectAction__ = Tuple[int, int]


# Player turn enum.
class __PlayerTurn__(IntEnum):
    EMPTY = 0
    HERO = 1
    VILLAIN = -1


@dataclass
class __ConnectState__:
    """Class for keeping track of current board state and player move"""
    board: np.ndarray
    current_turn: __PlayerTurn__
    winner: __PlayerTurn__ = __PlayerTurn__.EMPTY

    def get_next_player(self) -> __PlayerTurn__:
        # Player 1 is positive 1 and player 2 is -1 (Look above).
        return __PlayerTurn__(-self.current_turn)

    def get_previous_player(self) -> __PlayerTurn__:
        return self.get_next_player()

    def update_state(self, action: __ConnectAction__):
        """Take an action."""
        self.board[action] = self.current_turn
        self.current_turn = self.get_next_player()

    def is_game_over(self) -> bool:
        """The winner is not empty or the board does not contain empty spaces."""
        return self.winner is not __PlayerTurn__.EMPTY or 0 not in self.board


class __Connect__(ABC):
    """
    This is the underlying class.
    Each method returns a new instance of the class. Doesn't actually change base class.
    """

    class _Decorators:
        @classmethod
        def assert_legal_action(cls, take_action):
            """A function decorate to take_action to ensure the move is legal."""

            def wrapper(self: __Connect__, action: __ConnectAction__):
                assert self.legal_action(action), f'The action {action} is illegal.'
                take_action(self, action)

            return wrapper

        @classmethod
        def assert_game_not_over(cls, take_action):
            """Assert the game isn't over"""

            def wrapper(self: __Connect__, action: __ConnectAction__):
                assert not self.state.is_game_over(), f'The game has finished!'
                take_action(self, action)

            return wrapper

        @classmethod
        def check_winner(cls, take_action):
            """Move, then check if someone has won the game"""

            def wrapper(self: __Connect__, action: __ConnectAction__):
                # Get the current player and see if he wins with this move.
                current_player = self.state.current_turn

                take_action(self, action)

                if self.check_winning_move(current_player):
                    self.state.winner = current_player

            return wrapper

    def __init__(self, state: __ConnectState__, required_to_win: int):
        self.state = state
        self.required_to_win = required_to_win

    @abstractmethod
    def legal_action(self, action: __ConnectAction__) -> bool:
        """Implement this method to determine what game we play."""
        pass

    @_Decorators.assert_game_not_over
    @_Decorators.assert_legal_action
    @_Decorators.check_winner
    def take_action(self, action: __ConnectAction__):
        self.state.update_state(action)

    def get_legal_actions(self) -> List[__ConnectAction__]:
        """Return all legal actions."""
        legal_actions = []
        for action, value in np.ndenumerate(self.state.board):
            if self.legal_action(action):
                legal_actions.append(action)
        return legal_actions

    @property
    def current_players_turn(self) -> __PlayerTurn__:
        return self.state.current_turn

    @property
    def next_players_turn(self) -> __PlayerTurn__:
        return self.state.get_next_player()

    @property
    def is_game_over(self) -> bool:
        return self.state.is_game_over()

    @property
    def winner(self) -> __PlayerTurn__:
        return self.state.winner

    # TODO: Improve. This method is trash.
    """
    Can be improved by giving it the latest action as the input. this will cut down the search space
    significantly. Also, this monster is the only thing that limits us from moving to different
    dimensions other than 2d.
    """

    def check_winning_move(self, player) -> bool:
        """Check if player x has won"""
        horizontal_kernel = np.array([[1 for _ in range(self.required_to_win)]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(self.required_to_win, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

        for kernel in detection_kernels:
            if (convolve2d(self.state.board == player, kernel, mode='valid') == self.required_to_win).any():
                return True
        return False

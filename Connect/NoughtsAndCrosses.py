import numpy as np

from Connect.ConnectGame import __Connect__, __ConnectState__, __PlayerTurn__, __ConnectAction__


class NoughtsAndCrosses(__Connect__):
    base_state = __ConnectState__(board=np.full((3, 3), __PlayerTurn__.EMPTY),
                                  current_turn=__PlayerTurn__.HERO)
    required_to_win = 3

    def __init__(self, state: __ConnectState__ = base_state):
        super().__init__(state, NoughtsAndCrosses.required_to_win)

    def legal_action(self, action: __ConnectAction__) -> bool:
        """TODO: Since all Connect games have this feature, perhaps move it up to the __Connect__ class.
        Maybe even just use the Error raise thing that it does."""

        # Check if the action is within the bounds of the board (0, 2) and the space is empty.
        x, y = action
        return 0 <= x <= 2 and 0 <= y <= 2 and self.state.board[action] == __PlayerTurn__.EMPTY

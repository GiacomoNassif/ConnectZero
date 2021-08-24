import numpy as np
from numba import njit


@njit
def encode_board(board: np.ndarray, player):
    result = np.zeros((3, board.shape[0], board.shape[1]), dtype=np.int8)

    for index, element in np.ndenumerate(board):
        result[element][index] = 1
    result[0, :, :] = player
    return result



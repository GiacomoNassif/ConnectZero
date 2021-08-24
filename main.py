from MCTSNode import MCTSNode
import numpy as np
from MCTS import MonteCarloTreeSearch
from ConnectGame import XesAndOes, __ConnectState__, __PlayerTurn__

initial_state = XesAndOes()


# winnable_board = np.array([[1, 0, -1], [1, 0, 0], [0, -1, -1]])
# winnable_state = __ConnectState__(winnable_board, __PlayerTurn__.HERO, 3)
# initial_state = XesAndOes(winnable_state)

root = MCTSNode(initial_state)
mcts = MonteCarloTreeSearch(root)
best_node = mcts.best_action(2000)

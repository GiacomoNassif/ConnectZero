from MCTS import ConnectMCTS, MonteCarloTreeSearch
from Connect import NoughtsAndCrosses

initial_state = NoughtsAndCrosses()


# winnable_board = np.array([[1, 0, -1], [1, 0, 0], [0, -1, -1]])
# winnable_state = __ConnectState__(winnable_board, __PlayerTurn__.HERO)
# initial_state = NoughtsAndCrosses(winnable_state)

root = ConnectMCTS(initial_state)
mcts = MonteCarloTreeSearch(root)
best_node = mcts.best_action(3000)

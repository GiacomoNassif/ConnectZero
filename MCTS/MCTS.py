import time

from MCTS.ConnectMCTS import ConnectMCTS


class MonteCarloTreeSearch:

    def __init__(self, node: ConnectMCTS):
        self.root = node

    def best_action(self, simulations_number: int = None, total_simulation_seconds: float = None):

        if simulations_number is None:
            assert (total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while time.time() < end_time:
                v = self._tree_policy()
                reward = v.simulate()
                v.backpropagate(reward)
        else:
            for _ in range(0, simulations_number):
                v = self._tree_policy()
                reward = v.simulate()
                v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):

        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

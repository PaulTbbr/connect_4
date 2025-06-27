from typing import Optional

import numpy as np

from src.environment import GridGame, TicTacToe


class Node:
    def __init__(
        self,
        game: GridGame,
        args: dict,
        state: np.ndarray,
        parent: Optional["Node"] = None,
        action_taken: int | None = None,
    ) -> None:
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children: list["Node"] = []
        self.expandable_moves = game.get_valid_moves(state)

        self.n_visits = 0
        self.value_sums = 0

    def is_fully_expanded(self) -> bool:
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self) -> "Node":
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child: "Node") -> float:
        # As a parent, we should chose children with lower q values, i.e. that least benefits the opponent
        q = 1 - ((child.value_sums / child.n_visits) + 1) / 2
        return q + self.args["c"] * np.sqrt(np.log(self.n_visits) / child.n_visits)

    def expand(self) -> "Node":
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0

        child_state = self.state.copy()
        child_state = self.game.get_next_state(
            child_state, action, 1
        )  # Each node will act as 1 and see children as opponents
        child_state = self.game.change_perspective(child_state)

        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self) -> float:
        value, is_terminal = self.game.get_value_and_terminated(
            self.state, self.action_taken
        )
        value = self.game.get_opponent_value(value)
        if is_terminal:
            return value

        n_wins = 0
        for _ in range(self.args["n_rollouts"]):
            rollout_player = 1
            rollout_state = self.state.copy()
            while True:
                action = np.random.choice(
                    np.where(self.game.get_valid_moves(rollout_state) == 1)[0]
                )
                rollout_state = self.game.get_next_state(
                    rollout_state, action, rollout_player
                )
                value, is_terminal = self.game.get_value_and_terminated(
                    rollout_state, action
                )
                if is_terminal:
                    n_wins += value * rollout_player
                    break
                rollout_player = self.game.get_opponent(rollout_player)

        return n_wins / self.args["n_rollouts"]

    def backpropagate(self, value: float | int) -> None:
        self.value_sums += value
        self.n_visits += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game: "GridGame", args: dict) -> None:
        self.game = game
        self.args = args

    def search(self, state: np.ndarray):
        # root node
        root = Node(self.game, self.args, state)

        for search in range(self.args["n_searches"]):
            # selection
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                # expansion
                node = node.expand()
                # simulation
                value = node.simulate()

            # backprop
            node.backpropagate(value)

        action_probs = np.zeros(self.game.n_actions)
        for child in root.children:
            action_probs[child.action_taken] = child.n_visits
        action_probs /= np.sum(action_probs)
        return action_probs


if __name__ == "__main__":
    tictactoe = TicTacToe()
    player = 1

    args = {"c": 1.41, "n_searches": 1000, "n_rollouts": 1}

    mcts = MCTS(tictactoe, args)

    state = tictactoe.get_initial_state()

    while True:
        tictactoe.render(state)

        if player == 1:
            valid_moves = tictactoe.get_valid_moves(state)
            print(
                "valid_moves",
                [i for i in range(tictactoe.n_actions) if valid_moves[i] == 1],
            )
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("action not valid")
                continue

        else:
            neutral_state = tictactoe.change_perspective(state)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        state = tictactoe.get_next_state(state, action, player)

        value, is_terminal = tictactoe.get_value_and_terminated(state, action)

        if is_terminal:
            tictactoe.render(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break

        player = tictactoe.get_opponent(player)

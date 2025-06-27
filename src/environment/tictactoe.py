import numpy as np

from src.environment.base_game import GridGame


class TicTacToe(GridGame):
    def __init__(self) -> None:
        super().__init__(n_rows=3, n_cols=3)

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        row = action // self.n_cols
        col = action % self.n_rows
        state[row, col] = player
        return state

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray[np.uint8]:
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state: np.ndarray, action: int) -> bool:
        if action is None:
            # will happen when we get value from root node and action None in MCTS
            return False

        row = action // self.n_cols
        col = action % self.n_rows
        player = state[row, col]

        return (
            np.sum(state[row, :]) == player * self.n_cols
            or np.sum(state[:, col]) == player * self.n_rows
            or np.sum(np.diag(state)) == player * self.n_rows
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.n_rows
        )

    def render(self, state: np.ndarray, show_positions: bool = False) -> None:
        print("\n" + "=" * 13)
        for row in range(self.n_rows):
            print("|", end="")
            for col in range(self.n_cols):
                cell_value = state[row, col]
                if cell_value == 1:
                    symbol = " X "
                elif cell_value == -1:
                    symbol = " O "
                else:
                    if show_positions:
                        position = row * self.n_cols + col
                        symbol = f" {position} "
                    else:
                        symbol = "   "

                print(symbol, end="|")
            print()
            if row < self.n_rows - 1:
                print("|---|---|---|")
        print("=" * 13 + "\n")

    def render_with_positions(self, state: np.ndarray) -> None:
        self.render(state, show_positions=True)


if __name__ == "__main__":
    tictactoe = TicTacToe()
    player = 1

    state = tictactoe.get_initial_state()

    while True:
        tictactoe.render(state)
        valid_moves = tictactoe.get_valid_moves(state)
        print(
            "valid_moves",
            [i for i in range(tictactoe.n_actions) if valid_moves[i] == 1],
        )
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue

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

import numpy as np

from src.environment.base_game import GridGame


class ConnectFour(GridGame):
    def __init__(self) -> None:
        super().__init__(n_rows=6, n_cols=7, action_size=7)
        self.in_a_row = 4

    def get_next_state(
        self, state: np.ndarray, action: int, player: int
    ) -> np.ndarray:
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row, column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.n_rows
                    or c < 0 
                    or c >= self.n_cols
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
    
    def render(self, state: np.ndarray, show_columns: bool = True) -> None:
        print()
        
        if show_columns:
            print("  ", end="")
            for col in range(self.n_cols):
                print(f"  {col} ", end=" ")
            print()
        
        print("  " + "┌" + "────┬" * (self.n_cols - 1) + "────┐")
        for row in range(self.n_rows):
            print("  │", end="")
            for col in range(self.n_cols):
                cell_value = state[row, col]
                if cell_value == 1:
                    symbol = " 🔴"
                elif cell_value == -1:
                    symbol = " 🟡" 
                else:
                    symbol = "   "
                print(symbol, end=" │")
            print()
            if row < self.n_rows - 1:
                print("  ├" + "────┼" * (self.n_cols - 1) + "────┤")
        print("  └" + "────┴" * (self.n_cols - 1) + "────┘")
        print()


if __name__ == "__main__":
    game = ConnectFour()
    player = 1

    state = game.get_initial_state()

    while True:
        print(state)
        
        valid_moves = game.get_valid_moves(state)
        print("valid_moves", [i for i in range(game.n_actions) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue
            
        state = game.get_next_state(state, action, player)
        
        value, is_terminal = game.get_value_and_terminated(state, action)
        
        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break
            
        player = game.get_opponent(player)
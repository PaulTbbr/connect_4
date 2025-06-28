from abc import ABC, abstractmethod
import numpy as np

class GridGame(ABC):
    def __init__(self, n_rows, n_cols, action_size=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_actions = action_size or (n_rows * n_cols)
        
    def __repr__(self):
        return self.__class__.__name__

    def get_initial_state(self) -> np.ndarray:
        """Return an empty board state."""
        return np.zeros((self.n_rows, self.n_cols))

    @abstractmethod
    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        """Apply action for player and return new state."""
        pass

    @abstractmethod
    def get_valid_moves(self,  state: np.ndarray) -> np.ndarray[np.uint8]:
        """Return a binary vector of valid actions."""
        pass

    @abstractmethod
    def check_win(self, state: np.ndarray, action: int) -> bool:
        """Return True if the last action caused a win."""
        pass

    def get_value_and_terminated(self, state: np.ndarray, action: int) -> tuple[int, bool]:
        """Return (value, terminated) after action. Value is +1/-1 for win/loss, 0 otherwise."""
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player: int) -> int:
        """Return the opposing player marker."""
        return -player

    def get_opponent_value(self, value: int) -> int:
        """Invert a value from one player's perspective to the other's."""
        return -value

    def change_perspective(self, state: np.ndarray) -> np.ndarray:
        """Flip perspective so current player is always +1."""
        return -state

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        """One-hot encode state into planes for -1, 0, +1."""
        return np.stack((state == -1, state == 0, state == 1)).astype(np.float32)


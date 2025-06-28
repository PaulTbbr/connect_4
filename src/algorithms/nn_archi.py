import torch.nn as nn
import torch.nn.functional as F

from src.environment import GridGame


class ResNet(nn.Module):
    def __init__(self, game: GridGame, n_res_blocks: int, n_hidden: int, device):
        super().__init__()
        self.device = device
        self.start_block = nn.Sequential(
            nn.Conv2d(3, n_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_hidden),
            nn.ReLU(),
        )

        self.backbone = nn.ModuleList([ResBlock(n_hidden) for _ in range(n_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(n_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.n_rows * game.n_cols, game.n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(n_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.n_rows * game.n_cols, 1),
            nn.Tanh(),
        )

        self.to(device)

    def forward(self, x):
        x = self.start_block(x)
        for res_block in self.backbone:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, n_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_hidden)
        self.conv2 = nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_hidden)

    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        out = F.relu(out)
        return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    from src.environment import TicTacToe

    tictactoe = TicTacToe()

    state = tictactoe.get_initial_state()
    state = tictactoe.get_next_state(state, 2, 1)
    state = tictactoe.get_next_state(state, 7, -1)

    print(state)

    encoded_state = tictactoe.get_encoded_state(state)

    print(encoded_state)

    tensor_state = torch.tensor(encoded_state).unsqueeze(0)

    model = ResNet(tictactoe, 4, 64)
    model.load_state_dict(torch.load("checkpoints/model_TicTacToe_2.pt"))
    model.eval()

    policy, value = model(tensor_state)
    value = value.item()
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    print(value, policy)

    plt.bar(range(tictactoe.n_actions), policy)
    plt.savefig("outputs/test_bis_torch.png")

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from src.algorithms.nn_archi import ResNet
from src.environment import GridGame, TicTacToe


class AlphaNode:
    def __init__(
        self,
        game: GridGame,
        args: dict,
        state: np.ndarray,
        parent: Optional["AlphaNode"] = None,
        action_taken: int | None = None,
        prior: float = 0.0,
        n_visits: int = 0,
    ) -> None:
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children: list["AlphaNode"] = []

        self.n_visits = n_visits
        self.value_sums = 0

    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0

    def select(self) -> "AlphaNode":
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child: "AlphaNode") -> float:
        if not child.n_visits:
            q = 0
        else:
            # As a parent, we should chose children with lower q values, i.e. that least benefits the opponent
            q = 1 - ((child.value_sums / child.n_visits) + 1) / 2
        return q + self.args["c"] * child.prior * np.sqrt(
            self.n_visits / (child.n_visits + 1)
        )

    def expand(self, policy: np.ndarray) -> None:
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(
                    child_state, action, 1
                )  # Each node will act as 1 and see children as opponents
                child_state = self.game.change_perspective(child_state)

                child = AlphaNode(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value: float | int) -> None:
        self.value_sums += value
        self.n_visits += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class AlphaMCTS:
    def __init__(self, game: "GridGame", args: dict, model: ResNet) -> None:
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state: np.ndarray):
        # root node
        root = AlphaNode(self.game, self.args, state, n_visits=1)

        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_state(state), device=self.model.device
            ).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.n_actions)
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

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
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(node.state),
                        device=self.model.device,
                    ).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                # expansion
                node.expand(policy)

            # backprop
            node.backpropagate(value)

        action_probs = np.zeros(self.game.n_actions)
        for child in root.children:
            action_probs[child.action_taken] = child.n_visits
        action_probs /= np.sum(action_probs)
        return action_probs


class AlphaZero:
    def __init__(self, model: ResNet, optimizer, game: GridGame, args: dict) -> None:
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        self.mcts = AlphaMCTS(game, args, model)

    def self_play(self) -> list[tuple]:
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            if self.args["temperature"] == 0:
                action = np.argmax(action_probs)
            else:
                temp_action_probs = action_probs ** (
                    1 / (self.args["temperature"] + 1e-7)
                )
                temp_action_probs /= np.sum(temp_action_probs)
                action = np.random.choice(self.game.n_actions, p=temp_action_probs)

            state = self.game.get_next_state(neutral_state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                return_mem = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = (
                        value
                        if hist_player == player
                        else self.game.get_opponent_value(value)
                    )
                    return_mem.append(
                        (
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome,
                        )
                    )
                return return_mem

            player = self.game.get_opponent(player)

    def train(self, memory: list[tuple]) -> None:
        random.shuffle(memory)

        for batch_id in range(0, len(memory), self.args["batch_size"]):
            sample = memory[
                batch_id : min(len(memory) - 1, batch_id + self.args["batch_size"])
            ]

            state, policy_targets, value_targets = zip(*sample)

            state = torch.tensor(
                np.array(state), dtype=torch.float32, device=self.model.device
            )
            policy_targets = torch.tensor(
                np.array(policy_targets), dtype=torch.float32, device=self.model.device
            )
            value_targets = torch.tensor(
                np.array(value_targets), dtype=torch.float32, device=self.model.device
            )

            out_policy, out_value = self.model(state)
            out_value = out_value.squeeze(-1)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self) -> None:
        for iteration in range(self.args["n_iterations"]):
            print(f"Iteration {iteration + 1}/{self.args['n_iterations']}...")
            memory = []

            print("Self-play...")
            self.model.eval()
            for self_play_iteration in trange(self.args["n_self_play_iterations"]):
                memory += self.self_play()

            print("Training...")
            self.model.train()
            for epoch in trange(self.args["n_epochs"]):
                self.train(memory)

            checkpoints_dir = self.args["checkpoints_dir"]
            os.makedirs(checkpoints_dir, exist_ok=True)
            torch.save(
                self.model.state_dict(),
                f"{checkpoints_dir}/model_{self.game}_{iteration}.pt",
            )
            torch.save(
                self.optimizer.state_dict(),
                f"{checkpoints_dir}/optimizer_{self.game}_{iteration}.pt",
            )


if __name__ == "__main__":
    # tictactoe = TicTacToe()
    # player = 1

    # args = {
    #     'c': 2,
    #     'n_searches': 1000
    # }

    # model = ResNet(tictactoe, 4, 64)
    # model.eval()

    # mcts = AlphaMCTS(tictactoe, args, model)

    # state = tictactoe.get_initial_state()

    # while True:
    #     print(state)

    #     if player == 1:
    #         valid_moves = tictactoe.get_valid_moves(state)
    #         print("valid_moves", [i for i in range(tictactoe.n_actions) if valid_moves[i] == 1])
    #         action = int(input(f"{player}:"))

    #         if valid_moves[action] == 0:
    #             print("action not valid")
    #             continue

    #     else:
    #         neutral_state = tictactoe.change_perspective(state)
    #         mcts_probs = mcts.search(neutral_state)
    #         action = np.argmax(mcts_probs)

    #     state = tictactoe.get_next_state(state, action, player)

    #     value, is_terminal = tictactoe.get_value_and_terminated(state, action)

    #     if is_terminal:
    #         print(state)
    #         if value == 1:
    #             print(player, "won")
    #         else:
    #             print("draw")
    #         break

    #     player = tictactoe.get_opponent(player)

    tictactoe = TicTacToe()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(tictactoe, 4, 64, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        "c": 2,
        "n_searches": 60,
        "n_iterations": 3,
        "n_self_play_iterations": 50,
        "n_epochs": 4,
        "batch_size": 64,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
    }

    alpha_zero = AlphaZero(model, optimizer, tictactoe, args)
    alpha_zero.learn()

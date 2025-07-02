import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from src.algorithms.nn_archi import ResNet
from src.environment import GridGame

from src.algorithms.alpha_zero import AlphaNode

class AlphaMCTSParallel:
    def __init__(self, game: "GridGame", args: dict, model: ResNet) -> None:
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states: np.ndarray, sp_games: np.ndarray):
        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_state(states), device=self.model.device
            )
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.n_actions, size=policy.shape[0])

        for i, spg in enumerate(sp_games):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = AlphaNode(self.game, self.args, states[i], n_visits=1)
            spg.root.expand(spg_policy)

        for search in range(self.args["n_searches"]):
            # selection
            for spg in sp_games:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(
                    node.state, node.action_taken
                )
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandable_sp_games = [idx for idx, sp_game in enumerate(sp_games) if sp_game.node is not None]

            if len(expandable_sp_games) > 0:
                states = np.stack([sp_games[idx].node.state for idx in expandable_sp_games])
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(states),
                        device=self.model.device,
                    )
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, idx in enumerate(expandable_sp_games):
                node = sp_games[idx].node
                spg_policy, spg_value = policy[i], value[i]

                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)




class AlphaZeroParallel:
    def __init__(self, model: ResNet, optimizer, game: GridGame, args: dict) -> None:
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        self.mcts = AlphaMCTSParallel(game, args, model)

    def self_play(self) -> list[tuple]:
        rerturn_memory = []
        player = 1
        sp_games = [SPG(self.game) for _ in range(self.args["n_parallel_games"])]

        while len(sp_games) > 0:
            states = np.stack([spg.state for spg in sp_games])
            
            neutral_states = self.game.change_perspective(states)
            self.mcts.search(neutral_states, sp_games)
            
            for i in range(len(sp_games))[::-1]:
                spg = sp_games[i]
                action_probs = np.zeros(self.game.n_actions)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.n_visits
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                if self.args["temperature"] == 0:
                    action = np.argmax(action_probs)
                else:
                    temp_action_probs = action_probs ** (
                        1 / (self.args["temperature"] + 1e-7)
                    )
                    temp_action_probs /= np.sum(temp_action_probs)
                    action = np.random.choice(self.game.n_actions, p=temp_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = (
                            value
                            if hist_player == player
                            else self.game.get_opponent_value(value)
                        )
                        rerturn_memory.append(
                            (
                                self.game.get_encoded_state(hist_neutral_state),
                                hist_action_probs,
                                hist_outcome,
                            )
                        )
                    del sp_games[i]

            player = self.game.get_opponent(player)

        return rerturn_memory
    
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
            for _ in trange(self.args["n_self_play_iterations"] // self.args["n_parallel_games"]):
                memory += self.self_play()

            print("Training...")
            self.model.train()
            for _ in trange(self.args["n_epochs"]):
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

class SPG:
    def __init__(self, game: GridGame) -> None:
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

if __name__ == "__main__":
    from src.environment import ConnectFour
    connect_four = ConnectFour()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(connect_four, 4, 64, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        "c": 2,
        "n_searches": 600,
        "n_iterations": 8,
        "n_self_play_iterations": 500,
        "n_parallel_games": 100,
        "n_epochs": 4,
        "batch_size": 128,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
    }

    alpha_zero = AlphaZeroParallel(model, optimizer, connect_four, args)
    alpha_zero.learn()

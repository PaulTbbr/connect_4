import argparse

import numpy as np
import torch

from src.algorithms import MCTS, AlphaMCTS, AlphaZero, ResNet
from src.environment import ConnectFour, TicTacToe


def get_game(name: str):
    if name == "tictactoe":
        return TicTacToe()
    elif name == "connect_four":
        return ConnectFour()
    else:
        raise ValueError(f"Unknown game: {name}")


def train(args):
    game = get_game(args.game)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, args.n_blocks, args.n_filters, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    az = AlphaZero(model, optimizer, game, vars(args))
    az.learn()


def play(args):
    game = get_game(args.game)
    state = game.get_initial_state()
    player = 1

    # load model if alpha_mcts
    if args.algo in ["alpha", "alpha_mcts"]:
        if not args.model_path:
            raise ValueError("`--model_path` is required for alpha_mcts")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNet(game, args.n_blocks, args.n_filters, device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        searcher = AlphaMCTS(game, vars(args), model)
    else:
        searcher = MCTS(game, vars(args))

    while True:
        game.render_with_positions(state) if args.show_positions else game.render(state)
        if player == 1 and args.vs_ai:
            valid = game.get_valid_moves(state)
            print("valid moves", np.where(valid)[0].tolist())
            action = int(input(f"Player {player} move: "))
            if valid[action] == 0:
                print("Invalid move")
                continue
        else:
            # AI move
            neutral = game.change_perspective(state)
            probs = searcher.search(neutral)
            action = int(np.argmax(probs))
            print(f"AI plays: {action}")

        state = game.get_next_state(state, action, player)
        value, done = game.get_value_and_terminated(state, action)
        if done:
            game.render(state)
            if value == 1:
                print(f"Player {player} wins!")
            else:
                print("Draw")
            break
        player = game.get_opponent(player)


def solo(args):
    game = get_game(args.game)
    state = game.get_initial_state()
    player = 1
    while True:
        game.render(state)
        valid = game.get_valid_moves(state)
        print("valid moves", np.where(valid)[0].tolist())
        action = int(input(f"Player {player} move: "))
        if valid[action] == 0:
            print("Invalid move")
            continue
        state = game.get_next_state(state, action, player)
        value, done = game.get_value_and_terminated(state, action)
        if done:
            game.render(state)
            if value == 1:
                print(f"Player {player} wins!")
            else:
                print("Draw")
            break
        player = game.get_opponent(player)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play or train board games with MCTS or AlphaZero"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--game", choices=["tictactoe", "connect_four"], required=True)
    p_train.add_argument("--n_blocks", type=int, default=4)
    p_train.add_argument("--n_filters", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--c", type=float, default=2)
    p_train.add_argument("--n_searches", type=int, default=60)
    p_train.add_argument("--n_iterations", type=int, default=3)
    p_train.add_argument("--n_self_play_iterations", type=int, default=50)
    p_train.add_argument("--n_epochs", type=int, default=4)
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--temperature", type=float, default=1.25)
    p_train.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    p_train.add_argument("--dirichlet_alpha", type=float, default=0.3)
    p_train.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p_train.set_defaults(func=train)

    # play
    p_play = sub.add_parser("play")
    p_play.add_argument("--game", choices=["tictactoe", "connect_four"], required=True)
    p_play.add_argument("--algo", choices=["mcts", "alpha_mcts"], default="mcts")
    p_play.add_argument("--vs_ai", action="store_true", help="Human first vs AI")
    p_play.add_argument("--model_path", type=str, default="")
    p_play.add_argument("--n_blocks", type=int, default=4)
    p_play.add_argument("--n_filters", type=int, default=64)
    p_play.add_argument("--show_positions", action="store_true")
    p_play.add_argument("--n_searches", type=int, default=1000)
    p_play.add_argument("--n_rollouts", type=int, default=1)
    p_play.add_argument("--c", type=float, default=1.41)
    p_play.add_argument("--temperature", type=float, default=1.25)
    p_play.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    p_play.add_argument("--dirichlet_alpha", type=float, default=0.3)
    p_play.set_defaults(func=play)

    # solo
    p_solo = sub.add_parser("solo")
    p_solo.add_argument("--game", choices=["tictactoe", "connect_four"], required=True)
    p_solo.set_defaults(func=solo)

    args = parser.parse_args()
    args.func(args)

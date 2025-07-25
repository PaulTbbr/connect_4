import glob
import os

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from src.algorithms import MCTS, AlphaMCTS, ResNet
from src.environment import ConnectFour

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(APP_ROOT, os.pardir))

DEFAULT_MCTS_PARAMS = {
    "c": 1.41,
    "n_searches": 1000,
    "n_rollouts": 1,
    "dirichlet_epsilon": 0,
    "temperature": 0.0,
    "dirichlet_alpha": 0.03,
}

DEFAULT_CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")


def _get_latest_checkpoint(game_name: str) -> str:
    pattern = os.path.join(DEFAULT_CHECKPOINT_DIR, f"model_{game_name}_6.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint found in {DEFAULT_CHECKPOINT_DIR}")
    return max(files, key=os.path.getmtime)


def make_searcher(algo, model_path=None, params=None):
    game = ConnectFour()
    cfg = params.copy() if params else {}

    if algo in ["alpha", "alpha_mcts"]:
        path = model_path or _get_latest_checkpoint("ConnectFour")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ResNet(
            game,
            n_res_blocks=cfg.get("n_blocks", 9),
            n_hidden=cfg.get("n_filters", 128),
            device=device,
        )

        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        for k, v in DEFAULT_MCTS_PARAMS.items():
            cfg.setdefault(k, v)
        cfg["c"] = 2
        return AlphaMCTS(game, cfg, model)

    cfg = cfg or {}
    for k, v in DEFAULT_MCTS_PARAMS.items():
        cfg.setdefault(k, v)
    return MCTS(game, cfg)


# global state
state = ConnectFour().get_initial_state()
player = 1
searcher = None

game_mode = "human"
game = ConnectFour()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/setup", methods=["POST"])
def setup():
    global state, player, searcher, game_mode
    cfg = request.json or {}
    state = game.get_initial_state()
    player = 1
    game_mode = cfg.get("algo", "human")
    if game_mode != "human":
        try:
            searcher = make_searcher(game_mode, cfg.get("model_path"), cfg)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        searcher = None
    return jsonify({"status": "ok"})


@app.route("/move", methods=["POST"])
def move():
    global state, player
    data = request.json or {}
    col = data.get("column")
    if data.get("human"):
        valid = game.get_valid_moves(state)
        if valid[col] == 0:
            return jsonify({"error": "Invalid"})
        action = col
        state = game.get_next_state(state, action, player)
    else:
        if not searcher:
            return jsonify({"error": "AI not initialized"}), 400
        neutral = game.change_perspective(state)
        probs = searcher.search(neutral)
        action = int(np.argmax(probs))
        state = game.get_next_state(state, action, player)

    value, done = game.get_value_and_terminated(state, action)
    win_positions = []
    winner = None
    if done and value == 1:
        winner = int(player)
        win_positions = game.get_win_positions(state, action)
        if win_positions:
            win_positions = [[int(r), int(c)] for r, c in win_positions]

    board = state.tolist()
    player = game.get_opponent(player)

    return jsonify(
        {
            "board": board,
            "player": int(player),
            "done": done,
            "winner": winner,
            "win_positions": win_positions,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

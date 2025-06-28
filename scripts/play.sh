# Usage:
#   Human first vs MCTS:   ./play.sh tictactoe mcts --vs_ai
#   Human first vs Alpha:  ./play.sh connect_four alpha_mcts --model_path checkpoints/model_ConnectFour_0.pt --vs_ai
#   AI vs AI:              ./play.sh tictactoe alpha_mcts --model_path checkpoints/model_TicTacToe_0.pt

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate connect-4-venv

GAME=${1:-tictactoe}
ALGO=${2:-mcts}
MODEL_PATH=${3:-}
shift 3

CMD_ARGS=(
  --game "${GAME}"
  --algo "${ALGO}"
)

if [[ -n "${MODEL_PATH}" ]]; then
  CMD_ARGS+=(--model_path "${MODEL_PATH}")
fi

# --show_positions for numbered boards
CMD_ARGS+=( "$@" )

python -m src.main play "${CMD_ARGS[@]}"

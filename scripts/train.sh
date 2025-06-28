# Usage: ./train.sh <game> [optional args...]
# e.g. ./train.sh tictactoe --n_blocks 6 --n_filters 128 --n_iterations 5

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate connect-4-venv


GAME=${1:-tictactoe}
shift

python -m src.main train \
  --game "${GAME}" \
  --n_blocks 4 \
  --n_filters 64 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --c 2 \
  --n_searches 60 \
  --n_iterations 3 \
  --n_self_play_iterations 50 \
  --n_epochs 4 \
  --batch_size 64 \
  --temperature 1.25 \
  --dirichlet_epsilon 0.25 \
  --dirichlet_alpha 0.3 \
  --checkpoints_dir checkpoints/test_cli \
  "$@"


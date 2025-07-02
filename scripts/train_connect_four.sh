eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate connect-4-venv

GAME=${1:-connect_four}
shift

python -m src.main train \
  --game "${GAME}" \
  --n_blocks 9 \
  --n_filters 128 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --c 2 \
  --n_searches 600 \
  --n_iterations 8 \
  --n_self_play_iterations 500 \
  --n_epochs 4 \
  --batch_size 128 \
  --temperature 1.25 \
  --dirichlet_epsilon 0.25 \
  --dirichlet_alpha 0.3 \
  --checkpoints_dir checkpoints/ \
  "$@"


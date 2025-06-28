eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate connect-4-venv

GAME=${1:-tictactoe}

python -m src.main solo --game "${GAME}"
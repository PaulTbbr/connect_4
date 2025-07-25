let currentPlayer = 1;
let gameMode = 'human';
const boardDiv = document.getElementById('board');
const messageDiv = document.getElementById('message');
const winnerMessageDiv = document.getElementById('winner-message');
const statusDiv = document.getElementById('status');
const columnIndicatorsDiv = document.getElementById('columnIndicators');
let gameOver = false;

function createColumnIndicators() {
  columnIndicatorsDiv.innerHTML = '';
  for (let col = 0; col < 7; col++) {
    const indicator = document.createElement('div');
    indicator.className = 'column-indicator';
    indicator.onclick = () => handleColumnClick(col);
    columnIndicatorsDiv.appendChild(indicator);
  }
}

function updateGameStatus() {
  if (gameOver) return;
  
  // Convert -1 to 2 for display purposes
  const displayPlayer = currentPlayer === -1 ? 2 : currentPlayer;
  
  statusDiv.innerHTML = `
    <span class="player-indicator player-${displayPlayer}"></span>
    Player ${displayPlayer}'s Turn
  `;
}

function render(board) {
  boardDiv.innerHTML = '';
  boardDiv.style.display = 'grid';

  board.forEach((row, rowIndex) => {
    row.forEach((cell, colIndex) => {
      const div = document.createElement('div');
      div.classList.add('cell');
      div.dataset.row = rowIndex;
      div.dataset.col = colIndex;
      if (cell === 1) div.classList.add('red');
      if (cell === -1) div.classList.add('yellow');
      div.addEventListener('click', () => handleColumnClick(colIndex));
      boardDiv.appendChild(div);
    });
  });
}

function highlightWin(positions) {
  // boardDiv.children is a flat list of 6*7 cells, row‑major
  positions.forEach(([r, c]) => {
    const idx = r * 7 + c;
    boardDiv.children[idx].classList.add('win');
  });
}

async function startGame() {
  const mode = document.getElementById('mode').value;
  const modelPath = document.getElementById('model_path').value;
  gameMode = mode;
  gameOver = false;
  messageDiv.style.display = 'none';
  winnerMessageDiv.textContent = '';
  
  createColumnIndicators();
  
  const resp = await fetch('/setup', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algo: mode, model_path: modelPath })
  });
  const data = await resp.json();
  if (!resp.ok) {
    return alert(`Setup error: ${data.error}`);
  }
  currentPlayer = 1;
  updateGameStatus();
  const empty = Array.from({length:6}, () => Array(7).fill(0));
  render(empty);
}

async function makeMove(col, human) {
  const body = { human };
  if (col !== null && col !== undefined) body.column = col;

  const resp = await fetch('/move', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  const data = await resp.json();
  if (data.error) return alert(data.error);
  render(data.board);
  if (data.done) {
    // show winner message under the board
    if (data.winner) {
      // Convert winner value for display (backend sends -1 for player 2)
      const displayWinner = data.winner === -1 ? 2 : data.winner;
      winnerMessageDiv.textContent = `Player ${displayWinner} wins!`;
    } else {
      winnerMessageDiv.textContent = 'Draw!';
    }

    // highlight the four winning cells
    if (data.win_positions) {
      highlightWin(data.win_positions);
    }

    // freeze further play
    gameOver = true;
    return;
  }
  currentPlayer = data.player;
  updateGameStatus();
}

function handleColumnClick(col) {
  if (gameOver) return;          // stop clicks once game is over
  // ignore clicks if it's AI's turn
  if (gameMode !== 'human' && currentPlayer !== 1) return;

  // human move
  makeMove(col, true).then(() => {
    // after human move, if AI mode, do AI move
    if (gameMode !== 'human' && !gameOver) {
      makeMove(null, false);
    }
  });
}

function handleClick(event) {
  const col = parseInt(event.currentTarget.dataset.col);
  handleColumnClick(col);
}


document.getElementById('start').onclick = startGame;
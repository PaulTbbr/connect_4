let currentPlayer = 1;
let gameMode = 'human';
const boardDiv = document.getElementById('board');

function render(board) {
  boardDiv.innerHTML = '';
  boardDiv.style.gridTemplateRows = `repeat(${board.length}, 80px)`;
  boardDiv.style.gridTemplateColumns = `repeat(${board[0].length}, 80px)`;
  boardDiv.style.display = 'grid';
  boardDiv.style.gap = '5px';

  board.forEach((row, rowIndex) => {
    row.forEach((cell, colIndex) => {
      const div = document.createElement('div');
      div.classList.add('cell');
      if (cell === 1) div.classList.add('red');
      if (cell === -1) div.classList.add('yellow');
      div.addEventListener('click', handleClick);
      boardDiv.appendChild(div);
    });
  });
}

async function startGame() {
  const mode = document.getElementById('mode').value;
  const modelPath = document.getElementById('model_path').value;
  gameMode = mode;
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
  if (data.done) return alert(data.winner ? `Player ${data.winner} wins!` : 'Draw');
  currentPlayer = data.player;
}

function handleClick(event) {
  // ignore clicks if it's AI's turn
  if (gameMode !== 'human' && currentPlayer !== 1) return;

  const cells = Array.from(boardDiv.children);
  const idx = cells.indexOf(event.currentTarget);
  const col = idx % 7;

  // human move
  makeMove(col, true).then(() => {
    // after human move, if AI mode, do AI move
    if (gameMode !== 'human') {
      makeMove(null, false);
    }
  });
}

document.getElementById('start').onclick = startGame;
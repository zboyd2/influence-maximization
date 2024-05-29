let graphNodes = [];
let adjList = [];
let opinions = [];
let controls = [];
let config = [];
let playerTurn = 0;
let turnCount = 0;
let numTurns = 3;
let gameRunning = true;

function updateNodeCountDisplay(value) {
    document.getElementById('node-count-display').textContent = value;
}

async function startGame() {
    const graphType = document.getElementById('graph-type').value;
    const player1 = document.getElementById('player1').value;
    const player2 = document.getElementById('player2').value;
    const nodeCount = parseInt(document.getElementById('node-count').value);
    numTurns = parseInt(document.getElementById('turn-count').value);

    document.getElementById('startup-screen').style.display = 'none';
    document.getElementById('gameplay').style.display = 'flex';

    const player1Text = document.getElementById('player1-indicator');
    const player1Display = player1.charAt(0).toUpperCase() + player1.slice(1);
    const player2Text = document.getElementById('player2-indicator');
    const player2Display = player2.charAt(0).toUpperCase() + player2.slice(1);

    player1Text.textContent = `Player 1: ${player1Display}`;
    player2Text.textContent = `Player 2: ${player2Display}`;

    try {
        const data = await fetchGraphData(graphType, nodeCount);
        plotGraph(data.nodes, data.edges);
        initializeGraphState(data.nodes.length);
        mainGame();

        if (player1 !== 'human') {
            await makeBotMove(player1);
        }
    } catch (error) {
        alert('Failed to fetch graph data!');
    }
}

async function fetchGraphData(graphType, nodeCount) {
    let url;
    switch (graphType) {
        case 'distribution':
            url = `/api/distribution?nodes=${nodeCount}`;
            break;
        case 'tree':
            url = `/api/tree?nodes=${nodeCount}`;
            break;
        case 'ladder':
            url = `/api/ladder?nodes=${nodeCount}`;
            break;
        case 'square':
            url = `/api/square?nodes=${nodeCount}`;
            break;
        case 'hexagon':
            url = `/api/hexagon?nodes=${nodeCount}`;
            break;
        case 'triangle':
            url = `/api/triangle?nodes=${nodeCount}`;
            break;
        case 'cycle':
            url = `/api/cycle?nodes=${nodeCount}`;
            break;
        case 'random_proximity':
            url = `/api/random_proximity?nodes=${nodeCount}`;
            break;
        default:
            return Promise.reject('Invalid graph type');
    }

    const response = await fetch(url);
    if (!response.ok) {
        alert('Network response was invalid');
    }
    return await response.json();
}

function initializeGraphState(numNodes) {
    opinions = Array(numNodes).fill(0);
    controls = Array(numNodes).fill(null);
    playerTurn = 0;
    turnCount = 0;
    gameRunning = true;
    updateTurnNotification();
}

function mainGame() {
    if (!gameRunning) return;

    const canvas = document.getElementById('game-canvas');
    const player1 = document.getElementById('player1').value;
    const player2 = document.getElementById('player2').value;
    if (player1 === 'human' || player2 === 'human') {
        canvas.addEventListener('click', handleCanvasClick);
    }
}

function handleCanvasClick(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    let nodeClicked = false;

    graphNodes.forEach(node => {
        if (Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2) <= node.radius && controls[node.id] === null) {
            controls[node.id] = playerTurn;
            config.push(node.id);
            updateOpinions();
            updateCanvas();
            updateScoreBar();
            nodeClicked = true;
        }
    });

    if (nodeClicked) {
        updateTurn();
    }

    // alert(opinions.toString());
}

function updateOpinions() {
    let notPrecise = true;
    
    while (notPrecise) {
        let newOpinions = [];

        for (let i = 0; i < graphNodes.length; i++) {
            const neighborOpinions = adjList[i].map(j => opinions[j]);
            if (controls[i] !== null) {
                newOpinions[i] = controls[i] * 2 - 1;
            } else if (neighborOpinions.length > 0) {
                newOpinions[i] = neighborOpinions.reduce((sum, val) => sum + val, 0) / neighborOpinions.length;
            } else {
                newOpinions[i] = opinions[i];
            }
        }
        
        let opinionsPrecise = true;
        for (let i = 0; i < opinions.length; i++) {
            if (Math.abs(opinions[i] - newOpinions[i]) > 1e-6) {
                opinionsPrecise = false;
                break;
            }
        }

        if (opinionsPrecise) {
            notPrecise = false;
        }

        opinions = newOpinions.slice();
    }
    // alert(opinions.toString());
}

function getGraphLaplacian() {
    const numNodes = adjList.length;

    let degreeMatrix = Array.from({ length: numNodes }, () => 0);
    let laplacianMatrix = Array.from({ length: numNodes }, () => Array(numNodes).fill(0));

    // Fill degree matrix and adjacency matrix
    for (let i = 0; i < numNodes; i++) {
        degreeMatrix[i] = adjList[i].length;
        adjList[i].forEach(j => {
            laplacianMatrix[i][j] = -1;
        });
    }

    // Fill the diagonal of the Laplacian matrix with the degrees
    for (let i = 0; i < numNodes; i++) {
        laplacianMatrix[i][i] = degreeMatrix[i];
    }

    return laplacianMatrix;
}

async function getBotMove(difficulty) {
    const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');
    const laplacian = getGraphLaplacian();
    const response = await fetch('/api/bot_move/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({ laplacian, config, difficulty, numTurns })
    });

    if (!response.ok) {
        throw new Error('Failed to fetch bot move.');
    }

    const data = await response.json();
    return data.move;
}

async function makeBotMove(difficulty) {
    try {
        const move = await getBotMove(difficulty);
        if (difficulty === 'easy' || difficulty === 'medium') {
            await sleep(500);
        }

        controls[move] = playerTurn;
        config.push(move);
        updateOpinions();
        updateCanvas();
        updateScoreBar();
        updateTurn();
    } catch (error) {
        console.error('Error fetching bot move:', error);
        alert('Failed to get bot move.');
    }
}

async function updateTurn() {
    if (turnCount < numTurns * 2 - 1) {
        playerTurn = (playerTurn === 0) ? 1 : 0;
        turnCount++;
        updateTurnNotification();

        const player1 = document.getElementById('player1').value;
        const player2 = document.getElementById('player2').value;

        if ((playerTurn === 0 && player1 !== 'human') || (playerTurn === 1 && player2 !== 'human')) {
            const difficulty = (playerTurn === 0) ? player1 : player2;
            await makeBotMove(difficulty);
        }
    } else {
        determineWinner();
        gameRunning = false;
    }
}

function updateTurnNotification() {
    const turnBox = document.getElementById('turn-notification');
    const playerNum = (playerTurn === 0) ? 1 : 2;
    turnBox.textContent = `Player ${playerNum}'s Turn`;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function determineWinner() {
    const averageOpinion = opinions.reduce((sum, val) => sum + val, 0) / opinions.length;
    let winner;
    if (averageOpinion < 0) {
        winner = "Player 1 Wins!";
    } else if (averageOpinion > 0) {
        winner = "Player 2 Wins!";
    } else {
        winner = "Draw";
    }

    await sleep(1000);
    alert(`${winner}`);
}

function plotGraph(nodes, edges) {
    graphNodes = nodes.map((coords, index) => ({
        x: coords[0],
        y: coords[1],
        radius: 12,
        id: index
    }));
    adjList = Array(nodes.length).fill().map(() => []);
    edges.forEach(edge => {
        adjList[edge[0]].push(edge[1]);
        adjList[edge[1]].push(edge[0]);
    });
    edges.forEach(edge => plotEdge(edge[0], edge[1]));
    graphNodes.forEach(node => plotNode(node.x, node.y, 0, null));
}

function plotNode(x, y, opinion, control) {
    const canvas = document.getElementById('game-canvas');
    const ctx = canvas.getContext('2d');
    const radius = 12;

    // Converts opinion from -1 to 1 to rgb value between blue and red
    const shiftedOpinion = (opinion + 1) / 2;
    ctx.fillStyle = `rgb(${255 * shiftedOpinion}, 0, ${255 * (1 - shiftedOpinion)})`;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();

    if (control !== null) {
        const controlColor = (control === 1) ? 'white' : 'black';
        drawControlMark(ctx, x, y, radius / 4, controlColor);
    }
}

function drawControlMark(ctx, x, y, size, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, size, 0, 2 * Math.PI);
    ctx.fill();
}

function plotEdge(nodeid1, nodeid2) {
    const canvas = document.getElementById('game-canvas');
    const ctx = canvas.getContext('2d');

    const node1 = graphNodes.find(node => node.id === nodeid1);
    const node2 = graphNodes.find(node => node.id === nodeid2);

    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(node1.x, node1.y);
    ctx.lineTo(node2.x, node2.y);
    ctx.stroke();
}

function updateCanvas() {
    const canvas = document.getElementById('game-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    adjList.forEach((neighbors, index) => {
        neighbors.forEach(neighbor => plotEdge(index, neighbor));
    });
    for (let i = 0; i < graphNodes.length; i++) {
        const node = graphNodes[i];
        plotNode(node.x, node.y, opinions[i], controls[i]);
    }
    // Add highlighted node drawing from askCoach()
}

function askCoach() {

}

async function resetGame() {
    const graphType = document.getElementById('graph-type').value;
    const nodeCount = parseInt(document.getElementById('node-count').value);

    graphNodes = [];
    adjList = [];
    opinions = [];
    controls = [];
    config = [];
    playerTurn = 0;
    turnCount = 0;
    gameRunning = true;

    const canvas = document.getElementById('game-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
        const data = await fetchGraphData(graphType, nodeCount);
        plotGraph(data.nodes, data.edges);
        initializeGraphState(data.nodes.length);
        mainGame();

        const player1 = document.getElementById('player1').value;
        if (player1 !== 'human') {
            await makeBotMove(player1);
        }
    } catch (error) {
        alert('Failed to fetch graph data!');
    }

    updateScoreBar();
}

function updateScoreBar() {
    const scoreBar = document.getElementById('score-bar');
    const averageOpinion = opinions.reduce((sum, val) => sum + val, 0) / opinions.length;

    const scoreText = document.getElementById('score');
    scoreText.textContent = averageOpinion.toFixed(3).toString();

    const adjustedOpinion = (45 * averageOpinion) + 50;
    scoreBar.style.backgroundImage = `linear-gradient(to top, red 0%, red ${adjustedOpinion - 5}%, blue ${adjustedOpinion + 5}%, blue 100%)`;
}

const canvas = document.getElementById('game-canvas');
canvas.addEventListener('mousemove', function(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    let isCursorOverNode = false;

    graphNodes.forEach(node => {
        if (Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2) <= node.radius) {
            isCursorOverNode = true;
        }
    });

    canvas.style.cursor = isCursorOverNode ? 'pointer' : 'default';
});
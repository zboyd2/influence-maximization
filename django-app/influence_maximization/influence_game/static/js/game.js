let graphNodes = [];
let adjList = [];
let opinions = [];
let controls = [];
let config = [];
let gameStates = [];
let playerTurn = 1;
let turnCount = 0;
let numTurns = 3;
let gameRunning = true;
let updatingOpinions = false;
let botMakingMove = false;

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

    gameStates.push({
        opinions: opinions.slice(),
        controls: controls.slice()
    });

    playerTurn = 1;
    turnCount = 0;
    gameRunning = true;
    updateTurnNotification();
    updateScoreBar();
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

async function handleCanvasClick(event) {
    if (!gameRunning || updatingOpinions || botMakingMove) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    // Iterate in reverse order to prioritize the topmost node
    for (let i = graphNodes.length - 1; i >= 0; i--) {
        const node = graphNodes[i];
        if (Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2) <= node.radius && controls[node.id] === null) {
            controls[node.id] = playerTurn;
            config.push(node.id);

            await updateOpinions();
            updateCanvas();
            updateScoreBar();
            updateTurn();
            break;
        }
    }
}

async function updateOpinions() {
    updatingOpinions = true;
    let notPrecise = true;
    let count = 0;

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

        opinions = newOpinions.slice();
        updateCanvas();
        updateScoreBar();

        if (opinionsPrecise) {
            notPrecise = false;
        }

        // Adjust to control how fast the influence spreads
        const influenceSpeed = (-0.002 * count) + 7;
        await sleep(influenceSpeed);
        count++;
    }

    gameStates.push({
        opinions: opinions.slice(),
        controls: controls.slice()
    });

    updatingOpinions = false;
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
        throw new Error('Failed to fetch bot move.' + response.status);
    }

    const data = await response.json();
    return data.move;
}

async function makeBotMove(difficulty) {
    if (updatingOpinions) return;
    botMakingMove = true;

    try {
        const move = await getBotMove(difficulty);
        await sleep(400);

        controls[move] = playerTurn;
        config.push(move);

        await updateOpinions();
        updateCanvas();
        updateScoreBar();
        updateTurn();
    } catch (error) {
        console.error('Error fetching bot move:', error);
        alert('Failed to get bot move.' + error.message);
    }

    botMakingMove = false;
}

function isBotMove() {
    const player1 = document.getElementById('player1').value;
    const player2 = document.getElementById('player2').value;
    return (playerTurn === 1 && player1 !== 'human') || (playerTurn === 0 && player2 !== 'human');
}

async function updateTurn() {
    turnCount++;
    if (turnCount < numTurns * 2) {
        playerTurn = (playerTurn === 1) ? 0 : 1;
        updateTurnNotification();

        if (isBotMove()) {
            const player1 = document.getElementById('player1').value;
            const player2 = document.getElementById('player2').value;
            const difficulty = (playerTurn === 1) ? player1 : player2;
            await makeBotMove(difficulty);
        }
    } else {
        updateTurnNotification();
        await determineWinner();
        gameRunning = false;
    }
}

function updateTurnNotification() {
    const turnBox = document.getElementById('turn-notification');
    let playerNum = (playerTurn === 1) ? 1 : 2;
    
    if (turnCount === numTurns * 2) {
        turnBox.textContent = 'Game Over';
    } else {
        turnBox.textContent = `Player ${playerNum}'s Turn`;
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function showModal(message) {
    const modal = document.getElementById('winner-modal');
    const messageElement = document.getElementById('winner-message');
    messageElement.textContent = message;
    modal.style.display = 'flex';
}

function closeModal() {
    const modal = document.getElementById('winner-modal');
    modal.style.display = 'none';
}

async function determineWinner() {
    const averageOpinion = opinions.reduce((sum, val) => sum + val, 0) / opinions.length;

    let winner;
    if (averageOpinion > 0) {
        winner = "Player 1 Wins!";
    } else if (averageOpinion < 0) {
        winner = "Player 2 Wins!";
    } else {
        winner = "Draw";
    }

    await sleep(500);
    showModal(winner);
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

    // Draw edges
    adjList.forEach((neighbors, index) => {
        neighbors.forEach(neighbor => plotEdge(index, neighbor));
    });

    // Draw nodes without control
    graphNodes.forEach((node, index) => {
        if (controls[node.id] === null) {
            plotNode(node.x, node.y, opinions[index], null);
        }
    });

    // Draw nodes with control
    graphNodes.forEach((node, index) => {
        if (controls[node.id] !== null) {
            plotNode(node.x, node.y, opinions[index], controls[node.id]);
        }
    });
}

function highlightNode(nodeid) {
    const canvas = document.getElementById('game-canvas');
    const ctx = canvas.getContext('2d');
    const node = graphNodes.find(node => node.id === nodeid);

    if (node) {
        const ringRadius = node.radius + 10;
        ctx.strokeStyle = '#36FF00';
        ctx.lineWidth = 6;

        ctx.beginPath();
        ctx.arc(node.x, node.y, ringRadius, 0, 2 * Math.PI);
        ctx.stroke();
    }
}

async function askCoach() {
    const player1 = document.getElementById('player1').value;
    const player2 = document.getElementById('player2').value;

    // Verifies that the bot isn't doing its turn
    if (isBotMove()) {
        return;
    }

    const move = await getBotMove('hard');
    highlightNode(move);
}

async function resetGame() {
    const graphType = document.getElementById('graph-type').value;
    const nodeCount = parseInt(document.getElementById('node-count').value);

    graphNodes = [];
    adjList = [];
    opinions = [];
    controls = [];
    config = [];
    gameStates = [];
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
        updateScoreBar();

        const player1 = document.getElementById('player1').value;
        if (player1 !== 'human') {
            await makeBotMove(player1);
        }
    } catch (error) {
        alert('Failed to fetch graph data!');
    }
}

async function undoMove() {
    if (botMakingMove || updatingOpinions || gameStates.length <= 1) return;

    // Handles when a bot is player 1
    if (gameStates.length == 2 && ((playerTurn === 1 && player1 !== 'human') || (playerTurn === 0 && player2 !== 'human'))) return;

    if (!gameRunning) {
        gameRunning = true;
        playerTurn = (playerTurn === 1) ? 0 : 1;
    }

    const player1 = document.getElementById('player1').value;
    const player2 = document.getElementById('player2').value;


    do {        
        gameStates.pop();
        const prevState = gameStates.at(-1);
        opinions = prevState.opinions.slice();
        controls = prevState.controls.slice();
        config.pop();

        turnCount--;
        playerTurn = (playerTurn === 1) ? 0 : 1;
    } while (isBotMove() && gameStates.length > 1);

    updateCanvas();
    updateScoreBar();
    updateTurnNotification();

    // If both players are bots, trigger the next move
    if (isBotMove()) {
        await sleep(500);
        turnCount--;
        playerTurn = (playerTurn === 1) ? 0 : 1;
        updateTurn();
    }
}

function startupScreen() {
    document.getElementById('gameplay').style.display = 'none';
    document.getElementById('startup-screen').style.display = 'block';

    graphNodes = [];
    adjList = [];
    opinions = [];
    controls = [];
    config = [];
    gameStates = [];
    playerTurn = 1;
    turnCount = 0;
    gameRunning = true;
    updatingOpinions = false;
    botMakingMove = false;

    const canvas = document.getElementById('game-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function updateScoreBar() {
    const scoreBar = document.getElementById('score-bar');
    const scoreBox = document.getElementById('score-box');
    const averageOpinion = opinions.reduce((sum, val) => sum + val, 0) / opinions.length;

    const scoreText = document.getElementById('score');
    scoreText.textContent = averageOpinion.toFixed(3).toString();

    const adjustedOpinion = (45 * averageOpinion) + 50;
    scoreBar.style.backgroundImage = `linear-gradient(to bottom, red 0%, red ${adjustedOpinion - 5}%, blue ${adjustedOpinion + 5}%, blue 100%)`;

    // Change background color based on the score
    if (averageOpinion > 0) {
        scoreBox.style.backgroundColor = '#fdc0c0';
    } else if (averageOpinion < 0) {
        scoreBox.style.backgroundColor = '#c0c0fc';
    } else {
        scoreBox.style.backgroundColor = '#ccc';
    }
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
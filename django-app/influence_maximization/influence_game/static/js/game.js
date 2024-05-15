function startGame() {
    const graphType = document.getElementById('graph-type').value;
    const player1 = document.getElementById('player1').value;
    const player2 = document.getElementById('player2').value;

    document.getElementById('startup-screen').style.display = 'none';
    document.getElementById('gameplay').style.display = 'flex';

    const player1Text = document.getElementById('player1-indicator');
    player1Text.textContent = `Player 1: ${player1.charAt(0).toUpperCase() + player1.slice(1)}`;
    const player2Text = document.getElementById('player2-indicator');
    player2Text.textContent = `Player 2: ${player2.charAt(0).toUpperCase() + player2.slice(1)}`;

    switch (graphType) {
        case 'distribution':
            callDistribution();
            break;
        case 'tree':
            callTree();
            break;
        case 'ladder':
            callLadder();
            break;
        case 'square':
            callSquare();
            break;
        case 'hexagon':
            callHexagon();
            break;
        case 'triangle':
            callTriangle();
            break;
        case 'cycle':
            callCycle();
            break;
        case 'random-proximity':
            callRandomProximity();
            break;
        default:
            alert('Invalid graph type');
            break;
    }
}

let graphNodes = [];
let opinions = [];
let playerTurn = 1;

function plotNode(x, y) {
    const canvas = document.getElementById('game-canvas');
    const ctx = canvas.getContext('2d');
    const radius = 12;

    ctx.fillStyle = 'purple';
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
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

function plotGraph(nodes, edges) {
    const nodeCoords = JSON.stringify(nodes);
    const coordsArray = JSON.parse(nodeCoords);

    const edgeList = JSON.stringify(edges);
    const edgeArray = JSON.parse(edgeList);
    const nodeRadius = 12;

    coordsArray.forEach((coords, index) => {
        graphNodes.push({x: coords[0], y: coords[1], radius: nodeRadius, id: index});
        opinions.push(0);
    });

    edgeArray.forEach(edge => {
        plotEdge(edge[0], edge[1]);
    });

    coordsArray.forEach((coords, index) => {
        plotNode(coords[0], coords[1]);
    });
}

function updateTurn() {
    const turnBox = document.getElementById('turn-notification');
    if (playerTurn == 1) {
        playerTurn = 2;
    } else {
        playerTurn = 1;
    }

    turnBox.textContent = `Player ${playerTurn}'s Turn`;
}

function askCoach() {

}

function resetGame() {

}

function updateScoreBar() {
    const scoreBar = document.getElementById('score-bar');
    const redPercent = 0;
    const bluePercent = 100;
    scoreBar.style.backgroundImage = 'linear-gradient(to right, right ${redPercent}%, blue ${bluePercent}%)';
}

function callDistribution() {
    fetch('/api/distribution/')
    .then(response => response.json())
    .then(data => {
        plotGraph(data.nodes, data.edges);
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        alert('Failed to fetch graph data!');
    });
}

function callTree() {
    fetch('/api/tree/')
    .then(response => response.json())
    .then(data => {
        plotGraph(data.nodes, data.edges);
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        alert('Failed to fetch graph data!');
    });
}

function callLadder() {
    fetch('/api/ladder/')
    .then(response => response.json())
    .then(data => {
        plotGraph(data.nodes, data.edges);
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        alert('Failed to fetch graph data!');
    });
}

function callSquare() {
    fetch('/api/square/')
    .then(response => response.json())
    .then(data => {
        plotGraph(data.nodes, data.edges);
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        alert('Failed to fetch graph data!');
    });
}

function callHexagon() {
    fetch('/api/hexagon/')
    .then(response => response.json())
    .then(data => {
        plotGraph(data.nodes, data.edges);
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        alert('Failed to fetch graph data!');
    });
}

function callTriangle() {
    fetch('/api/triangle/')
    .then(response => response.json())
    .then(data => {
        plotGraph(data.nodes, data.edges);
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        alert('Failed to fetch graph data!');
    });
}

function callCycle() {
    fetch('/api/cycle/')
    .then(response => response.json())
    .then(data => {
        plotGraph(data.nodes, data.edges);
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        alert('Failed to fetch graph data!');
    });
}

function callRandomProximity() {
    fetch('/api/random_proximity/')
    .then(response => response.json())
    .then(data => {
        plotGraph(data.nodes, data.edges);
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        alert('Failed to fetch graph data!');
    });
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

canvas.addEventListener('click', function(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    let cursorOverNode = false;

    graphNodes.forEach(node => {
        if (Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2) <= node.radius) {
            alert('Clicked on node ' + node.id);
            cursorOverNode = true;
        }
    });

    canvas.style.cursor = cursorOverNode ? 'pointer' : 'default';
});
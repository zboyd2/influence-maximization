function startGame() {
    const graphType = document.getElementById('graph-type').value;
    const player1 = document.getElementById('player1').value;
    const player2 = document.getElementById('player2').value;

    document.getElementById('startup-screen').style.display = 'none';

    document.getElementById('gameplay').style.display = 'block';
    callRandomProximity();
}

let nodes = [];

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

    const node1 = nodes.find(node => node.id === nodeid1);
    const node2 = nodes.find(node => node.id === nodeid2);

    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(node1.x, node1.y);
    ctx.lineTo(node2.x, node2.y);
    ctx.stroke();
}

function callRandomProximity() {
    fetch('/api/random_proximity/')
    .then(response => response.json())
    .then(data => {
        const nodesOutput = document.getElementById('nodes-display');
        const edgesOutput = document.getElementById('edges-display');

        const nodeCoords = JSON.stringify(data.nodes);
        const coordsArray = JSON.parse(nodeCoords);

        const edgeList = JSON.stringify(data.edges);
        const edgeArray = JSON.parse(edgeList);
        const nodeRadius = 12;

        coordsArray.forEach((coords, index) => {
            nodes.push({x: coords[0], y: coords[1], radius: nodeRadius, id: index});
        });

        edgeArray.forEach(edge => {
            plotEdge(edge[0], edge[1]);
        });

        coordsArray.forEach((coords, index) => {
            plotNode(coords[0], coords[1]);
        });
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

    nodes.forEach(node => {
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

    nodes.forEach(node => {
        if (Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2) <= node.radius) {
            alert('Clicked on node ' + node.id);
            cursorOverNode = true;
        }
    });

    canvas.style.cursor = cursorOverNode ? 'pointer' : 'default';
});
function startGame() {
    const graphType = document.getElementById('graph-type').value;
    const player1 = document.getElementById('player1').value;
    const player2 = document.getElementById('player2').value;

    document.getElementById('startup-screen').style.display = 'none';

    console.log('Starting game with:', { graphType, player1, player2 });
}
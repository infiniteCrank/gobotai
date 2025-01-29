const connection = new WebSocket('ws://localhost:8080/ws');

connection.onopen = () => {
    console.log('WebSocket connected');
};

connection.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    
    if (msg.type === "response") {
        const messagesContainer = document.getElementById('messages');
        messagesContainer.innerHTML += `<div>${msg.response}</div>`;
    }
};

// Event listener for the send button
document.getElementById('send').onclick = () => {
    const query = document.getElementById('query').value;
    connection.send(JSON.stringify({ type: "query", query }));
    document.getElementById('query').value = ''; // Clear input field
};


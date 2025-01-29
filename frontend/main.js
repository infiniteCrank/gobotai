const connection = new WebSocket('ws://localhost:8080/ws');

connection.onopen = () => {
    console.log('WebSocket connected');
};

connection.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    
    if (msg.type === "response") {
        const messagesContainer = document.getElementById('messages');
        const htmlContent = marked(msg.response);
        messagesContainer.innerHTML += `<div>${htmlContent}</div>`;
    }
};

// Event listener for the send button
document.getElementById('send').onclick = () => {
    const query = document.getElementById('query').value;
    connection.send(JSON.stringify({ type: "query", query }));
    document.getElementById('query').value = ''; // Clear input field
};


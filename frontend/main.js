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
    if (msg.type === "scores") {

        const corpusScores = JSON.parse(msg.corpus);
        formattedCorpusScores = formatD3Data(corpusScores);
        //console.log(formattedCorpusScores);
        // Create the corpus word cloud
        createWordCloud(formattedCorpusScores, "#corpusCloud");

        const queryScores = JSON.parse(msg.query);
        formattedQueryScores = formatD3Data(queryScores);
        //console.log(formattedQueryScores);
        // Create the query word cloud
        createWordCloud(formattedQueryScores, "#queryCloud");
    }
};

function formatD3Data(jsonObject) {
    let formatted = [];
    for (const key in jsonObject) {
        if (jsonObject.hasOwnProperty(key)) {
            const value = jsonObject[key];
            formatted.push({ "word": key, "score": value })
        }
    }
    return formatted;
}

const width = 800;
const height = 600;

function createWordCloud(data, svgId) {
    const words = data.map(d => ({
        text: d.word,
        size: d.score * 2500 // Scale the score for better visibility
    }));

    const layout = d3.layout.cloud()
        .size([width, height])
        .words(words)
        .padding(5)
        .rotate(() => ~~(Math.random() * 2) * 90)
        .fontSize(d => d.size)
        .on("end", draw);

    layout.start();

    function draw(words) {
        d3.select(svgId).append("g")
            .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")")
            .selectAll("text")
            .data(words)
            .enter().append("text")
            .style("font-size", d => d.size + "px")
            .style("fill", () => d3.schemeCategory10[Math.floor(Math.random() * 10)])
            .attr("text-anchor", "middle")
            .attr("transform", d => "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")")
            .text(d => d.text);
    }
}

// Event listener for the send button
document.getElementById('send').onclick = () => {
    const query = document.getElementById('query').value;
    connection.send(JSON.stringify({ type: "query", query }));
    const messagesContainer = document.getElementById('messages');
    messagesContainer.innerHTML += `<div class="userQuery">${document.getElementById('query').value}</div>`;
    document.getElementById('query').value = ''; // Clear input field
};


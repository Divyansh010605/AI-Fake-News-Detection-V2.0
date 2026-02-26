let currentMode = 'text';

function switchTab(mode) {
    currentMode = mode;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    if (mode === 'text') {
        document.getElementById('text-input-section').style.display = 'block';
        document.getElementById('url-input-section').style.display = 'none';
    } else {
        document.getElementById('text-input-section').style.display = 'none';
        document.getElementById('url-input-section').style.display = 'block';
    }
    document.getElementById('result').style.display = 'none';
}

async function analyzeNews() {
    const text = document.getElementById('newsText').value;
    const url = document.getElementById('newsUrl').value;
    const resultBox = document.getElementById('result');
    const loader = document.getElementById('loader');
    
    if (currentMode === 'text' && !text) return alert("Please enter text.");
    if (currentMode === 'url' && !url) return alert("Please enter a URL.");

    loader.style.display = 'block';
    resultBox.style.display = 'none';

    const payload = currentMode === 'text' ? { text: text } : { url: url };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        loader.style.display = 'none';

        if (data.error) {
            alert(data.error);
        } else {
            displayResult(data);
            addToHistory(data.label, currentMode === 'text' ? text.substring(0, 30) + "..." : url);
        }
    } catch (error) {
        loader.style.display = 'none';
        alert("Server error. Is app.py running?");
    }
}

function displayResult(data) {
    document.getElementById('result').style.display = 'block';
    const label = document.getElementById('predictionLabel');
    const bar = document.getElementById('confidenceBar');
    
    label.innerText = data.label;
    label.style.color = data.label === 'REAL' ? '#3fb950' : '#f85149';
    bar.style.width = data.confidence + "%";
    bar.style.backgroundColor = data.label === 'REAL' ? '#3fb950' : '#f85149';
    document.getElementById('confidenceText').innerText = data.confidence + "% Confidence";
    document.getElementById('verificationNote').innerText = data.verification_note || "Analysis Complete.";

    const tagContainer = document.getElementById('explanation-tags');
    tagContainer.innerHTML = '';
    data.explanation.forEach(item => {
        const span = document.createElement('span');
        span.className = 'explain-tag';
        span.innerText = `${item[0]} (${item[1].toFixed(2)})`;
        span.style.borderColor = item[1] > 0 ? '#f85149' : '#3fb950'; 
        tagContainer.appendChild(span);
    });

    document.getElementById('factStatus').innerText = data.fact_check.status;
    const sourceList = document.getElementById('sourcesList');
    sourceList.innerHTML = '';
    if (data.fact_check.sources.length === 0) {
        sourceList.innerHTML = '<li>No relevant sources found.</li>';
    } else {
        data.fact_check.sources.forEach(src => {
            const li = document.createElement('li');
            li.innerHTML = `<a href="${src.link}" target="_blank">${src.title}</a> <br> <small>${src.source}</small>`;
            sourceList.appendChild(li);
        });
    }
}

function addToHistory(label, snippet) {
    const list = document.getElementById('history-list');
    const item = document.createElement('div');
    item.className = 'history-item';
    const tagClass = label === 'REAL' ? 'tag-real' : 'tag-fake';
    item.innerHTML = `<span class="history-tag ${tagClass}">${label}</span> ${snippet}`;
    list.prepend(item);
    
    let history = JSON.parse(localStorage.getItem('veritas_history') || '[]');
    history.unshift({ label, snippet });
    if (history.length > 10) history.pop();
    localStorage.setItem('veritas_history', JSON.stringify(history));
}

function loadHistory() {
    const list = document.getElementById('history-list');
    list.innerHTML = '';
    let history = JSON.parse(localStorage.getItem('veritas_history') || '[]');
    history.forEach(h => {
        const item = document.createElement('div');
        item.className = 'history-item';
        const tagClass = h.label === 'REAL' ? 'tag-real' : 'tag-fake';
        item.innerHTML = `<span class="history-tag ${tagClass}">${h.label}</span> ${h.snippet}`;
        list.appendChild(item);
    });
}

function clearHistory() {
    localStorage.removeItem('veritas_history');
    loadHistory();
}

function shareResult() {
    alert("Screenshot this page to share your analysis!");
}

window.onload = loadHistory;
/* ============================================
   Veritas Pro — Client-Side Logic
   ============================================ */

let currentMode = 'text';
let lastResult = null;

// ============================================
// Toast Notification System
// ============================================
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icons = {
        error: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" /></svg>',
        success: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>',
        info: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" /></svg>'
    };

    toast.innerHTML = `${icons[type] || icons.info}<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ============================================
// Tab Switching
// ============================================
function switchTab(mode, btnElement) {
    currentMode = mode;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btnElement.classList.add('active');

    const textSection = document.getElementById('text-input-section');
    const urlSection = document.getElementById('url-input-section');

    if (mode === 'text') {
        textSection.style.display = 'block';
        urlSection.style.display = 'none';
    } else {
        textSection.style.display = 'none';
        urlSection.style.display = 'block';
    }

    document.getElementById('result').style.display = 'none';
}

// ============================================
// Mobile Sidebar Toggle
// ============================================
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    if (!sidebar || !overlay) return;

    sidebar.classList.toggle('open');
    overlay.classList.toggle('active');
}

// ============================================
// Analysis
// ============================================
async function analyzeNews() {
    const text = document.getElementById('newsText').value.trim();
    const url = document.getElementById('newsUrl').value.trim();
    const resultBox = document.getElementById('result');
    const loader = document.getElementById('loader');
    const btn = document.getElementById('analyzeBtn');

    if (currentMode === 'text' && !text) {
        showToast('Please enter some text to analyze.', 'error');
        return;
    }
    if (currentMode === 'url' && !url) {
        showToast('Please enter a URL to analyze.', 'error');
        return;
    }

    // Show loading state
    loader.style.display = 'block';
    resultBox.style.display = 'none';
    btn.disabled = true;
    btn.querySelector('span').textContent = 'ANALYZING...';

    const payload = currentMode === 'text' ? { text } : { url };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        loader.style.display = 'none';
        btn.disabled = false;
        btn.querySelector('span').textContent = 'RUN ANALYSIS';

        if (data.error) {
            showToast(data.error, 'error');
        } else {
            lastResult = data;
            displayResult(data);
            addToHistory(data.label, currentMode === 'text' ? text.substring(0, 35) + '...' : url);
            showToast('Analysis complete!', 'success', 2500);

            // Smooth scroll to results
            setTimeout(() => {
                resultBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        }
    } catch (error) {
        loader.style.display = 'none';
        btn.disabled = false;
        btn.querySelector('span').textContent = 'RUN ANALYSIS';
        showToast('Connection error. Is the server running?', 'error');
    }
}

// ============================================
// Display Results
// ============================================
function displayResult(data) {
    const resultBox = document.getElementById('result');
    resultBox.style.display = 'block';

    const label = document.getElementById('predictionLabel');
    const bar = document.getElementById('confidenceBar');
    const confText = document.getElementById('confidenceText');

    const isReal = data.label === 'REAL';
    const color = isReal ? '#22c55e' : '#ef4444';

    // Verdict
    label.innerText = data.label;
    label.style.color = color;

    // Animated confidence bar
    bar.style.width = '0%';
    bar.style.backgroundColor = color;
    requestAnimationFrame(() => {
        setTimeout(() => {
            bar.style.width = data.confidence + '%';
        }, 50);
    });

    // Animated confidence counter
    animateCounter(confText, 0, data.confidence, 1000);

    // Verification note with color-coding
    const noteEl = document.getElementById('verificationNote');
    const noteText = data.verification_note || 'Analysis complete.';
    noteEl.innerText = noteText;
    noteEl.className = 'verification-note';

    if (noteText.includes('High Risk')) {
        noteEl.classList.add('note-danger');
    } else if (noteText.includes('Verified')) {
        noteEl.classList.add('note-success');
    } else if (noteText.includes('Caution')) {
        noteEl.classList.add('note-warning');
    }

    // Text preview
    const previewSection = document.getElementById('textPreviewSection');
    const previewEl = document.getElementById('textPreview');
    if (data.preview) {
        previewSection.style.display = 'block';
        previewEl.innerText = data.preview;
    } else {
        previewSection.style.display = 'none';
    }

    // AI Explanation tags
    const tagContainer = document.getElementById('explanation-tags');
    tagContainer.innerHTML = '';

    if (data.explanation && data.explanation.length > 0) {
        data.explanation.forEach(item => {
            const span = document.createElement('span');
            const isPositive = item[1] > 0;
            span.className = `explain-tag ${isPositive ? 'tag-positive' : 'tag-negative'}`;
            span.innerText = `${item[0]} (${item[1].toFixed(3)})`;
            tagContainer.appendChild(span);
        });
    } else {
        tagContainer.innerHTML = '<span style="color: var(--text-dim); font-size: 0.85rem;">No word-level explanations available.</span>';
    }

    // Fact Check
    const factStatus = document.getElementById('factStatus');
    const sourceList = document.getElementById('sourcesList');
    sourceList.innerHTML = '';

    if (data.fact_check) {
        const status = data.fact_check.status || 'Unknown';
        factStatus.innerText = status;
        factStatus.className = 'fact-status-badge';

        if (status.includes('Verified')) {
            factStatus.classList.add('status-verified');
        } else if (status.includes('Found')) {
            factStatus.classList.add('status-found');
        } else {
            factStatus.classList.add('status-none');
        }

        if (data.fact_check.sources && data.fact_check.sources.length > 0) {
            data.fact_check.sources.forEach(src => {
                const li = document.createElement('li');
                const badge = src.reputable
                    ? '<span class="source-badge reputable">Verified Source</span>'
                    : '';
                li.innerHTML = `<a href="${src.link}" target="_blank" rel="noopener noreferrer">${escapeHtml(src.title)}</a>${badge}<br><small>${escapeHtml(src.source)}</small>`;
                sourceList.appendChild(li);
            });
        } else {
            sourceList.innerHTML = '<li class="no-sources">No relevant sources found.</li>';
        }
    } else {
        factStatus.innerText = 'Unavailable';
        factStatus.className = 'fact-status-badge status-none';
        sourceList.innerHTML = '<li class="no-sources">Fact checking unavailable.</li>';
    }
}

// ============================================
// Animated Counter
// ============================================
function animateCounter(element, start, end, duration) {
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * eased;
        element.innerText = current.toFixed(1) + '% Confidence';
        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.innerText = end.toFixed(2) + '% Confidence';
        }
    }
    requestAnimationFrame(update);
}

// ============================================
// History Management
// ============================================
function addToHistory(label, snippet) {
    const list = document.getElementById('history-list');
    if (!list) return;

    const item = document.createElement('div');
    item.className = 'history-item';
    const tagClass = label === 'REAL' ? 'tag-real' : 'tag-fake';
    item.innerHTML = `<span class="history-tag ${tagClass}">${label}</span> ${escapeHtml(snippet)}`;
    list.prepend(item);

    let history = JSON.parse(localStorage.getItem('veritas_history') || '[]');
    history.unshift({ label, snippet });
    if (history.length > 15) history.pop();
    localStorage.setItem('veritas_history', JSON.stringify(history));
}

function loadHistory() {
    const list = document.getElementById('history-list');
    if (!list) return;

    list.innerHTML = '';
    let history = JSON.parse(localStorage.getItem('veritas_history') || '[]');
    history.forEach(h => {
        const item = document.createElement('div');
        item.className = 'history-item';
        const tagClass = h.label === 'REAL' ? 'tag-real' : 'tag-fake';
        item.innerHTML = `<span class="history-tag ${tagClass}">${h.label}</span> ${escapeHtml(h.snippet)}`;
        list.appendChild(item);
    });
}

function clearHistory() {
    localStorage.removeItem('veritas_history');
    loadHistory();
    showToast('History cleared.', 'info', 2000);
}

// ============================================
// Share Result
// ============================================
function shareResult() {
    if (!lastResult) {
        showToast('No analysis to share yet.', 'error');
        return;
    }

    const text = `🔍 Veritas Pro Analysis\n━━━━━━━━━━━━━━━━━━━━\n📊 Verdict: ${lastResult.label}\n📈 Confidence: ${lastResult.confidence}%\n🔎 Fact Check: ${lastResult.fact_check?.status || 'N/A'}\n📝 ${lastResult.verification_note || ''}\n\nAnalyzed by Veritas Pro — AI Truth Engine`;

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            showToast('Result copied to clipboard!', 'success', 2500);
        }).catch(() => {
            fallbackCopy(text);
        });
    } else {
        fallbackCopy(text);
    }
}

function fallbackCopy(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    try {
        document.execCommand('copy');
        showToast('Result copied to clipboard!', 'success', 2500);
    } catch (e) {
        showToast('Could not copy. Please copy manually.', 'error');
    }
    document.body.removeChild(textarea);
}

// ============================================
// Utilities
// ============================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

// ============================================
// Initialize
// ============================================
window.onload = loadHistory;
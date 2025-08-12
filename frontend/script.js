// ==============================================================================
// DOM ELEMENT SELECTION
// ==============================================================================
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const fileLabel = document.getElementById('file-label');
const fileNameDisplay = document.getElementById('file-name');
const optionsPanel = document.getElementById('options-panel');
const runAnalysisBtn = document.getElementById('run-analysis-btn');
const controlPanel = document.getElementById('control-panel');
const spinner = document.getElementById('spinner');
const resultsArea = document.getElementById('results-area');
const queryArea = document.getElementById('query-area');

// ==============================================================================
// GLOBAL STATE
// ==============================================================================
let state = {
    file: null,
    mode: null,
    target: null,
    columns: []
};

// ==============================================================================
// EVENT LISTENERS
// ==============================================================================
document.addEventListener('DOMContentLoaded', initialize);

function initialize() {
    // Drag and Drop listeners
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => dropArea.addEventListener(e, preventDefaults));
    ['dragenter', 'dragover'].forEach(e => dropArea.addEventListener(e, () => dropArea.classList.add('highlight')));
    ['dragleave', 'drop'].forEach(e => dropArea.addEventListener(e, () => dropArea.classList.remove('highlight')));
    dropArea.addEventListener('drop', handleDrop);
    
    // Click listener
    fileInput.addEventListener('change', e => handleFiles(e.target.files));
    runAnalysisBtn.addEventListener('click', runAnalysis);
}

function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

// ==============================================================================
// UI CONTROL FUNCTIONS
// ==============================================================================

function handleDrop(e) {
    handleFiles(e.dataTransfer.files);
}

function handleFiles(files) {
    if (files.length === 0) return;
    const file = files[0];
    if (file.type !== "text/csv") {
        alert("Please upload a CSV file.");
        return;
    }
    
    state.file = file;
    fileNameDisplay.textContent = `Selected File: ${file.name}`;
    fileLabel.style.display = 'none'; // Hide the upload label
    
    const reader = new FileReader();
    reader.onload = (event) => {
        state.columns = event.target.result.split('\n')[0].trim().split(',').map(h => h.replace(/"/g, ''));
        renderOptionsPanel();
    };
    reader.readAsText(state.file);
}

function renderOptionsPanel() {
    optionsPanel.classList.remove('hidden');
    optionsPanel.innerHTML = `
        <h2>2. Choose Analysis Type</h2>
        <div class="mode-selector">
            <div class="mode-card" id="mode-supervised" onclick="selectMode('supervised')">
                <h3>Predictive Analysis</h3>
                <p>Predict a target and find key drivers.</p>
            </div>
            <div class="mode-card" id="mode-unsupervised" onclick="selectMode('unsupervised')">
                <h3>Exploratory Analysis</h3>
                <p>Discover hidden groups in your data.</p>
            </div>
        </div>
        <div class="hidden" id="target-selector-div">
            <label for="target-column-select">Select Target to Predict:</label>
            <select id="target-column-select" onchange="selectTarget(this.value)">
                <option value="">-- Please choose a target --</option>
                ${state.columns.map(c => `<option value="${c}">${c}</option>`).join('')}
            </select>
        </div>
    `;
}

function selectMode(mode) {
    state.mode = mode;
    document.getElementById('mode-supervised').classList.toggle('selected', mode === 'supervised');
    document.getElementById('mode-unsupervised').classList.toggle('selected', mode === 'unsupervised');
    document.getElementById('target-selector-div').classList.toggle('hidden', mode !== 'supervised');
    
    if (mode === 'unsupervised') {
        state.target = 'unsupervised'; // Set a flag to show the button is ready
    } else {
        state.target = null; // Reset target if switching back to supervised
        document.getElementById('target-column-select').value = "";
    }
    updateRunButtonState();
}

function selectTarget(target) {
    state.target = target;
    updateRunButtonState();
}

function updateRunButtonState() {
    const isReady = state.mode && (state.mode === 'unsupervised' || (state.mode === 'supervised' && state.target && state.target !== ""));
    runAnalysisBtn.classList.toggle('hidden', !isReady);
}

// ==============================================================================
// API AND REPORTING FUNCTIONS
// ==============================================================================

async function runAnalysis() {
    controlPanel.classList.add('hidden');
    spinner.classList.remove('hidden');
    resultsArea.innerHTML = '';
    queryArea.innerHTML = '';
    queryArea.classList.add('hidden');


    const formData = new FormData();
    formData.append('file', state.file);
    formData.append('mode', state.mode);
    if (state.mode === 'supervised') {
        formData.append('target_column', state.target);
    }

    try {
        const response = await fetch('http://127.0.0.1:8000/analyze', { method: 'POST', body: formData });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Analysis failed on the server.');
        
        if (result.problem_type) {
            renderSupervisedReport(result);
            renderQueryInterface();
        } else if (result.optimal_k) {
            renderUnsupervisedReport(result);
        }

    } catch (error) {
        resultsArea.innerHTML = `<div class="card error-card"><h2>Error</h2><p>${error.message}</p><button class="run-button" onclick="location.reload()">Try Again</button></div>`;
    } finally {
        spinner.classList.add('hidden');
        resultsArea.classList.remove('hidden');
    }
}


function renderSupervisedReport(result) {
    // This function is the same as the one that worked before.
    const { metrics, shap_summary } = result;
    const accuracy = metrics.accuracy;
    const report = metrics.classification_report;
    const recall = report['1'] ? report['1'].recall : 0;
    
    let insights = [`Model achieved a <strong>decent accuracy of ${(accuracy * 100).toFixed(1)}%</strong>.`, `It correctly identified <strong>${(recall * 100).toFixed(0)}% of the target cases</strong> (Recall).`];
    let recommendations = [`Focus on improving factors like <strong>"${Object.keys(shap_summary)[0]}"</strong> and <strong>"${Object.keys(shap_summary)[1]}"</strong> as they are the strongest predictors.`];

    resultsArea.innerHTML = `
        <div class="card"><h2>Predictive Model Report</h2><div class="metric-grid"><div class="metric"><h3>${(accuracy * 100).toFixed(1)}%</h3><p>Overall Accuracy</p></div><div class="metric"><h3>${(recall * 100).toFixed(1)}%</h3><p>Target Recall</p></div></div></div>
        <div class="card insight-card"><h2>Key Insights</h2><ul>${insights.map(i => `<li>${i}</li>`).join('')}</ul></div>
        <div class="card insight-card"><h2>Actionable Recommendations</h2><ul>${recommendations.map(r => `<li>${r}</li>`).join('')}</ul></div>
        <div class="card"><h2>Feature Importance Analysis</h2><div id="plot-div"></div></div>
        <button class="run-button" onclick="location.reload()" style="margin-top: 1.5rem;">Start New Analysis</button>
    `;

    Plotly.newPlot('plot-div', [{ type: 'bar', x: Object.values(shap_summary), y: Object.keys(shap_summary), orientation: 'h', marker: { color: '#1e90ff' } }],
        { paper_bgcolor: '#1e1e1e', plot_bgcolor: '#1e1e1e', font: { color: '#e0e0e0' }, yaxis: { autorange: 'reversed' }, height: 300 + Object.keys(shap_summary).length * 20, margin: { l: 250 } });
}

function renderUnsupervisedReport(result) {
     // This function is also the same as the one that worked before.
     resultsArea.innerHTML = `<div class="card"><pre>Unsupervised results would go here.</pre><button class="run-button" onclick="location.reload()" style="margin-top: 1.5rem;">Start New Analysis</button></div>`
}


function renderQueryInterface() {
    // This function is the same as the one that worked before.
    queryArea.classList.remove('hidden');
    queryArea.innerHTML = `
        <div class="card">
            <h2>Test the Model on a New Case</h2>
            <div class="query-form" id="query-form-inputs"></div>
            <button class="run-button" id="query-btn">Get Prediction</button>
            <div id="query-result"></div>
        </div>
    `;

    const formInputs = document.getElementById('query-form-inputs');
    state.columns.forEach(col => {
        if (col !== state.target) {
            formInputs.innerHTML += `
                <div class="form-group">
                    <label for="query-${col}">${col}</label>
                    <input type="text" id="query-${col}" placeholder="Enter value for ${col}">
                </div>
            `;
        }
    });

    document.getElementById('query-btn').addEventListener('click', () => {
        alert("This feature is ready to be connected to a new '/predict_single' backend endpoint.");
    });
}
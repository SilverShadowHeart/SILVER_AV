// File: frontend/script.js

// ==============================================================================
// DOM Element References & Global State
// ==============================================================================
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const fileLabel = document.getElementById('file-label');
const fileNameDisplay = document.getElementById('file-name');
const optionsPanel = document.getElementById('options-panel');
const runAnalysisBtn = document.getElementById('run-analysis-btn');
const spinner = document.getElementById('spinner');
const initialSetupContainer = document.getElementById('initial-setup-container');
const workspace = document.getElementById('workspace');
const dashboardGrid = document.getElementById('dashboard-grid');
const plotModalOverlay = document.getElementById('plot-modal-overlay');
const kpiModalOverlay = document.getElementById('kpi-modal-overlay');

let state = {
    file: null,
    mode: null,
    target: null,
    column_details: null,
    all_columns: [],
    dimensions: [],
    measures: []
};

// ==============================================================================
// Event Listeners
// ==============================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Initial Setup Screen Listeners
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => dropArea.addEventListener(e, (ev) => { ev.preventDefault(); ev.stopPropagation(); }));
    ['dragenter', 'dragover'].forEach(e => dropArea.addEventListener(e, () => dropArea.classList.add('highlight')));
    ['dragleave', 'drop'].forEach(e => dropArea.addEventListener(e, () => dropArea.classList.remove('highlight')));
    dropArea.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
    fileInput.addEventListener('change', e => handleFiles(e.target.files));
    runAnalysisBtn.addEventListener('click', runAnalysis);

    // Workspace & Modal Listeners
    document.getElementById('add-plot-btn').addEventListener('click', openPlotModal);
    document.getElementById('create-plot-btn').addEventListener('click', addPlotToDashboard);
    document.getElementById('plot-type-select').addEventListener('change', renderPlotOptions);
    document.getElementById('add-kpi-btn').addEventListener('click', openKpiModal);
    document.getElementById('create-kpi-btn').addEventListener('click', addKpiCardToDashboard);
});

// ==============================================================================
// Initial Setup & Analysis Trigger
// ==============================================================================
function handleFiles(files) {
    if (files.length === 0) return;
    const file = files[0];
    if (file.type !== "text/csv") { alert("Please upload a CSV file."); return; }
    state.file = file;
    fileNameDisplay.textContent = `Selected File: ${file.name}`;
    dropArea.style.borderColor = '#1e90ff';
    fileLabel.innerHTML = `<p><strong>${file.name}</strong> loaded.</p><p style="font-size: 0.9em; color: #888;">Choose an analysis type below.</p>`;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        const firstLine = event.target.result.split('\n')[0].trim();
        state.all_columns = firstLine.split(',').map(h => h.replace(/"/g, ''));
        renderOptionsPanel();
    };
    reader.readAsText(state.file);
}

function renderOptionsPanel() {
    optionsPanel.classList.remove('hidden');
    optionsPanel.innerHTML = `
        <h2>2. Choose Analysis Type</h2>
        <select id="mode-select" onchange="selectMode(this.value)">
            <option value="">-- Select Mode --</option>
            <option value="supervised">Predictive Analysis</option>
            <option value="unsupervised">Exploratory Analysis (Coming Soon)</option>
        </select>
        <div class="hidden" id="target-selector-div">
            <label for="target-column-select">Select Target to Predict:</label>
            <select id="target-column-select" onchange="selectTarget(this.value)">
                <option value="">-- Please choose a target --</option>
                ${state.all_columns.map(c => `<option value="${c}">${c}</option>`).join('')}
            </select>
        </div>`;
}

function selectMode(mode) {
    state.mode = mode;
    document.getElementById('target-selector-div').classList.toggle('hidden', mode !== 'supervised');
    updateRunButtonState();
}

function selectTarget(target) {
    state.target = target;
    updateRunButtonState();
}

function updateRunButtonState() {
    const isReady = state.file && state.mode === 'supervised' && state.target;
    runAnalysisBtn.classList.toggle('hidden', !isReady);
}

async function runAnalysis() {
    document.getElementById('control-panel').classList.add('hidden');
    spinner.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', state.file);
    formData.append('mode', state.mode);
    formData.append('target_column', state.target);

    try {
        const response = await fetch('http://127.0.0.1:8000/analyze', { method: 'POST', body: formData });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Analysis failed.');
        
        // Store all state returned from the backend
        state.column_details = result.column_details;
        state.dimensions = Object.keys(result.column_details).filter(k => result.column_details[k].type !== 'numeric');
        state.measures = Object.keys(result.column_details).filter(k => result.column_details[k].type === 'numeric');

        // Transition to the main workspace view
        initialSetupContainer.classList.add('hidden');
        spinner.classList.add('hidden');
        workspace.classList.remove('hidden');

        // Build the dynamic parts of the workspace
        buildDataSchemaPanel();
        if (result.shap_summary) {
            addShapReportCard(result.shap_summary);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
        location.reload(); // Reload on failure
    }
}

// ==============================================================================
// Workspace & Dashboard Building
// ==============================================================================
function buildDataSchemaPanel() {
    const dimensionsUl = document.querySelector('#dimensions-list ul');
    const measuresUl = document.querySelector('#measures-list ul');
    dimensionsUl.innerHTML = '';
    measuresUl.innerHTML = '';

    state.dimensions.forEach(dim => {
        dimensionsUl.innerHTML += `<li><i class="fa-solid fa-font"></i>${dim}</li>`;
    });
    state.measures.forEach(measure => {
        measuresUl.innerHTML += `<li><i class="fa-solid fa-hashtag"></i>${measure}</li>`;
    });
}

function createDashboardCard(innerHTML, customClass = '') {
    const card = document.createElement('div');
    card.className = `analysis-card ${customClass}`;
    card.innerHTML = innerHTML;
    return card;
}

function addShapReportCard(shapSummary) {
    const plotId = `plot-shap-${Date.now()}`;
    const cardHTML = `
        <div class="card-header">
            <h2>Initial Feature Importance</h2>
            <div class="card-controls">
                <button onclick="this.closest('.analysis-card').remove()">✖</button>
            </div>
        </div>
        <div class="plot-container" id="${plotId}"></div>`;
    const card = createDashboardCard(cardHTML);
    dashboardGrid.appendChild(card);
    Plotly.newPlot(plotId, [{
        type: 'bar',
        x: Object.values(shapSummary),
        y: Object.keys(shapSummary),
        orientation: 'h',
        marker: { color: '#1e90ff' }
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#e0e0e0' },
        yaxis: { autorange: 'reversed' },
        margin: { l: 200, t: 30, b: 30, r: 20 }
    }, { responsive: true });
}

// --- KPI Card Logic ---
function openKpiModal() {
    const measureSelect = document.getElementById('kpi-measure-select');
    measureSelect.innerHTML = state.measures.map(m => `<option value="${m}">${m}</option>`).join('');
    kpiModalOverlay.classList.remove('hidden');
}

async function addKpiCardToDashboard() {
    const column = document.getElementById('kpi-measure-select').value;
    const aggregation = document.getElementById('kpi-agg-select').value;
    kpiModalOverlay.classList.add('hidden');

    try {
        const response = await fetch('http://127.0.0.1:8000/kpi', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ column, aggregation }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail);
        
        const cardHTML = `
            <div class="card-controls" style="text-align: right;">
                <button onclick="this.closest('.analysis-card').remove()">✖</button>
            </div>
            <div class="kpi-card-content">
                <h3 class="kpi-value">${result.kpi_value}</h3>
                <p class="kpi-title">${result.title}</p>
            </div>`;
        const card = createDashboardCard(cardHTML);
        dashboardGrid.appendChild(card);
    } catch (error) {
        alert(`Failed to create KPI card: ${error.message}`);
    }
}

// --- Charting Logic ---
function openPlotModal() {
    renderPlotOptions();
    plotModalOverlay.classList.remove('hidden');
}

function renderPlotOptions() {
    const plotType = document.getElementById('plot-type-select').value;
    const container = document.getElementById('plot-options-container');
    const { measures, dimensions } = state;

    let html = '';
    if (plotType === 'histogram') {
        html = `<div class="form-group"><label for="plot-col1">Select Measure:</label><select id="plot-col1">${measures.map(c => `<option value="${c}">${c}</option>`).join('')}</select></div>`;
    } else if (plotType === 'scatterplot') {
        html = `<div class="form-group"><label for="plot-col1">X-Axis (Measure):</label><select id="plot-col1">${measures.map(c => `<option value="${c}">${c}</option>`).join('')}</select></div>
                <div class="form-group"><label for="plot-col2">Y-Axis (Measure):</label><select id="plot-col2">${measures.map(c => `<option value="${c}">${c}</option>`).join('')}</select></div>
                <div class="form-group"><label for="plot-col3">Color By (Dimension, Optional):</label><select id="plot-col3"><option value="">None</option>${dimensions.map(c => `<option value="${c}">${c}</option>`).join('')}</select></div>`;
    } else if (plotType === 'boxplot') {
        html = `<div class="form-group"><label for="plot-col1">Variable (Measure):</label><select id="plot-col1">${measures.map(c => `<option value="${c}">${c}</option>`).join('')}</select></div>
                <div class="form-group"><label for="plot-col2">Group By (Dimension, Optional):</label><select id="plot-col2"><option value="">None</option>${dimensions.map(c => `<option value="${c}">${c}</option>`).join('')}</select></div>`;
    }
    container.innerHTML = html;
}

async function addPlotToDashboard() {
    const plotType = document.getElementById('plot-type-select').value;
    const col1 = document.getElementById('plot-col1').value;
    const col2_el = document.getElementById('plot-col2');
    const col3_el = document.getElementById('plot-col3');

    const requestBody = {
        plot_type: plotType,
        col1: col1,
        col2: col2_el ? col2_el.value : null,
        col3: col3_el ? col3_el.value : null,
    };
    plotModalOverlay.classList.add('hidden');
    
    try {
        const response = await fetch('http://127.0.0.1:8000/generate_plot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail);

        const { plot_data, title, xaxis, yaxis } = result;
        const plotId = `plot-${Date.now()}`;

        const cardHTML = `
            <div class="card-header">
                <h2>${title}</h2>
                <div class="card-controls">
                    <button onclick="Plotly.downloadImage('${plotId}', {format: 'png', filename: '${title.replace(/ /g, '_')}'})">PNG</button>
                    <button onclick="this.closest('.analysis-card').remove()">✖</button>
                </div>
            </div>
            <div class="plot-container" id="${plotId}"></div>`;
        
        const card = createDashboardCard(cardHTML);
        dashboardGrid.appendChild(card);

        const layout = {
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#e0e0e0' },
            margin: { l: 50, r: 20, t: 40, b: 50 },
            xaxis: { title: xaxis || '', gridcolor: '#444' },
            yaxis: { title: yaxis || '', gridcolor: '#444' },
            showlegend: plot_data.length > 1,
            legend: { x: 1, xanchor: 'right', y: 1 }
        };
        
        Plotly.newPlot(plotId, plot_data, layout, { responsive: true });
    } catch (error) {
        alert(`Failed to create plot: ${error.message}`);
    }
}
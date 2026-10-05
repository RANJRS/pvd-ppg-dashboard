// PVD PPG Classifier Pipeline Dashboard Javascript Logic

// Global state variables
let processedData = [];
let featureData = [];
let activeTab = "overview";
let consolePollInterval = null;
let pipelineRunning = false;
let waveformChart = null;
let classDistChart = null;

const selectedUploadFiles = {
    overview: [],
    pipeline: []
};

const labelNames = ["Normal", "Moderate", "High"];

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    // 1. Initial Data Fetch
    reloadData();

    // 2. Set up event listeners for Drag & Drop uploads
    setupUploadDropzone("overview");
    setupUploadDropzone("pipeline");

    // 3. Keep updating logs and status periodically
    startStatusPolling();
});

// Switch Dashboard Tabs
function switchTab(tabId) {
    activeTab = tabId;
    
    // Update nav item active state
    document.querySelectorAll(".nav-item").forEach(btn => {
        btn.classList.remove("active");
    });
    const activeNav = document.getElementById(`nav-${tabId}`);
    if (activeNav) activeNav.classList.add("active");

    // Update section active state
    document.querySelectorAll(".tab-content").forEach(sec => {
        sec.classList.remove("active");
    });
    const activeSec = document.getElementById(`tab-content-${tabId}`);
    if (activeSec) activeSec.classList.add("active");

    // Update Header Text
    const titleMap = {
        overview: ["Dashboard Overview", "Summary metrics and dataset distributions"],
        waveforms: ["PPG Waveform Visualizer", "View raw & filtered finger and toe PPG pulses"],
        features: ["Feature Explorer", "Review extracted vectors sent to training"],
        model: ["Model Evaluation", "Confusion matrix and normalized scaler metrics"],
        pipeline: ["Pipeline Manager", "Run scripts, track execution and view console logs"]
    };
    
    if (titleMap[tabId]) {
        document.getElementById("page-title").innerText = titleMap[tabId][0];
        document.getElementById("page-subtitle").innerText = titleMap[tabId][1];
    }
}

// Reload Dashboard Telemetry and Datasets
async function reloadData() {
    console.log("[Dashboard] Reloading data...");
    
    // Fetch Pipeline config from python files
    fetchPipelineConfig();

    // Load datasets
    await Promise.all([
        loadProcessedData(),
        loadFeatureData(),
        loadScalerData(),
        loadModelReport()
    ]);

    // Recalculate stats and re-draw Overview charts
    calculateOverviewStats();
}

// Fetch Pipeline Config variables
async function fetchPipelineConfig() {
    try {
        const res = await fetch("/api/config");
        if (res.ok) {
            const config = await res.json();
            if (config.error) {
                console.error("Config error:", config.error);
                return;
            }
            document.getElementById("spec-sample-rate").innerText = `${config.sample_rate_hz} Hz`;
            document.getElementById("spec-bandpass").innerText = `${config.bandpass_low_hz} - ${config.bandpass_high_hz} Hz`;
            document.getElementById("spec-window").innerText = `${config.window_seconds}s (${config.window_samples} samples)`;
            document.getElementById("spec-step").innerText = `${config.step_seconds}s (${config.overlap_pct}% overlap)`;
            document.getElementById("spec-arch").innerText = config.target_arch || "ESP32-S3 (TensorFlow Lite Micro)";
        }
    } catch (e) {
        console.error("Failed to load config:", e);
    }
}

// Load Windowed processed signals
async function loadProcessedData() {
    try {
        const res = await fetch("/data/processed.json");
        if (res.ok) {
            processedData = await res.json();
            console.log(`[Dashboard] Loaded ${processedData.length} segmented windows.`);
            
            // Populate subject filter dropdown in Waveform tab
            populateSubjectSelect();
        } else {
            processedData = [];
            console.warn("processed.json not found on server.");
        }
    } catch (e) {
        console.error("Failed to load processed data:", e);
        processedData = [];
    }
}

// Load Extracted Feature vectors
async function loadFeatureData() {
    const tableBody = document.getElementById("features-table-body");
    try {
        const res = await fetch("/data/features.csv");
        if (res.ok) {
            const text = await res.text();
            featureData = parseCSV(text);
            console.log(`[Dashboard] Loaded ${featureData.length} feature rows.`);
            
            // Populate feature table
            displayFeatureTable(featureData);
        } else {
            featureData = [];
            tableBody.innerHTML = `<tr><td colspan="10" class="text-center" style="color: var(--color-rose);">No features.csv found. Run feature extraction or training first.</td></tr>`;
        }
    } catch (e) {
        console.error("Failed to load feature data:", e);
        featureData = [];
        tableBody.innerHTML = `<tr><td colspan="10" class="text-center">Error reading feature data.</td></tr>`;
    }
}

// Parse CSV utility
function parseCSV(text) {
    const lines = text.trim().split("\n");
    if (lines.length < 2) return [];
    
    const headers = lines[0].split(",").map(h => h.trim());
    const result = [];
    
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        const currentline = lines[i].split(",");
        const obj = {};
        for (let j = 0; j < headers.length; j++) {
            obj[headers[j]] = currentline[j] ? currentline[j].trim() : "";
        }
        result.push(obj);
    }
    return result;
}

// Display feature vectors table
function displayFeatureTable(data) {
    const tableBody = document.getElementById("features-table-body");
    tableBody.innerHTML = "";
    
    if (data.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="10" class="text-center">No matching features found.</td></tr>`;
        document.getElementById("table-search-count").innerText = "Showing 0 rows";
        return;
    }
    
    data.forEach(row => {
        const tr = document.createElement("tr");
        
        let labelName = "Unknown";
        const lbl = parseInt(row.label);
        if (lbl >= 0 && lbl < labelNames.length) {
            labelName = labelNames[lbl];
        }
        
        // Display columns: subject_id, PI_finger, AI_finger, HRV_rmssd, dicrotic_ratio, heart_rate, crest_factor, skewness, kurtosis, label
        tr.innerHTML = `
            <td><strong>${row.subject_id || "N/A"}</strong></td>
            <td>${parseFloat(row.PI_finger || row.PI || 0).toFixed(4)}</td>
            <td>${parseFloat(row.AI_finger || row.AI || 0).toFixed(4)}</td>
            <td>${parseFloat(row.HRV_rmssd || 0).toFixed(4)}</td>
            <td>${parseFloat(row.dicrotic_ratio || 0).toFixed(4)}</td>
            <td>${parseFloat(row.heart_rate || 0).toFixed(2)}</td>
            <td>${parseFloat(row.crest_factor || 0).toFixed(4)}</td>
            <td>${parseFloat(row.skewness || 0).toFixed(4)}</td>
            <td>${parseFloat(row.kurtosis || 0).toFixed(4)}</td>
            <td><span class="pipeline-badge badge-${labelName.toLowerCase()}">${labelName}</span></td>
        `;
        tableBody.appendChild(tr);
    });

    document.getElementById("table-search-count").innerText = `Showing ${data.length} rows`;
}

// Search Feature Table
function onSearchTable() {
    const query = document.getElementById("table-search").value.toLowerCase().trim();
    if (!query) {
        displayFeatureTable(featureData);
        return;
    }

    const filtered = featureData.filter(row => {
        const sid = (row.subject_id || "").toLowerCase();
        const lbl = (row.label || "");
        const lblName = (labelNames[parseInt(lbl)] || "").toLowerCase();
        
        return sid.includes(query) || lblName.includes(query);
    });

    displayFeatureTable(filtered);
}

// Load Scaler metrics
async function loadScalerData() {
    const tableBody = document.getElementById("scaler-stats-body");
    try {
        const res = await fetch("/api/scaler");
        if (res.ok) {
            const stats = await res.json();
            if (stats.error) {
                tableBody.innerHTML = `<tr><td colspan="3" class="text-center" style="color: var(--color-rose);">${stats.error}</td></tr>`;
                return;
            }
            
            tableBody.innerHTML = "";
            for (const [feat, stat] of Object.entries(stats)) {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${feat}</strong></td>
                    <td>${stat.mean.toFixed(6)}</td>
                    <td>${stat.scale.toFixed(6)}</td>
                `;
                tableBody.appendChild(tr);
            }
        }
    } catch (e) {
        console.error("Failed to load scaler stats:", e);
        tableBody.innerHTML = `<tr><td colspan="3" class="text-center">Error reading scaler metrics.</td></tr>`;
    }
}

// Load Model Training report and Confusion Matrix
async function loadModelReport() {
    try {
        const res = await fetch("/api/model/report");
        if (res.ok) {
            const report = await res.json();
            if (report.error) {
                console.warn("Training report error:", report.error);
                resetConfusionMatrix();
                return;
            }
            
            // Populate Confusion matrix in model tab
            const cm = report.confusion_matrix;
            if (cm && cm.length === 3) {
                document.getElementById("cm-0-0").innerText = cm[0][0];
                document.getElementById("cm-0-1").innerText = cm[0][1];
                document.getElementById("cm-0-2").innerText = cm[0][2];

                document.getElementById("cm-1-0").innerText = cm[1][0];
                document.getElementById("cm-1-1").innerText = cm[1][1];
                document.getElementById("cm-1-2").innerText = cm[1][2];

                document.getElementById("cm-2-0").innerText = cm[2][0];
                document.getElementById("cm-2-1").innerText = cm[2][1];
                document.getElementById("cm-2-2").innerText = cm[2][2];
            } else if (cm && cm.length === 2) {
                // If the model was trained on 2 classes, show 2 classes and zero the moderate/high appropriately
                document.getElementById("cm-0-0").innerText = cm[0][0];
                document.getElementById("cm-0-1").innerText = cm[0][1];
                document.getElementById("cm-0-2").innerText = 0;

                document.getElementById("cm-1-0").innerText = cm[1][0];
                document.getElementById("cm-1-1").innerText = cm[1][1];
                document.getElementById("cm-1-2").innerText = 0;

                document.getElementById("cm-2-0").innerText = 0;
                document.getElementById("cm-2-1").innerText = 0;
                document.getElementById("cm-2-2").innerText = 0;
            }
        } else {
            resetConfusionMatrix();
        }
    } catch (e) {
        console.error("Failed to load model report:", e);
        resetConfusionMatrix();
    }
}

function resetConfusionMatrix() {
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            const cell = document.getElementById(`cm-${i}-${j}`);
            if (cell) cell.innerText = "-";
        }
    }
}

// Calculate Summary Statistics for Overview
function calculateOverviewStats() {
    if (processedData.length === 0) {
        document.getElementById("stat-subjects").innerText = "0";
        document.getElementById("stat-windows").innerText = "0";
        document.getElementById("stat-normal").innerText = "0";
        document.getElementById("stat-moderate-high").innerText = "0";
        
        drawClassDistChart(0, 0, 0);
        return;
    }
    
    // 1. Total window segments
    const totalWindows = processedData.length;
    document.getElementById("stat-windows").innerText = totalWindows;
    
    // 2. Total subjects
    const subjects = [...new Set(processedData.map(d => d.subject_id))];
    document.getElementById("stat-subjects").innerText = subjects.length;
    
    // 3. Counts by class (normal=0, moderate=1, high=2)
    let normalCount = 0;
    let moderateCount = 0;
    let highCount = 0;
    
    processedData.forEach(d => {
        if (d.label === 0) normalCount++;
        else if (d.label === 1) moderateCount++;
        else if (d.label === 2) highCount++;
    });
    
    document.getElementById("stat-normal").innerText = normalCount;
    document.getElementById("stat-moderate-high").innerText = moderateCount + highCount;
    
    // 4. Draw Chart.js distribution
    drawClassDistChart(normalCount, moderateCount, highCount);
}

// Draw Class Distribution Pie Chart
function drawClassDistChart(normal, moderate, high) {
    const ctx = document.getElementById("class-dist-chart").getContext("2d");
    
    if (classDistChart) {
        classDistChart.destroy();
    }
    
    const colors = getComputedStyle(document.documentElement);
    const textThemeColor = colors.getPropertyValue("--text-primary").trim();
    
    classDistChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal', 'Moderate', 'High'],
            datasets: [{
                data: [normal, moderate, high],
                backgroundColor: [
                    '#10b981', // emerald
                    '#f59e0b', // amber
                    '#f43f5e'  // rose
                ],
                borderWidth: 2,
                borderColor: colors.getPropertyValue("--bg-sidebar").trim() || '#111827'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: textThemeColor,
                        font: {
                            family: 'Inter',
                            size: 11,
                            weight: '500'
                        },
                        padding: 15
                    }
                }
            },
            cutout: '65%'
        }
    });
}

// Populate Waveform Subject ID Filter list
function populateSubjectSelect() {
    const select = document.getElementById("select-subject");
    select.innerHTML = "";
    
    if (processedData.length === 0) {
        select.innerHTML = '<option value="">No data loaded</option>';
        return;
    }
    
    const subjects = [...new Set(processedData.map(d => d.subject_id))].sort();
    
    subjects.forEach(subj => {
        const opt = document.createElement("option");
        opt.value = subj;
        opt.innerText = subj;
        select.appendChild(opt);
    });

    // Populate windows for first subject
    onSubjectChange();
}

// Populate Waveform Window Filter list when Subject changes
function onSubjectChange() {
    const subjectId = document.getElementById("select-subject").value;
    const windowSelect = document.getElementById("select-window");
    windowSelect.innerHTML = "";
    
    if (!subjectId) return;
    
    const subjectWindows = processedData.filter(d => d.subject_id === subjectId);
    
    subjectWindows.forEach((win, index) => {
        const opt = document.createElement("option");
        opt.value = index;
        opt.innerText = `Window ${index + 1} (${Math.round(win.window_start_ms / 1000)}s)`;
        windowSelect.appendChild(opt);
    });
    
    // Automatically draw the first window
    onWindowChange();
}

// Re-draw PPG Waveform Plot when selected window changes
function onWindowChange() {
    const subjectId = document.getElementById("select-subject").value;
    const windowIdx = parseInt(document.getElementById("select-window").value);
    
    if (!subjectId || isNaN(windowIdx)) {
        if (waveformChart) waveformChart.destroy();
        return;
    }
    
    const subjectWindows = processedData.filter(d => d.subject_id === subjectId);
    const win = subjectWindows[windowIdx];
    
    if (!win) return;
    
    // Set Label Badge
    let labelName = "Unknown";
    if (win.label >= 0 && win.label < labelNames.length) {
        labelName = labelNames[win.label];
    }
    
    const badge = document.getElementById("subject-label-badge");
    badge.innerText = `Diagnosis Label: ${labelName}`;
    badge.className = `subject-info-badge badge-${labelName.toLowerCase()}`;
    
    // Plot Signals
    const timestamps = win.timestamps_ms.map(t => (t - win.window_start_ms) / 1000.0); // seconds relative to start
    
    const ctx = document.getElementById("ppg-waveform-chart").getContext("2d");
    if (waveformChart) {
        waveformChart.destroy();
    }
    
    const colors = getComputedStyle(document.documentElement);
    const textThemeColor = colors.getPropertyValue("--text-primary").trim();
    const borderThemeColor = colors.getPropertyValue("--border-color").trim();
    const violetColor = colors.getPropertyValue("--color-violet").trim() || '#8b5cf6';
    const cyanColor = colors.getPropertyValue("--color-cyan").trim() || '#06b6d4';
    
    waveformChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [
                {
                    label: 'Finger PPG (Filtered)',
                    data: win.filtered_finger,
                    borderColor: violetColor,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.2,
                    yAxisID: 'y'
                },
                {
                    label: 'Toe PPG (Filtered)',
                    data: win.filtered_toe,
                    borderColor: cyanColor,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.2,
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Time (seconds)',
                        color: textThemeColor,
                        font: { family: 'Inter', weight: '600' }
                    },
                    ticks: {
                        color: textThemeColor,
                        font: { family: 'Inter' }
                    },
                    grid: { color: borderThemeColor }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Filtered Amplitude',
                        color: textThemeColor,
                        font: { family: 'Inter', weight: '600' }
                    },
                    ticks: {
                        color: textThemeColor,
                        font: { family: 'Inter' }
                    },
                    grid: { color: borderThemeColor }
                }
            },
            plugins: {
                legend: {
                    display: false // We use our own customized header legend
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    bodyFont: { family: 'Inter' },
                    titleFont: { family: 'Inter' }
                }
            }
        }
    });
}

// Drag & Drop Upload Handlers
function setupUploadDropzone(type) {
    const dropzone = document.getElementById(`upload-dropzone-${type}`);
    const fileInput = document.getElementById(`dataset-file-input-${type}`);
    
    if (!dropzone || !fileInput) return;
    
    dropzone.addEventListener("click", () => fileInput.click());
    
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });
    
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleSelectedFiles(Array.from(e.dataTransfer.files), type);
        }
    });
    
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleSelectedFiles(Array.from(fileInput.files), type);
        }
    });
}

// Update UI with selected files to upload
function handleSelectedFiles(files, type) {
    selectedUploadFiles[type] = files;
    
    const summary = document.getElementById(`selected-files-summary-${type}`);
    const filesList = document.getElementById(`selected-files-list-${type}`);
    const uploadBtn = document.getElementById(`btn-upload-dataset-${type}`);
    
    if (!summary || !filesList || !uploadBtn) return;
    
    filesList.innerHTML = "";
    
    if (files.length === 0) {
        summary.style.display = "none";
        uploadBtn.disabled = true;
        return;
    }
    
    summary.style.display = "flex";
    uploadBtn.disabled = false;
    
    selectedUploadFiles[type].forEach(file => {
        const item = document.createElement("div");
        item.style.display = "flex";
        item.style.justifyContent = "space-between";
        item.style.color = "var(--text-secondary)";
        item.style.gap = "12px";
        
        let sizeStr = (file.size / 1024).toFixed(1) + " KB";
        if (file.size > 1024 * 1024) {
            sizeStr = (file.size / (1024 * 1024)).toFixed(1) + " MB";
        }
        
        item.innerHTML = `
            <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 200px;" title="${file.name}">${file.name}</span>
            <span>(${sizeStr})</span>
        `;
        filesList.appendChild(item);
    });
}

// Upload selected dataset to server
async function uploadDataset(type) {
    const uploadBtn = document.getElementById(`btn-upload-dataset-${type}`);
    if (!uploadBtn || !selectedUploadFiles[type] || selectedUploadFiles[type].length === 0) return;
    
    const clearCheckbox = document.getElementById(`upload-clear-checkbox-${type}`);
    const clearExisting = clearCheckbox ? clearCheckbox.checked : false;
    
    uploadBtn.disabled = true;
    uploadBtn.innerText = "⏳ Uploading...";
    
    try {
        // Read all files as base64
        const readPromises = selectedUploadFiles[type].map(file => {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    // Extract base64 part
                    const base64Content = reader.result.split(',')[1];
                    resolve({
                        filename: file.name,
                        content: base64Content
                    });
                };
                reader.onerror = () => reject(new Error(`Failed to read file ${file.name}`));
                reader.readAsDataURL(file);
            });
        });
        
        const filesDataPayload = await Promise.all(readPromises);
        
        const res = await fetch("/api/dataset/upload", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                clear_existing: clearExisting,
                files: filesDataPayload
            })
        });
        
        if (res.ok) {
            const result = await res.json();
            
            let message = "";
            if (result.uploaded && result.uploaded.length > 0) {
                message += `Successfully uploaded ${result.uploaded.length} dataset files!\n`;
            }
            if (result.errors && result.errors.length > 0) {
                message += `\nErrors / Warnings:\n` + result.errors.join("\n");
            }
            
            if (result.status === "success") {
                alert(message || "Dataset uploaded successfully!");
                // Clear state
                selectedUploadFiles[type] = [];
                const fileInput = document.getElementById(`dataset-file-input-${type}`);
                if (fileInput) fileInput.value = "";
                handleSelectedFiles([], type);
                // Reload dashboard state
                reloadData();
            } else {
                alert("Upload completed with notes:\n" + message);
                selectedUploadFiles[type] = [];
                const fileInput = document.getElementById(`dataset-file-input-${type}`);
                if (fileInput) fileInput.value = "";
                handleSelectedFiles([], type);
                reloadData();
            }
        } else {
            alert("Failed to upload dataset to the server.");
        }
    } catch (e) {
        console.error("Dataset upload failed:", e);
        alert(`Dataset upload failed: ${e.message}`);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.innerText = "📤 Upload Dataset";
    }
}

// Trigger Pipeline script execution
async function runPipelineStep(stepKey) {
    if (pipelineRunning) return;
    
    setPipelineUIRunning(true);
    
    try {
        const res = await fetch("/api/pipeline/run", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ step: stepKey })
        });
        
        if (res.ok) {
            console.log(`[Dashboard] Pipeline step ${stepKey} started.`);
            // Quick reload of logs
            pollStatusOnce();
        } else {
            const err = await res.json();
            alert(`Error: ${err.error || "Failed to start pipeline script"}`);
            setPipelineUIRunning(false);
        }
    } catch (e) {
        console.error("Failed to run pipeline step:", e);
        setPipelineUIRunning(false);
    }
}

// Trigger Pipeline step from Home/Overview page quick controls
async function runPipelineStepHome(stepKey) {
    // Jump to the pipeline tab so the user can see console logs
    switchTab("pipeline");
    runPipelineStep(stepKey);
}

// Stop current pipeline execution
async function stopPipelineStep() {
    try {
        const res = await fetch("/api/pipeline/stop", {
            method: "POST"
        });
        if (res.ok) {
            console.log("[Dashboard] Request to stop pipeline sent.");
        } else {
            const err = await res.json();
            alert(`Error: ${err.error || "Failed to stop pipeline"}`);
        }
    } catch (e) {
        console.error("Failed to stop pipeline:", e);
    }
}

// Clear synthetic files and output data
async function clearPipelineData() {
    if (!window.location.search.includes('bypass_confirm') && !confirm("Are you sure you want to delete all synthetic data, processed signals, trained model weights, and compiled configurations?")) {
        return;
    }
    
    try {
        const res = await fetch("/api/pipeline/clear_data", {
            method: "POST"
        });
        if (res.ok) {
            const result = await res.json();
            console.log("[Dashboard] Cleanup complete.");
            document.getElementById("pipeline-terminal").innerText = result.logs;
            
            // Reload all dashboard panels
            reloadData();
        }
    } catch (e) {
        console.error("Failed to clear pipeline data:", e);
    }
}

// Clear pipeline from Home/Overview page quick actions
function clearPipelineDataHome() {
    switchTab("pipeline");
    clearPipelineData();
}

// Download raw dataset ZIP
function downloadRawDataset() {
    window.open("/api/download/raw", "_blank");
}

// Download features CSV
function downloadFeaturesDataset() {
    window.open("/data/features.csv", "_blank");
}

// Poll Pipeline state and logs
function startStatusPolling() {
    pollStatusOnce();
    
    // Poll every 1 second
    setInterval(pollStatusOnce, 1000);
}

async function pollStatusOnce() {
    try {
        const res = await fetch("/api/pipeline/status");
        if (res.ok) {
            const data = await res.json();
            updatePipelineUI(data);
        }
    } catch (e) {
        console.error("Error polling pipeline status:", e);
    }
}

// Update pipeline badges, console and header status
function updatePipelineUI(state) {
    const isRunning = state.status === "running";
    pipelineRunning = isRunning;
    
    // 1. Update indicator lights
    const headerIndicator = document.querySelector(".status-indicator");
    const headerStatusText = document.querySelector(".status-text");
    
    if (isRunning) {
        headerIndicator.className = "status-indicator running";
        headerStatusText.innerText = `Running: ${state.current_step}`;
        setPipelineUIRunning(true);
    } else {
        headerIndicator.className = "status-indicator online";
        headerStatusText.innerText = "Pipeline Connected";
        setPipelineUIRunning(false);
    }

    // 2. Update Console logs
    const consoleText = document.getElementById("pipeline-terminal");
    if (consoleText) {
        const wasAtBottom = consoleText.parentElement.scrollHeight - consoleText.parentElement.clientHeight <= consoleText.parentElement.scrollTop + 50;
        consoleText.innerText = state.logs || "> Console loaded. Ready to run scripts...";
        
        // Auto scroll if checked
        const autoScroll = document.getElementById("console-autoscroll");
        if (autoScroll && autoScroll.checked && (isRunning || wasAtBottom)) {
            consoleText.parentElement.scrollTop = consoleText.parentElement.scrollHeight;
        }
    }

    // 3. Update Pipeline manager Badges and Active row classes
    const activeStepIndex = state.active_step_index;
    
    // Standard steps are 0 to 4. Full pipeline is 99. Train sub-pipeline is 98.
    for (let i = 0; i < 5; i++) {
        const item = document.getElementById(`step-item-${i}`);
        const badge = document.getElementById(`badge-step-${i}`);
        const runBtn = document.getElementById(`btn-run-${i}`);
        const stopBtn = document.getElementById(`btn-stop-${i}`);
        
        if (!item || !badge) continue;
        
        if (isRunning) {
            runBtn.disabled = true;
            
            if (activeStepIndex === i || activeStepIndex === 99 || (activeStepIndex === 98 && i >= 1)) {
                item.classList.add("active-step");
                badge.className = "pipeline-badge badge-running";
                badge.innerText = "Running";
                if (stopBtn) stopBtn.style.display = "inline-flex";
            } else {
                item.classList.remove("active-step");
                badge.className = "pipeline-badge badge-idle";
                badge.innerText = "Idle";
                if (stopBtn) stopBtn.style.display = "none";
            }
        } else {
            // Idle state
            item.classList.remove("active-step");
            if (runBtn) runBtn.disabled = false;
            if (stopBtn) stopBtn.style.display = "none";
            
            // Check if pipeline success/failed
            if (state.status === "success") {
                badge.className = "pipeline-badge badge-success";
                badge.innerText = "Success";
            } else if (state.status === "failed") {
                badge.className = "pipeline-badge badge-failed";
                badge.innerText = "Failed";
            } else {
                badge.className = "pipeline-badge badge-idle";
                badge.innerText = "Idle";
            }
        }
    }

    // If training succeeded or stopped, reload metrics to show newest curves
    if (!isRunning && state.status === "success" && state.current_step !== "Cleanup") {
        // Only reload if we transitions from running to idle success
        if (window.lastPipelineState === "running") {
            reloadData();
        }
    }
    
    window.lastPipelineState = state.status;
}

// Toggle Home & Pipeline page buttons running state
function setPipelineUIRunning(isRunning) {
    const homeRunAll = document.getElementById("home-btn-run-all");
    const homeQuickTrain = document.getElementById("home-btn-quick-train");
    const homeQuickSynthetic = document.getElementById("home-btn-quick-synthetic");
    const homeStop = document.getElementById("home-btn-stop");
    const homeClear = document.getElementById("home-btn-clear");

    const pipeRunAll = document.getElementById("btn-run-all");
    const pipeQuickTrain = document.getElementById("btn-quick-train");
    const pipeQuickSynthetic = document.getElementById("btn-quick-synthetic");
    const pipeStop = document.getElementById("btn-stop-all");
    const pipeClear = document.getElementById("btn-clear-pipeline");

    if (isRunning) {
        if (homeRunAll) homeRunAll.style.display = "none";
        if (homeQuickTrain) homeQuickTrain.style.display = "none";
        if (homeQuickSynthetic) homeQuickSynthetic.style.display = "none";
        if (homeClear) homeClear.disabled = true;
        if (homeStop) homeStop.style.display = "inline-flex";

        if (pipeRunAll) pipeRunAll.style.display = "none";
        if (pipeQuickTrain) pipeQuickTrain.style.display = "none";
        if (pipeQuickSynthetic) pipeQuickSynthetic.style.display = "none";
        if (pipeClear) pipeClear.disabled = true;
        if (pipeStop) pipeStop.style.display = "inline-flex";
    } else {
        if (homeRunAll) homeRunAll.style.display = "inline-flex";
        if (homeQuickTrain) homeQuickTrain.style.display = "inline-flex";
        if (homeQuickSynthetic) homeQuickSynthetic.style.display = "inline-flex";
        if (homeClear) homeClear.disabled = false;
        if (homeStop) homeStop.style.display = "none";

        if (pipeRunAll) pipeRunAll.style.display = "inline-flex";
        if (pipeQuickTrain) pipeQuickTrain.style.display = "inline-flex";
        if (pipeQuickSynthetic) pipeQuickSynthetic.style.display = "inline-flex";
        if (pipeClear) pipeClear.disabled = false;
        if (pipeStop) pipeStop.style.display = "none";
    }
}

// Clear Terminal live log visualizer
function clearConsole() {
    document.getElementById("pipeline-terminal").innerText = "> Console cleared.";
}

// Open Code Viewer Modal
async function viewCodeFile(filename) {
    const title = document.getElementById("code-modal-title");
    const body = document.getElementById("code-modal-body");
    const modal = document.getElementById("code-viewer-modal");
    
    if (!title || !body || !modal) return;
    
    title.innerText = `Source Code: ${filename}`;
    body.innerText = "Loading file content...";
    modal.style.display = "flex";
    
    try {
        const res = await fetch(`/api/code?file=${filename}`);
        if (res.ok) {
            body.innerText = await res.text();
        } else {
            const err = await res.json();
            body.innerText = `Error: ${err.error || "Failed to load file contents."}`;
        }
    } catch (e) {
        body.innerText = `Error connecting to API: ${e.message}`;
    }
}

// Close Code Viewer Modal
function closeCodeModal(event) {
    const modal = document.getElementById("code-viewer-modal");
    if (modal) modal.style.display = "none";
}

// Copy source code from Modal to Clipboard
function copyModalCode() {
    const codeText = document.getElementById("code-modal-body").innerText;
    navigator.clipboard.writeText(codeText)
        .then(() => {
            const btn = document.getElementById("btn-copy-code");
            const originalText = btn.innerText;
            btn.innerText = "✓ Copied!";
            setTimeout(() => {
                btn.innerText = originalText;
            }, 2000);
        })
        .catch(err => {
            console.error("Clipboard copy failed:", err);
            alert("Failed to copy code to clipboard.");
        });
}

// Toggle Light / Dark themes
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
    
    const themeBtn = document.getElementById("theme-toggle-btn");
    if (themeBtn) {
        themeBtn.innerText = newTheme === "dark" ? "☀️" : "🌙";
    }
    
    // Re-draw charts with updated theme colors
    calculateOverviewStats();
    onWindowChange();
}

// Synchronize theme icon on load
(function() {
    const savedTheme = localStorage.getItem("theme") || "dark";
    document.addEventListener("DOMContentLoaded", () => {
        const themeBtn = document.getElementById("theme-toggle-btn");
        if (themeBtn) {
            themeBtn.innerText = savedTheme === "dark" ? "☀️" : "🌙";
        }
    });
})();

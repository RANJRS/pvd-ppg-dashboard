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
        esp32: ["ESP32 Acquisition & Live Model Run", "Connect to hardware via Serial or BLE, record signals, and evaluate PVD Risk"],
        features: ["Feature Explorer", "Review extracted vectors sent to training"],
        model: ["Model Evaluation", "Confusion matrix and normalized scaler metrics"],
        pipeline: ["Pipeline Manager", "Run scripts, track execution and view console logs"]
    };
    
    if (titleMap[tabId]) {
        document.getElementById("page-title").innerText = titleMap[tabId][0];
        document.getElementById("page-subtitle").innerText = titleMap[tabId][1];
    }

    if (tabId === "esp32") {
        initEsp32LiveChart();
        loadRawDatasets();
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
        loadModelReport(),
        loadRawDatasets()
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
            const allData = await res.json();
            // Include all valid segmented windows
            processedData = allData.filter(d => d.subject_id);
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
            const allFeatures = parseCSV(text);
            // Include all feature rows
            featureData = allFeatures.filter(row => row.subject_id);
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
        tableBody.innerHTML = `<tr><td colspan="16" class="text-center">No matching features found.</td></tr>`;
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
        
        const fmt = (v, d=4) => parseFloat(v || 0).toFixed(d);
        
        // Display all 13 feature columns
        tr.innerHTML = `
            <td><strong>${row.subject_id || "N/A"}</strong></td>
            <td>${fmt(row.PI_finger)}</td>
            <td>${fmt(row.PI_toe)}</td>
            <td>${fmt(row.PI_ratio)}</td>
            <td>${fmt(row.PTT_ft, 1)}</td>
            <td>${fmt(row.AI_finger, 2)}</td>
            <td>${fmt(row.AI_toe, 2)}</td>
            <td>${fmt(row.HRV_rmssd, 2)}</td>
            <td>${fmt(row.dicrotic_ratio)}</td>
            <td style="color:var(--color-amber,#f59e0b);">${fmt(row.temp_finger, 2)}</td>
            <td style="color:var(--color-cyan);">${fmt(row.temp_toe, 2)}</td>
            <td style="color:var(--color-rose);">${fmt(row.temp_diff, 2)}</td>
            <td style="color:var(--color-violet);">${fmt(row.accel_std, 4)}</td>
            <td style="color:var(--color-emerald,#10b981);">${fmt(row.gyro_std, 4)}</td>
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
                resetModelMetrics();
                return;
            }
            
            // Populate Metrics summary
            const accuracyPct = report.accuracy !== undefined ? (report.accuracy * 100).toFixed(1) + "%" : "0%";
            document.getElementById("model-accuracy-val").innerText = accuracyPct;
            document.getElementById("model-epochs-val").innerText = report.epochs_run || "N/A";
            document.getElementById("model-samples-val").innerText = `${report.train_samples || 0} / ${report.test_samples || 0}`;
            document.getElementById("model-total-subjects-val").innerText = report.total_subjects || "N/A";
            
            // Also update the Model Accuracy card on the Overview tab
            const homeAccuracy = document.getElementById("home-stat-accuracy");
            if (homeAccuracy) {
                homeAccuracy.innerText = accuracyPct;
            }
            
            // Populate detailed class metrics
            if (report.per_class) {
                const classes = ["Normal", "Moderate", "High"];
                classes.forEach(cls => {
                    const stats = report.per_class[cls];
                    const idPrefix = `class-${cls.toLowerCase()}`;
                    if (stats) {
                        document.getElementById(`${idPrefix}-precision`).innerText = stats.precision.toFixed(3);
                        document.getElementById(`${idPrefix}-recall`).innerText = stats.recall.toFixed(3);
                        document.getElementById(`${idPrefix}-f1`).innerText = stats.f1.toFixed(3);
                    } else {
                        document.getElementById(`${idPrefix}-precision`).innerText = "-";
                        document.getElementById(`${idPrefix}-recall`).innerText = "-";
                        document.getElementById(`${idPrefix}-f1`).innerText = "-";
                    }
                });
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
            resetModelMetrics();
        }
    } catch (e) {
        console.error("Failed to load model report:", e);
        resetConfusionMatrix();
        resetModelMetrics();
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

function resetModelMetrics() {
    document.getElementById("model-accuracy-val").innerText = "-%";
    document.getElementById("model-epochs-val").innerText = "-";
    document.getElementById("model-samples-val").innerText = "- / -";
    document.getElementById("model-total-subjects-val").innerText = "-";
    
    const homeAccuracy = document.getElementById("home-stat-accuracy");
    if (homeAccuracy) {
        homeAccuracy.innerText = "--%";
    }
    
    const classes = ["normal", "moderate", "high"];
    classes.forEach(cls => {
        document.getElementById(`class-${cls}-precision`).innerText = "-";
        document.getElementById(`class-${cls}-recall`).innerText = "-";
        document.getElementById(`class-${cls}-f1`).innerText = "-";
    });
}

// Calculate Summary Statistics for Overview
function calculateOverviewStats() {
    // Prefer processedData windows, fallback to featureData rows if processed.json is not present
    const dataSource = processedData.length > 0 ? processedData : featureData;
    
    if (dataSource.length === 0) {
        document.getElementById("stat-subjects").innerText = "0";
        document.getElementById("stat-windows").innerText = "0";
        document.getElementById("stat-normal").innerText = "0";
        document.getElementById("stat-moderate-high").innerText = "0";
        
        drawClassDistChart(0, 0, 0);
        return;
    }
    
    // 1. Total window segments
    const totalWindows = dataSource.length;
    document.getElementById("stat-windows").innerText = totalWindows;
    
    // 2. Total subjects
    const subjects = [...new Set(dataSource.map(d => d.subject_id))];
    document.getElementById("stat-subjects").innerText = subjects.length;
    
    // 3. Counts by class (normal=0, moderate=1, high=2)
    let normalCount = 0;
    let moderateCount = 0;
    let highCount = 0;
    
    dataSource.forEach(d => {
        const lbl = parseInt(d.label);
        if (lbl === 0) normalCount++;
        else if (lbl === 1) moderateCount++;
        else if (lbl === 2) highCount++;
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
    if (!window.location.search.includes('bypass_confirm')) {
        // Step 1: Standard confirmation dialog
        const confirmed = confirm("Are you sure you want to delete all synthetic data, processed signals, trained model weights, and compiled configurations?");
        if (!confirmed) {
            return;
        }
        
        // Step 2: Verification prompt where user must type 'DELETE'
        const verification = prompt("This action is destructive and cannot be undone. To proceed, please type 'DELETE' in the box below:");
        if (verification !== "DELETE") {
            alert("Verification failed. The clear action was cancelled.");
            return;
        }
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
            reloadData().then(() => {
                // If full pipeline completed, auto-switch to overview to show extracted metrics
                if (state.current_step === "Full Pipeline" || state.active_step_index === -1) {
                    switchTab("overview");
                }
            });
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

// ============================================================================
// ESP32 Acquisition & Real-Time Model Inference Logic
// ============================================================================

// Global variables for ESP32 Connection and Telemetry
let esp32LiveChart = null;
let esp32SerialPort = null;
let esp32SerialReader = null;
let esp32BleDevice = null;
let esp32BleStreamChar = null;
let esp32BleRxChar = null;
let esp32BleFeatureChar = null;
let esp32BleRiskChar = null;
let bleTextBuffer = "";
let esp32SimInterval = null;
let isRecording = false;
let recordedSamples = []; // [{timestamp_ms, ir_finger, ir_toe}, ...]
let recordStartTime = 0;
let recordTimerInterval = null;
let packetCount = 0;
let lastPacketCount = 0;
let dataRateInterval = null;

let liveChartDataFinger = [];
let liveChartDataToe = [];
let liveChartTimestamps = [];

// Initialize Live Scrolling Chart
function initEsp32LiveChart() {
    const ctx = document.getElementById("esp32-live-chart");
    if (!ctx) return;
    
    if (esp32LiveChart) {
        return; // already initialized
    }
    
    const colors = getComputedStyle(document.documentElement);
    const textThemeColor = colors.getPropertyValue("--text-primary").trim();
    const borderThemeColor = colors.getPropertyValue("--border-color").trim();
    const violetColor = colors.getPropertyValue("--color-violet").trim() || '#8b5cf6';
    const cyanColor = colors.getPropertyValue("--color-cyan").trim() || '#06b6d4';
    
    // Initialize empty buffers (starts cleanly when data arrives)
    liveChartTimestamps = [];
    liveChartDataFinger = [];
    liveChartDataToe = [];
    
    esp32LiveChart = new Chart(ctx.getContext("2d"), {
        type: 'line',
        data: {
            labels: liveChartTimestamps,
            datasets: [
                {
                    label: 'Finger PPG (Live)',
                    data: liveChartDataFinger,
                    borderColor: violetColor,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.2,
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Relative Time (seconds)',
                        color: textThemeColor,
                        font: { family: 'Inter', weight: '600' }
                    },
                    ticks: {
                        color: textThemeColor,
                        font: { family: 'Inter' },
                        callback: function(value) { return value.toFixed(1) + 's'; }
                    },
                    grid: { color: borderThemeColor },
                    min: -4.0,
                    max: 0.0
                },
                y: {
                    title: {
                        display: true,
                        text: 'Raw Amplitude',
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
                legend: { display: false },
                tooltip: { enabled: false }
            }
        }
    });
}

// Connect to ESP32 (USB, BLE, or Simulation)
async function connectESP32() {
    const isSimMode = document.getElementById("esp32-sim-checkbox")?.checked;
    
    // Explicitly check selected interface radio button by ID
    const bleRadio = document.getElementById("conn-method-ble");
    const serialRadio = document.getElementById("conn-method-serial");
    
    let connMethod = "serial";
    if (bleRadio && bleRadio.checked) {
        connMethod = "ble";
    } else if (serialRadio && serialRadio.checked) {
        connMethod = "serial";
    } else {
        const checkedInput = document.querySelector('input[name="conn-method"]:checked');
        if (checkedInput) connMethod = checkedInput.value;
    }
    
    console.log(`[ESP32 Connection] Target Interface selected: ${connMethod.toUpperCase()} (SimMode: ${isSimMode})`);
    
    disconnectESP32(); // Clear any existing connection
    
    packetCount = 0;
    lastPacketCount = 0;
    
    if (isSimMode) {
        console.log("[ESP32] Connecting in Simulation Mode...");
        setEsp32ConnectedState("sim");
        
        let simClass = document.getElementById("rec-label").value;
        if (simClass === "random") {
            const classes = ["normal", "moderate", "high"];
            simClass = classes[Math.floor(Math.random() * classes.length)];
            document.getElementById("rec-label").value = simClass;
            console.log(`[ESP32 Sim] Chosen random class: ${simClass}`);
        }
        
        // Start 200 Hz simulation stream
        let simTime = 0;
        const heartRateHz = 1.25; // 75 bpm (inside 60-90 bpm range)

        esp32SimInterval = setInterval(() => {
            let activeClass = document.getElementById("rec-label").value;
            if (activeClass === "random") {
                activeClass = simClass;
            }
            
            // Define amplitude and PTT parameters based on risk class, matching generate_synthetic_data.py
            let A_toe = 125.0;  // normal default
            let ptt = 0.12;     // normal default (120ms)
            
            if (activeClass === "moderate") {
                A_toe = 78.0;   // moderate default (78 amplitude)
                ptt = 0.20;     // moderate default (200ms)
            } else if (activeClass === "high") {
                A_toe = 40.0;   // high default (40 amplitude)
                ptt = 0.28;     // high default (280ms)
            }
            
            // Finger signal: baseline 1000 + sine fundamentals + noise
            const rawFinger = 1000.0 + 200.0 * Math.sin(2 * Math.PI * heartRateHz * simTime) + 
                              60.0 * Math.sin(4 * Math.PI * heartRateHz * simTime + 0.8) + 
                              (Math.random() - 0.5) * 5.0;
                              
            // Toe signal: baseline 800 + sine fundamentals shifted by PTT + noise
            const rawToe = 800.0 + A_toe * Math.sin(2 * Math.PI * heartRateHz * (simTime - ptt)) + 
                            (A_toe * 0.3) * Math.sin(4 * Math.PI * heartRateHz * (simTime - ptt) + 0.8) + 
                            (Math.random() - 0.5) * 3.0;

            // Timestamp in milliseconds starting from 0 (perfect regular sampling)
            const tsMs = Math.round(simTime * 1000);
            
            // Pass full sensor data (PPG + Temp + IMU) to sample handler
            handleIncomingSample(tsMs, rawFinger, rawToe);
            simTime += 0.005; // 200 Hz (5ms steps)
        }, 5);
        return;
    }
    
    if (connMethod === "serial") {
        if (!("serial" in navigator)) {
            alert("Web Serial API is not supported by your browser. Please use Chrome, Edge, or Opera.");
            return;
        }
        
        const baudRateSelect = document.getElementById("esp32-baud-rate");
        const selectedBaud = baudRateSelect ? parseInt(baudRateSelect.value) : 115200;
        
        console.log(`[ESP32] Connecting via Web Serial at ${selectedBaud} baud...`);
        try {
            esp32SerialPort = await navigator.serial.requestPort();
            await esp32SerialPort.open({ baudRate: selectedBaud });
            setEsp32ConnectedState("usb");
            
            // Start asynchronous read loop
            readSerialStream();
        } catch (e) {
            console.error("Web Serial connection failed:", e);
            alert(`Failed to connect to Serial Port: ${e.message}`);
            disconnectESP32();
        }
    } else if (connMethod === "ble") {
        if (!("bluetooth" in navigator)) {
            alert("Web Bluetooth API is not supported by your browser or is disabled. Please open this dashboard in Google Chrome, Microsoft Edge, or a Web Bluetooth compatible browser over HTTPS.");
            return;
        }
        
        console.log("[ESP32] Connecting via Web Bluetooth...");
        try {
            esp32BleDevice = await navigator.bluetooth.requestDevice({
                filters: [
                    { name: "PVD-Patch" },
                    { namePrefix: "PVD" },
                    { namePrefix: "ESP32" }
                ],
                optionalServices: ["6e400001-b5a3-f393-e0a9-e50e24dcca9e"]
            });
            
            console.log("[ESP32] Selected BLE device:", esp32BleDevice.name || "PVD-Patch");
            
            esp32BleDevice.addEventListener('gattserverdisconnected', () => {
                console.warn("[ESP32] BLE connection closed by peripheral.");
                disconnectESP32();
            });

            const server = await esp32BleDevice.gatt.connect();
            const service = await server.getPrimaryService("6e400001-b5a3-f393-e0a9-e50e24dcca9e");
            
            // 1. Subscribe to Live Data Stream / TX Characteristic (UUID: ...0003)
            try {
                esp32BleStreamChar = await service.getCharacteristic("6e400003-b5a3-f393-e0a9-e50e24dcca9e");
                await esp32BleStreamChar.startNotifications();
                esp32BleStreamChar.addEventListener('characteristicvaluechanged', handleBleStreamChanged);
                console.log("[ESP32] Subscribed to BLE TX stream (6e400003).");
            } catch (txErr) {
                console.warn("[ESP32] Primary TX characteristic error, attempting fallback:", txErr);
                esp32BleStreamChar = await service.getCharacteristic("6e400002-b5a3-f393-e0a9-e50e24dcca9e");
                await esp32BleStreamChar.startNotifications();
                esp32BleStreamChar.addEventListener('characteristicvaluechanged', handleBleStreamChanged);
            }

            // 2. Optional RX Characteristic for device control (UUID: ...0002)
            try {
                esp32BleRxChar = await service.getCharacteristic("6e400002-b5a3-f393-e0a9-e50e24dcca9e");
            } catch (rxErr) {
                esp32BleRxChar = null;
            }
            
            setEsp32ConnectedState("ble");
            console.log("[ESP32] BLE connected and live stream active.");
        } catch (e) {
            console.error("Web Bluetooth connection failed:", e);
            if (e.name !== "NotFoundError") {
                alert(`Bluetooth Connection failed: ${e.message}`);
            }
            disconnectESP32();
        }
    }
}

// Handle incoming BLE telemetry packets
function handleBleStreamChanged(event) {
    try {
        const val = event.target.value;
        const decoder = new TextDecoder("utf-8");
        const chunk = decoder.decode(val);
        bleTextBuffer += chunk;
        
        let lines = bleTextBuffer.split("\n");
        bleTextBuffer = lines.pop(); // keep trailing incomplete segment
        
        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed) {
                parseSerialLine(trimmed);
            }
        }
    } catch (err) {
        console.error("[ESP32] Error processing BLE data:", err);
    }
}

// Disconnect ESP32
function disconnectESP32() {
    // 1. Clear simulation
    if (esp32SimInterval) {
        clearInterval(esp32SimInterval);
        esp32SimInterval = null;
    }
    
    // 2. Clear serial
    if (esp32SerialReader) {
        try { esp32SerialReader.cancel(); } catch(e) {}
        esp32SerialReader = null;
    }
    if (esp32SerialPort) {
        try { esp32SerialPort.close(); } catch(e) {}
        esp32SerialPort = null;
    }
    
    // 3. Clear BLE
    if (esp32BleStreamChar) {
        try { esp32BleStreamChar.stopNotifications(); } catch(e) {}
        esp32BleStreamChar = null;
    }
    if (esp32BleFeatureChar) {
        try { esp32BleFeatureChar.stopNotifications(); } catch(e) {}
        esp32BleFeatureChar = null;
    }
    if (esp32BleRiskChar) {
        try { esp32BleRiskChar.stopNotifications(); } catch(e) {}
        esp32BleRiskChar = null;
    }
    esp32BleRxChar = null;
    bleTextBuffer = "";
    if (esp32BleDevice && esp32BleDevice.gatt && esp32BleDevice.gatt.connected) {
        try { esp32BleDevice.gatt.disconnect(); } catch(e) {}
    }
    esp32BleDevice = null;
    
    // 4. Stop recording
    if (isRecording) {
        stopRecording();
    }
    
    // 5. Clear intervals
    if (dataRateInterval) {
        clearInterval(dataRateInterval);
        dataRateInterval = null;
    }
    
    // 6. Reset UI
    const statusBadge = document.getElementById("esp32-status-badge");
    if (statusBadge) {
        statusBadge.className = "pipeline-badge badge-idle";
        statusBadge.innerText = "Disconnected";
    }
    
    const packetText = document.getElementById("esp32-packet-count");
    if (packetText) packetText.innerText = "0";
    
    const dataRateText = document.getElementById("esp32-data-rate");
    if (dataRateText) dataRateText.innerText = "--";
    
    const gyroText = document.getElementById("esp32-imu-gyro");
    if (gyroText) gyroText.innerText = "--";
    
    const connectBtn = document.getElementById("btn-esp32-connect");
    if (connectBtn) connectBtn.disabled = false;
    
    const disconnectBtn = document.getElementById("btn-esp32-disconnect");
    if (disconnectBtn) disconnectBtn.disabled = true;
    
    const startRecBtn = document.getElementById("btn-rec-start");
    if (startRecBtn) startRecBtn.disabled = true;
    
    const overlay = document.getElementById("chart-overlay-message");
    if (overlay) {
        overlay.style.opacity = "1";
        overlay.style.pointerEvents = "all";
    }
    
    document.querySelectorAll('input[name="conn-method"]').forEach(r => r.disabled = false);
    document.getElementById("esp32-sim-checkbox").disabled = false;
}

// Update UI when connection is successful
function setEsp32ConnectedState(mode) {
    const statusBadge = document.getElementById("esp32-status-badge");
    let modeText = "Connected";
    if (mode === "sim") modeText = "Connected (Sim)";
    if (mode === "usb") modeText = "Connected (USB)";
    if (mode === "ble") modeText = "Connected (BLE)";
    
    if (statusBadge) {
        statusBadge.className = "pipeline-badge badge-connected";
        statusBadge.innerText = modeText;
    }
    
    const connectBtn = document.getElementById("btn-esp32-connect");
    if (connectBtn) connectBtn.disabled = true;
    
    const disconnectBtn = document.getElementById("btn-esp32-disconnect");
    if (disconnectBtn) disconnectBtn.disabled = false;
    
    const startRecBtn = document.getElementById("btn-rec-start");
    if (startRecBtn) startRecBtn.disabled = false;
    
    liveChartTimestamps = [];
    liveChartDataFinger = [];
    latestGyroData = { gx: 0, gy: 0, gz: 0, valid: false };
    
    const overlay = document.getElementById("chart-overlay-message");
    if (overlay) {
        overlay.style.opacity = "0";
        overlay.style.pointerEvents = "none";
    }
    
    document.querySelectorAll('input[name="conn-method"]').forEach(r => r.disabled = true);
    document.getElementById("esp32-sim-checkbox").disabled = true;
    
    // Packet rate monitoring loop
    dataRateInterval = setInterval(() => {
        const delta = packetCount - lastPacketCount;
        lastPacketCount = packetCount;
        
        const rateText = document.getElementById("esp32-data-rate");
        if (rateText) {
            if (mode === "ble") {
                rateText.innerText = "BLE Notifications Active";
            } else {
                rateText.innerText = `${delta} samples/sec`;
            }
        }
    }, 1000);
}

// Read loop for Web Serial
async function readSerialStream() {
    const textDecoder = new TextDecoderStream();
    const readableStreamClosed = esp32SerialPort.readable.pipeTo(textDecoder.writable);
    const reader = textDecoder.readable.getReader();
    esp32SerialReader = reader;
    
    let lineBuffer = "";
    
    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            lineBuffer += value;
            let lines = lineBuffer.split("\n");
            lineBuffer = lines.pop(); // save incomplete line
            
            for (let line of lines) {
                parseSerialLine(line.trim());
            }
        }
    } catch (e) {
        console.error("[ESP32] Serial read failed:", e);
    } finally {
        reader.releaseLock();
    }
}

// Parse Serial line
// Format: "DATA,ts,ir_finger,ir_toe"
function parseSerialLine(line) {
    if (!line) return;
    
    if (line.startsWith("DATA,")) {
        const parts = line.split(",");
        if (parts.length >= 3) {
            const ts     = parseInt(parts[1]);
            const finger = parseFloat(parts[2]);
            const toe    = parts.length >= 4 ? parseFloat(parts[3]) : finger;
            if (!isNaN(ts) && !isNaN(finger)) {
                handleIncomingSample(ts, finger, toe);
            }
            
            // Parse Gyro (gx, gy, gz) if provided
            let gx, gy, gz;
            if (parts.length >= 6) {
                gx = parseFloat(parts[3]);
                gy = parseFloat(parts[4]);
                gz = parseFloat(parts[5]);
            }
            if (gx !== undefined && !isNaN(gx) && !isNaN(gy) && !isNaN(gz)) {
                latestGyroData = { gx, gy, gz, valid: true };
            }
        }
        return;
    }
    
    // Legacy format 2: "1000,1250.5,850.2" (Direct raw CSV line)
    const parts = line.split(",");
    if (parts.length >= 3) {
        const ts     = parseInt(parts[0]);
        const finger = parseFloat(parts[1]);
        const toe    = parseFloat(parts[2]);
        if (!isNaN(ts) && !isNaN(finger) && !isNaN(toe)) {
            handleIncomingSample(ts, finger, toe);
        }
    }
}

let liveChartRenderPending = false;
let latestGyroData = { gx: 0, gy: 0, gz: 0, valid: false };

// Handle Incoming sample (simulated or serial)
function handleIncomingSample(ts, finger, toe) {
    packetCount++;
    
    // Push directly to rolling buffers
    liveChartDataFinger.push(finger);
    liveChartTimestamps.push(ts / 1000.0);
    
    // Keep last 3 seconds (600 samples at 200 Hz)
    const samplesToKeep = 600;
    if (liveChartDataFinger.length > samplesToKeep) {
        liveChartDataFinger.shift();
        liveChartTimestamps.shift();
    }
    
    // If recording, store sample
    if (isRecording) {
        recordedSamples.push({
            timestamp_ms: ts,
            ir_finger:    finger,
            ir_toe:       toe
        });
    }
    
    // Schedule decoupled 60 FPS animation render
    if (!liveChartRenderPending) {
        liveChartRenderPending = true;
        requestAnimationFrame(renderLiveChart);
    }
}

// Render chart smoothly at monitor refresh rate (prevents 200/sec browser lag)
function renderLiveChart() {
    liveChartRenderPending = false;
    if (!esp32LiveChart) return;
    
    const len = liveChartTimestamps.length;
    if (len === 0) return;
    
    // 1. Throttled DOM updates
    const packetText = document.getElementById("esp32-packet-count");
    if (packetText) packetText.innerText = packetCount;
    
    if (latestGyroData.valid) {
        const gyroEl = document.getElementById("esp32-imu-gyro");
        if (gyroEl) {
            gyroEl.innerText = `${latestGyroData.gx.toFixed(1)}, ${latestGyroData.gy.toFixed(1)}, ${latestGyroData.gz.toFixed(1)} °/s`;
        }
    }
    
    // 2. Relative timeline X-axis
    const nowS = liveChartTimestamps[len - 1];
    const relativeTimes = liveChartTimestamps.map(t => t - nowS);
    
    esp32LiveChart.data.labels = relativeTimes;
    esp32LiveChart.data.datasets[0].data = liveChartDataFinger;
    
    // 3. Fast O(N) min/max calculation without stack spread
    if (liveChartDataFinger.length > 20) {
        let minVal = Infinity;
        let maxVal = -Infinity;
        for (let i = 0; i < liveChartDataFinger.length; i++) {
            const v = liveChartDataFinger[i];
            if (v !== null && !isNaN(v)) {
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }
        }
        if (minVal !== Infinity && maxVal !== -Infinity && minVal < maxVal) {
            const pad = (maxVal - minVal) * 0.1 || 10;
            esp32LiveChart.options.scales.y.min = Math.floor(minVal - pad);
            esp32LiveChart.options.scales.y.max = Math.ceil(maxVal + pad);
        }
    }
    
    esp32LiveChart.update('none');
}

// Download currently recorded samples in memory as a CSV file
function downloadRecordedCSV() {
    if (!recordedSamples || recordedSamples.length === 0) {
        alert("No recorded samples available to download. Please start a recording first.");
        return;
    }

    let csvString = "timestamp_ms,ir_finger,ir_toe\n";
    recordedSamples.forEach(s => {
        csvString += `${s.timestamp_ms},${Math.round(s.ir_finger)},${Math.round(s.ir_toe)}\n`;
    });
    
    let subjectId = document.getElementById("rec-subject-id").value.trim();
    if (!subjectId) subjectId = "ESP32-Subject";
    const label = document.getElementById("rec-label").value;
    const filename = `${subjectId}_${label}.csv`;
    
    const blob = new Blob([csvString], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    console.log(`[ESP32] Dataset downloaded locally: ${filename}`);
}

// Start Acquisition recording
function startRecording() {
    isRecording = true;
    recordedSamples = [];
    recordStartTime = Date.now();
    
    // Update Badge State to blink recording
    const statusBadge = document.getElementById("esp32-status-badge");
    if (statusBadge) {
        statusBadge.className = "pipeline-badge badge-recording";
        statusBadge.innerText = "Recording";
    }
    
    document.getElementById("btn-rec-start").disabled = true;
    document.getElementById("btn-rec-stop").disabled = false;
    
    const downloadBtn = document.getElementById("btn-rec-download");
    if (downloadBtn) downloadBtn.disabled = true;
    
    let elapsed = 0;
    document.getElementById("rec-count").innerText = "0";
    
    recordTimerInterval = setInterval(() => {
        elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
        document.getElementById("rec-count").innerText = elapsed;
        
        // Auto-stop exactly after 15 seconds
        if (elapsed >= 15) {
            stopRecording();
        }
    }, 1000);
}

// Stop Acquisition recording & run prediction
async function stopRecording() {
    isRecording = false;
    if (recordTimerInterval) {
        clearInterval(recordTimerInterval);
        recordTimerInterval = null;
    }
    
    document.getElementById("btn-rec-start").disabled = false;
    document.getElementById("btn-rec-stop").disabled = true;
    
    const downloadBtn = document.getElementById("btn-rec-download");
    if (downloadBtn && recordedSamples.length > 0) {
        downloadBtn.disabled = false;
    }
    
    // Reset Badge state to connected
    const isSimMode = document.getElementById("esp32-sim-checkbox").checked;
    setEsp32ConnectedState(isSimMode ? "sim" : (esp32SerialPort ? "usb" : "ble"));
    
    console.log(`[ESP32] Recording stopped. Captured ${recordedSamples.length} samples.`);
    
    if (recordedSamples.length < 1600) {
        alert("Recording is too short! Please record at least 8 seconds of data to run the 8s sliding window MLP classifier.");
        return;
    }
    
    // Prompt to run inference
    runModelOnRecording();
}

// Run Keras Model on the recorded CSV data via Backend
async function runModelOnRecording() {
    let subjectId = document.getElementById("rec-subject-id").value.trim();
    if (!subjectId) subjectId = "ESP32-Subject";
    
    const label = document.getElementById("rec-label").value;

    let csvString = "timestamp_ms,ir_finger,ir_toe\n";
    recordedSamples.forEach(s => {
        csvString += `${s.timestamp_ms},${Math.round(s.ir_finger)},${Math.round(s.ir_toe)}\n`;
    });
    
    console.log(`[ESP32] Requesting inference on ${recordedSamples.length} lines of data...`);
    
    const startBtn = document.getElementById("btn-rec-start");
    startBtn.disabled = true;
    startBtn.innerText = "⏳ Running Model...";
    
    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                csv_data: csvString,
                subject_id: subjectId,
                clinician_label: label,
                save_dataset: true
            })
        });
        
        if (res.ok) {
            const report = await res.json();
            console.log("[ESP32] Inference success:", report);
            displayPredictionReport(report);
            loadRawDatasets();
        } else {
            const err = await res.json();
            alert(`Model run failed: ${err.error || "Unknown server error"}`);
        }
    } catch (e) {
        console.error("Prediction request failed:", e);
        alert(`Failed to run prediction: ${e.message}`);
    } finally {
        startBtn.disabled = false;
        startBtn.innerText = "🔴 Start Rec";
    }
}

// Display Prediction Report
function displayPredictionReport(report) {
    if (report.status !== "success") return;
    
    // Show Results Panel
    document.getElementById("esp32-results-grid").style.display = "grid";
    
    // 1. Calculate Average Features across all windows
    const windows = report.windows;
    const numWin = windows.length;
    
    let sumPif=0, sumPit=0, sumPir=0, sumPtt=0;
    let sumAif=0, sumAit=0, sumHrv=0, sumDicr=0;
    
    windows.forEach(w => {
        const f = w.features || {};
        sumPif   += f.PI_finger       || 0;
        sumPit   += f.PI_toe          || 0;
        sumPir   += f.PI_ratio        || 0;
        sumPtt   += f.PTT_ft          || 0;
        sumAif   += f.AI_finger       || 0;
        sumAit   += f.AI_toe          || 0;
        sumHrv   += f.HRV_rmssd       || 0;
        sumDicr  += f.dicrotic_ratio  || 0;
    });
    
    // PPG features
    document.getElementById("res-feat-pif").innerText  = (sumPif   / numWin).toFixed(2) + "%";
    document.getElementById("res-feat-pit").innerText  = (sumPit   / numWin).toFixed(2) + "%";
    document.getElementById("res-feat-pir").innerText  = (sumPir   / numWin).toFixed(3);
    document.getElementById("res-feat-ptt").innerText  = (sumPtt   / numWin).toFixed(1) + " ms";
    document.getElementById("res-feat-aif").innerText  = (sumAif   / numWin).toFixed(1) + "%";
    document.getElementById("res-feat-ait").innerText  = (sumAit   / numWin).toFixed(1) + "%";
    document.getElementById("res-feat-hrv").innerText  = (sumHrv   / numWin).toFixed(1) + " ms";
    document.getElementById("res-feat-dicr").innerText = (sumDicr  / numWin).toFixed(3);
    
    // 2. Draw Risk Gauge
    const finalRiskVal = report.summary.final_risk_level;
    const finalRiskName = report.summary.final_risk_name;
    const avgProbs = report.summary.average_probabilities;
    
    // Risk score representation: weighted sum of moderate (50%) and high (100%) probabilities
    const riskScore = (avgProbs[1] * 0.5 + avgProbs[2] * 1.0) * 100;
    
    drawRiskGauge(riskScore, finalRiskName);

    // Update Expected Output vs Predicted Output
    const recLabelElem = document.getElementById("rec-label");
    const rawExpected = recLabelElem ? recLabelElem.value : "auto";
    let expectedLabel = report.summary.expected_label;
    if (!expectedLabel || rawExpected !== "auto") {
        if (rawExpected === "normal") expectedLabel = "Normal";
        else if (rawExpected === "moderate") expectedLabel = "Moderate";
        else if (rawExpected === "high") expectedLabel = "High";
        else expectedLabel = finalRiskName;
    }

    const badgeExpected = document.getElementById("esp32-expected-badge");
    const compExpected = document.getElementById("esp32-comp-expected");
    const compPredicted = document.getElementById("esp32-comp-predicted");
    const compMatch = document.getElementById("esp32-comp-match");

    if (badgeExpected) badgeExpected.innerText = `Expected: ${expectedLabel}`;
    if (compExpected) {
        compExpected.innerText = expectedLabel;
        if (expectedLabel.toLowerCase() === "normal") compExpected.style.color = "var(--color-emerald)";
        else if (expectedLabel.toLowerCase() === "moderate") compExpected.style.color = "var(--color-amber)";
        else compExpected.style.color = "var(--color-rose)";
    }
    if (compPredicted) {
        compPredicted.innerText = finalRiskName;
        if (finalRiskName.toLowerCase() === "normal") compPredicted.style.color = "var(--color-emerald)";
        else if (finalRiskName.toLowerCase() === "moderate") compPredicted.style.color = "var(--color-amber)";
        else compPredicted.style.color = "var(--color-rose)";
    }
    if (compMatch) {
        const isMatch = expectedLabel.toLowerCase() === finalRiskName.toLowerCase();
        compMatch.innerText = isMatch ? "✅ Match" : "⚠️ Discrepancy";
        compMatch.className = `pipeline-badge ${isMatch ? "badge-normal" : "badge-moderate"}`;
    }
    
    // 3. Save status
    const saveStatus = document.getElementById("res-save-status");
    const saveFile = document.getElementById("res-save-filename");
    const downloadContainer = document.getElementById("res-download-container");
    const downloadLink = document.getElementById("lnk-res-download");
    
    if (report.summary.saved) {
        saveStatus.innerText = "Yes (Saved & Pooled)";
        saveStatus.style.color = "var(--color-emerald)";
        saveFile.innerText = `Filename: ${report.summary.saved_filename}`;
        
        if (downloadContainer && downloadLink) {
            downloadContainer.style.display = "flex";
            downloadLink.href = `/data/raw/${report.summary.saved_filename}`;
            downloadLink.download = report.summary.saved_filename;
        }
    } else {
        saveStatus.innerText = "No";
        saveStatus.style.color = "var(--color-rose)";
        saveFile.innerText = "";
        
        if (downloadContainer) {
            downloadContainer.style.display = "none";
        }
    }
}

// Semicircular Canvas Gauge drawing
function drawRiskGauge(riskScore, riskName) {
    const canvas = document.getElementById("esp32-risk-gauge");
    if (!canvas) return;
    
    // Set device pixel ratio scaling for crisp canvas renderings
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 220 * dpr;
    canvas.height = 120 * dpr;
    canvas.style.width = "220px";
    canvas.style.height = "120px";
    
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    
    ctx.clearRect(0, 0, 220, 120);
    
    const cx = 110;
    const cy = 110;
    const r = 85;
    
    // 1. Draw Gray Background Track
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI, false);
    ctx.lineWidth = 14;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.stroke();
    
    // 2. Draw Color Fill Track
    let color = "#10b981"; // Normal (emerald)
    if (riskName === "Moderate") color = "#f59e0b"; // Moderate (amber)
    else if (riskName === "High") color = "#f43f5e"; // High (rose)
    
    const endAngle = Math.PI + (riskScore / 100.0) * Math.PI;
    
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, endAngle, false);
    ctx.lineWidth = 14;
    ctx.strokeStyle = color;
    ctx.stroke();
    
    // 3. Update Text Label states
    document.getElementById("esp32-risk-pct").innerText = `${Math.round(riskScore)}%`;
    const badge = document.getElementById("esp32-result-risk-name");
    badge.innerText = riskName;
    badge.className = `pipeline-badge badge-${riskName.toLowerCase()}`;
    
    // Description text updates
    const desc = document.getElementById("esp32-result-desc");
    if (riskName === "Normal") {
        desc.innerText = "No indicators of Peripheral Vascular Disease detected. Heart rates, perfusion amplitudes and pulse transit latencies are within healthy clinical norms.";
    } else if (riskName === "Moderate") {
        desc.innerText = "Moderate signs of arterial stiffness, dicrotic notch dampening, or pulse wave delay. Standard checks and routine vascular monitoring suggested.";
    } else if (riskName === "High") {
        desc.innerText = "Substantial markers of Peripheral Vascular Disease. High pulse transit latency (PTT) and decreased perfusion. Early medical consultation is advised.";
    }
}

// Load the raw datasets list from the server and render the table
async function loadRawDatasets() {
    const listContainer = document.getElementById("raw-datasets-list");
    if (!listContainer) return;
    
    try {
        const res = await fetch("/api/raw/list");
        if (!res.ok) throw new Error("Failed to load raw datasets list");
        
        const allDatasets = await res.json();
        // Include all raw datasets
        const datasets = allDatasets.filter(d => d.subject_id);
        
        if (datasets.length === 0) {
            listContainer.innerHTML = `
                <tr>
                    <td colspan="8" style="text-align: center; color: var(--text-secondary); padding: 24px;">
                        No raw acquired datasets found in <code>data/raw/</code>. Record data or upload CSVs above.
                    </td>
                </tr>
            `;
            updateSelectedDatasetsUI();
            return;
        }
        
        // Sort datasets by timestamp descending (newest captured first!)
        datasets.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
        
        // Helper to convert UTC/epoch timestamps from server to user's local browser timezone
        function formatLocalTimestamp(d) {
            if (!d) return "--";
            let dt = null;
            if (d.timestamp) {
                dt = new Date(d.timestamp * 1000);
            } else if (d.iso_timestamp) {
                dt = new Date(d.iso_timestamp);
            } else if (d.captured_at) {
                let s = d.captured_at.trim();
                if (!s.endsWith("Z") && !s.includes("+") && !s.includes("UTC")) {
                    s = s.replace(" ", "T") + "Z";
                } else {
                    s = s.replace(" UTC", "Z").replace(" ", "T");
                }
                dt = new Date(s);
                if (isNaN(dt.getTime())) {
                    dt = new Date(d.captured_at);
                }
            }
            if (!dt || isNaN(dt.getTime())) {
                return d.captured_at || "--";
            }
            const pad = (n) => String(n).padStart(2, "0");
            const y = dt.getFullYear();
            const m = pad(dt.getMonth() + 1);
            const day = pad(dt.getDate());
            const hh = pad(dt.getHours());
            const mm = pad(dt.getMinutes());
            const ss = pad(dt.getSeconds());
            return `${y}-${m}-${day} ${hh}:${mm}:${ss}`;
        }

        let html = "";
        datasets.forEach(d => {
            const sizeKB = (d.size_bytes / 1024).toFixed(1);
            const localTimeStr = formatLocalTimestamp(d);
            
            // Format labels with badge style
            let badgeClass = "badge-idle";
            let labelName = d.label.charAt(0).toUpperCase() + d.label.slice(1);
            if (d.label === "normal") badgeClass = "badge-normal";
            else if (d.label === "moderate") badgeClass = "badge-moderate";
            else if (d.label === "high") badgeClass = "badge-high";
            else {
                badgeClass = "badge-failed";
                labelName = "Unrecognized";
            }
            
            // Selected options for labelling
            const options = `
                <select onchange="reLabelDataset('${d.filename}', this.value)" style="padding: 6px; font-size: 0.8rem; background: rgba(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 4px; color: var(--text-primary); cursor: pointer;">
                    <option value="" disabled selected>Choose Label...</option>
                    <option value="normal">Normal</option>
                    <option value="moderate">Moderate</option>
                    <option value="high">High</option>
                </select>
            `;
            
            html += `
                <tr>
                    <td style="text-align: center;">
                        <input type="checkbox" class="dataset-row-chk" value="${d.filename}" onchange="updateSelectedDatasetsUI()" style="cursor: pointer; width: 16px; height: 16px; accent-color: var(--color-cyan); vertical-align: middle;">
                    </td>
                    <td style="font-family: monospace; font-size: 0.8rem; word-break: break-all;">${d.filename}</td>
                    <td><strong>${d.subject_id}</strong></td>
                    <td style="color: var(--text-secondary); font-size: 0.78rem; white-space: nowrap;">📅 ${localTimeStr}</td>
                    <td style="color: var(--text-secondary);">${sizeKB} KB</td>
                    <td><span class="pipeline-badge ${badgeClass}" style="padding: 4px 10px; font-size: 0.75rem; border-radius: 6px;">${labelName}</span></td>
                    <td>${options}</td>
                    <td style="text-align: center;">
                        <div style="display: flex; gap: 6px; justify-content: center; align-items: center;">
                            <button class="btn btn-secondary" onclick="openShareModal('${d.filename}', '${d.subject_id}')" style="padding: 4px 8px; font-size: 0.75rem; border-color: rgba(6, 182, 212, 0.4); color: var(--color-cyan); border-radius: 6px; cursor: pointer; display: inline-flex; align-items: center; gap: 3px;" title="Share Dataset">🔗 Share</button>
                            <button class="btn btn-secondary" onclick="openDeleteModal('${d.filename}')" style="background-color: var(--color-rose-glow); color: var(--color-rose); border-color: rgba(244, 63, 94, 0.3); padding: 4px 8px; font-size: 0.75rem; border-radius: 6px; cursor: pointer; display: inline-flex; align-items: center; gap: 3px;" title="Delete Dataset">🗑️ Delete</button>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        listContainer.innerHTML = html;
        updateSelectedDatasetsUI();
    } catch (e) {
        console.error("Error loading raw datasets:", e);
        listContainer.innerHTML = `
            <tr>
                <td colspan="8" style="text-align: center; color: var(--color-rose); padding: 24px;">
                    Error loading raw datasets list: ${e.message}
                </td>
            </tr>
        `;
    }
}

// ---------------------------------------------------------------------------
// DATASET SELECTION HELPERS
// ---------------------------------------------------------------------------
function toggleSelectAllDatasets(selectAll) {
    const checkboxes = document.querySelectorAll(".dataset-row-chk");
    checkboxes.forEach(cb => { cb.checked = selectAll; });
    updateSelectedDatasetsUI();
}

function getSelectedDatasetFilenames() {
    const checkboxes = document.querySelectorAll(".dataset-row-chk:checked");
    return Array.from(checkboxes).map(cb => cb.value);
}

function updateSelectedDatasetsUI() {
    const selected = getSelectedDatasetFilenames();
    const count = selected.length;
    const allCheckboxes = document.querySelectorAll(".dataset-row-chk");
    const total = allCheckboxes.length;
    
    const badge = document.getElementById("selected-datasets-badge");
    const btnBatchShare = document.getElementById("btn-batch-share");
    const btnBatchDelete = document.getElementById("btn-batch-delete");
    const masterChk = document.getElementById("chk-select-all");
    
    if (masterChk) {
        masterChk.checked = total > 0 && count === total;
        masterChk.indeterminate = count > 0 && count < total;
    }
    
    if (count > 0) {
        if (badge) {
            badge.style.display = "inline-block";
            badge.innerText = `${count} selected`;
        }
        if (btnBatchShare) {
            btnBatchShare.style.display = "inline-block";
            btnBatchShare.innerText = `🔗 Share Selected (${count})`;
        }
        if (btnBatchDelete) {
            btnBatchDelete.style.display = "inline-block";
            btnBatchDelete.innerText = `🗑️ Delete Selected (${count})`;
        }
    } else {
        if (badge) badge.style.display = "none";
        if (btnBatchShare) btnBatchShare.style.display = "none";
        if (btnBatchDelete) btnBatchDelete.style.display = "none";
    }
}

// Re-label / Rename a dataset
async function reLabelDataset(filename, newLabel) {
    if (!newLabel) return;
    try {
        const res = await fetch("/api/raw/label", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                filename: filename,
                new_label: newLabel
            })
        });
        
        if (res.ok) {
            const result = await res.json();
            console.log(`[Dashboard] Re-labelled ${filename} to ${result.new_filename}`);
            await loadRawDatasets();
            reloadData();
        } else {
            const err = await res.json();
            alert(`Failed to re-label dataset: ${err.error || "Unknown server error"}`);
        }
    } catch (e) {
        console.error("Error re-labelling dataset:", e);
        alert(`Error re-labelling dataset: ${e.message}`);
    }
}

// ---------------------------------------------------------------------------
// DELETION MODAL & ACTIONS (Single, Batch, All)
// ---------------------------------------------------------------------------
let deleteMode = "single"; // "single" | "batch" | "all"
let pendingDeleteTarget = null; // string for single, array for batch, null for all

function openDeleteModal(filename) {
    deleteMode = "single";
    pendingDeleteTarget = filename;
    document.getElementById("delete-modal-title").innerText = "🗑️ Delete Raw Dataset";
    document.getElementById("delete-modal-message").innerHTML = `Are you sure you want to permanently delete the raw dataset <strong style="color: var(--color-rose);">${filename}</strong> from the server?`;
    document.getElementById("delete-confirm-modal").style.display = "flex";
}

function openBatchDeleteModal() {
    const selected = getSelectedDatasetFilenames();
    if (selected.length === 0) {
        alert("Please select at least one dataset to delete.");
        return;
    }
    deleteMode = "batch";
    pendingDeleteTarget = selected;
    document.getElementById("delete-modal-title").innerText = `🗑️ Delete ${selected.length} Selected Datasets`;
    document.getElementById("delete-modal-message").innerHTML = `Are you sure you want to permanently delete <strong style="color: var(--color-rose);">${selected.length} selected dataset(s)</strong> from the server?`;
    document.getElementById("delete-confirm-modal").style.display = "flex";
}

function openDeleteAllModal() {
    deleteMode = "all";
    pendingDeleteTarget = null;
    document.getElementById("delete-modal-title").innerText = "⚠️ Delete ALL Raw Datasets";
    document.getElementById("delete-modal-message").innerHTML = `Are you sure you want to permanently delete <strong style="color: var(--color-rose);">ALL raw datasets</strong> in the pool? This will wipe all recorded sessions and cannot be undone.`;
    document.getElementById("delete-confirm-modal").style.display = "flex";
}

function closeDeleteModal(e) {
    if (e && e.target !== e.currentTarget) return;
    document.getElementById("delete-confirm-modal").style.display = "none";
    pendingDeleteTarget = null;
}

async function executeDeleteAction() {
    closeDeleteModal();
    const btnConfirm = document.getElementById("btn-modal-confirm-delete");
    if (btnConfirm) btnConfirm.disabled = true;

    try {
        let res;
        if (deleteMode === "single") {
            res = await fetch("/api/raw/delete", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: pendingDeleteTarget })
            });
        } else if (deleteMode === "batch") {
            res = await fetch("/api/raw/delete_batch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filenames: pendingDeleteTarget })
            });
        } else if (deleteMode === "all") {
            res = await fetch("/api/raw/delete_all", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({})
            });
        }

        if (res) {
            let data = null;
            try {
                data = await res.json();
            } catch (_) {}

            if (res.ok) {
                console.log("[Dashboard] Delete success:", data);
            } else {
                const errMsg = (data && data.error) ? data.error : (res.status === 404 ? "File already removed." : `Server status ${res.status}`);
                console.warn("[Dashboard] Delete notice:", errMsg);
            }
        }
    } catch (e) {
        console.error("Error executing delete:", e);
    } finally {
        if (btnConfirm) btnConfirm.disabled = false;
        // Always refresh the dataset table so removed files disappear cleanly
        await loadRawDatasets();
        reloadData();
    }
}

// ---------------------------------------------------------------------------
// SHARE MODAL & ACTIONS
// ---------------------------------------------------------------------------
let currentShareData = { filename: "", subjectId: "", url: "" };

function openShareModal(filename, subjectId) {
    const origin = window.location.origin;
    const downloadUrl = `${origin}/api/raw/download/${encodeURIComponent(filename)}`;
    currentShareData = { filename, subjectId, url: downloadUrl };
    
    document.getElementById("share-modal-title").innerText = `Share: ${subjectId || filename}`;
    document.getElementById("share-modal-filename").innerText = filename;
    document.getElementById("share-modal-link-input").value = downloadUrl;
    document.getElementById("btn-share-direct-download").href = downloadUrl;
    document.getElementById("btn-copy-share-link").innerText = "📋 Copy";
    document.getElementById("share-modal").style.display = "flex";
}

function closeShareModal(e) {
    if (e && e.target !== e.currentTarget) return;
    document.getElementById("share-modal").style.display = "none";
}

async function copyShareModalLink() {
    const input = document.getElementById("share-modal-link-input");
    const btn = document.getElementById("btn-copy-share-link");
    try {
        await navigator.clipboard.writeText(input.value);
        btn.innerText = "✅ Copied!";
        setTimeout(() => { btn.innerText = "📋 Copy"; }, 2000);
    } catch (_) {
        input.select();
        document.execCommand("copy");
        btn.innerText = "✅ Copied!";
        setTimeout(() => { btn.innerText = "📋 Copy"; }, 2000);
    }
}

async function triggerNativeShare() {
    const title = `PVD Dataset: ${currentShareData.subjectId || currentShareData.filename}`;
    const text = `Raw physiological PPG session recorded for ${currentShareData.subjectId}: ${currentShareData.filename}`;
    const url = currentShareData.url;
    
    if (navigator.share) {
        try {
            await navigator.share({ title, text, url });
        } catch (err) {
            if (err.name !== "AbortError") {
                copyShareModalLink();
            }
        }
    } else {
        copyShareModalLink();
        alert("Direct link copied to clipboard! You can paste and share it with clinicians.");
    }
}

async function shareSelectedDatasets() {
    const selected = getSelectedDatasetFilenames();
    if (selected.length === 0) return;
    const origin = window.location.origin;
    
    if (selected.length === 1) {
        openShareModal(selected[0], "");
        return;
    }
    
    const links = selected.map(fn => `${origin}/api/raw/download/${encodeURIComponent(fn)}`).join("\n");
    const shareText = `PVD Datasets (${selected.length} sessions):\n` + links;
    
    if (navigator.share) {
        try {
            await navigator.share({
                title: `PVD Datasets (${selected.length} files)`,
                text: shareText
            });
            return;
        } catch (_) {}
    }
    
    try {
        await navigator.clipboard.writeText(shareText);
        alert(`Copied direct download links for ${selected.length} datasets to clipboard!`);
    } catch (_) {
        prompt("Copy dataset links below:", shareText);
    }
}

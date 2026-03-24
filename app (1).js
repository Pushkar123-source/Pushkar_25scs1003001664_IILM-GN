// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// State Management
let modelLoaded = false;
let modelClasses = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkModelStatus();
});

// ============================================================================
// MODEL MANAGEMENT
// ============================================================================

/**
 * Check if model is loaded and update UI
 */
async function checkModelStatus() {
    try {
        showLoading('Checking model status...');
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        updateModelStatus(data.model_loaded);
        
        if (data.model_loaded) {
            await getModelInfo();
        }
        
        hideLoading();
    } catch (error) {
        console.error('Error checking model status:', error);
        updateModelStatus(false);
        hideLoading();
        showToast('Backend server not available. Please start the Flask server.', 'error');
    }
}

/**
 * Train the model
 */
async function trainModel() {
    if (!confirm('Training the model will take 10-30 seconds. Continue?')) {
        return;
    }
    
    try {
        showLoading('Training model... This may take up to 30 seconds');
        
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                csv_path: '../onemg.csv'
            })
        });
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.success) {
            modelClasses = data.classes || [];
            updateModelStatus(true);
            showToast(`Model trained successfully! Accuracy: ${(data.accuracy * 100).toFixed(2)}%`, 'success');
            await getModelInfo();
        } else {
            showToast(`Training failed: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Error training model:', error);
        hideLoading();
        showToast('Error training model. Check console for details.', 'error');
    }
}

/**
 * Load pre-trained model
 */
async function loadModel() {
    try {
        showLoading('Loading pre-trained model...');
        
        const response = await fetch(`${API_BASE_URL}/load-model`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_path: '../models/'
            })
        });
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.success) {
            modelClasses = data.classes || [];
            updateModelStatus(true);
            showToast('Pre-trained model loaded successfully!', 'success');
            await getModelInfo();
        } else {
            showToast(`Loading failed: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Error loading model:', error);
        hideLoading();
        showToast('Error loading model. Check console for details.', 'error');
    }
}

/**
 * Get model information and update UI
 */
async function getModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        const data = await response.json();
        
        if (data.success) {
            const info = data.model_info;
            document.getElementById('num-classes').textContent = info.num_classes || '-';
            modelClasses = info.classes || [];
        }
    } catch (error) {
        console.error('Error getting model info:', error);
    }
}

/**
 * Update model status badge in UI
 */
function updateModelStatus(loaded) {
    modelLoaded = loaded;
    const statusElement = document.getElementById('model-status');
    
    if (loaded) {
        statusElement.innerHTML = '<span class="badge badge-success">Loaded</span>';
    } else {
        statusElement.innerHTML = '<span class="badge badge-warning">Not Loaded</span>';
    }
}

// ============================================================================
// PREDICTION
// ============================================================================

/**
 * Predict drug type from description
 */
async function predictDrugType() {
    const description = document.getElementById('drug-description').value.trim();
    
    if (!description) {
        showToast('Please enter a drug description', 'error');
        return;
    }
    
    if (!modelLoaded) {
        showToast('Please load or train the model first', 'error');
        return;
    }
    
    try {
        showLoading('Predicting drug type...');
        
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: description
            })
        });
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.success) {
            displayPredictionResults(data.prediction);
            showToast('Prediction completed successfully!', 'success');
        } else {
            showToast(`Prediction failed: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Error predicting:', error);
        hideLoading();
        showToast('Error making prediction. Check console for details.', 'error');
    }
}

/**
 * Display prediction results in UI
 */
function displayPredictionResults(prediction) {
    // Show results section
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';
    
    // Update main prediction
    document.getElementById('predicted-type').textContent = prediction.predicted_drug_type;
    document.getElementById('confidence-value').textContent = 
        `${(prediction.confidence * 100).toFixed(2)}%`;
    
    // Update probability bars
    const barsContainer = document.getElementById('probability-bars-container');
    barsContainer.innerHTML = '';
    
    // Get top 3 predictions
    const topPredictions = prediction.top_predictions.slice(0, 3);
    
    topPredictions.forEach((pred, index) => {
        const barHtml = `
            <div class="probability-bar">
                <div class="probability-label">
                    <span>${pred.drug_type}</span>
                    <span>${(pred.probability * 100).toFixed(2)}%</span>
                </div>
                <div class="probability-fill-container">
                    <div class="probability-fill" style="width: ${pred.probability * 100}%">
                    </div>
                </div>
            </div>
        `;
        barsContainer.innerHTML += barHtml;
    });
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Clear prediction input and results
 */
function clearPrediction() {
    document.getElementById('drug-description').value = '';
    document.getElementById('results-section').style.display = 'none';
    showToast('Cleared', 'info');
}

/**
 * Load sample drug description
 */
async function loadSampleData() {
    try {
        showLoading('Loading sample data...');
        
        const response = await fetch(`${API_BASE_URL}/sample-drugs`);
        const data = await response.json();
        
        hideLoading();
        
        if (data.success && data.samples.length > 0) {
            // Get random sample
            const randomSample = data.samples[Math.floor(Math.random() * data.samples.length)];
            document.getElementById('drug-description').value = randomSample.description;
            showToast('Sample data loaded! Click Predict to classify.', 'info');
        } else {
            // Fallback sample if API fails
            loadFallbackSample();
        }
    } catch (error) {
        console.error('Error loading sample:', error);
        hideLoading();
        loadFallbackSample();
    }
}

/**
 * Load a hardcoded sample if API fails
 */
function loadFallbackSample() {
    const sampleText = `This medication is an H2 receptor antagonist used to treat acid reflux and heartburn. It works by reducing the amount of acid produced in the stomach. Common side effects include headache, dizziness, and constipation. It belongs to the antiulcer therapeutic class and is typically taken orally.`;
    
    document.getElementById('drug-description').value = sampleText;
    showToast('Sample data loaded!', 'info');
}

// ============================================================================
// DATASET STATISTICS
// ============================================================================

/**
 * Load and display dataset statistics
 */
async function loadDatasetStats() {
    try {
        showLoading('Loading dataset statistics...');
        
        const response = await fetch(`${API_BASE_URL}/dataset-stats`);
        const data = await response.json();
        
        hideLoading();
        
        if (data.success) {
            displayDatasetStats(data.stats);
            showToast('Statistics loaded successfully!', 'success');
        } else {
            showToast(`Failed to load statistics: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Error loading statistics:', error);
        hideLoading();
        showToast('Error loading statistics. Check console for details.', 'error');
    }
}

/**
 * Display dataset statistics in UI
 */
function displayDatasetStats(stats) {
    const statsContent = document.getElementById('stats-content');
    
    let html = `
        <div class="stats-summary">
            <p><strong>Total Records:</strong> ${stats.total_records}</p>
            <p><strong>Total Features:</strong> ${stats.total_features}</p>
            <p><strong>Drug Types:</strong> ${stats.num_drug_types}</p>
        </div>
        <div class="stats-distribution">
            <h4>Drug Type Distribution:</h4>
            <div class="distribution-bars">
    `;
    
    // Sort drug types by count
    const sortedTypes = Object.entries(stats.drug_type_counts)
        .sort((a, b) => b[1] - a[1]);
    
    const maxCount = Math.max(...Object.values(stats.drug_type_counts));
    
    sortedTypes.forEach(([type, count]) => {
        const percentage = (count / maxCount * 100);
        html += `
            <div class="probability-bar">
                <div class="probability-label">
                    <span>${type}</span>
                    <span>${count} samples</span>
                </div>
                <div class="probability-fill-container">
                    <div class="probability-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    });
    
    html += `
            </div>
        </div>
    `;
    
    statsContent.innerHTML = html;
}

// ============================================================================
// UI UTILITIES
// ============================================================================

/**
 * Show loading overlay
 */
function showLoading(message = 'Processing...') {
    const overlay = document.getElementById('loading-overlay');
    const text = document.getElementById('loading-text');
    text.textContent = message;
    overlay.style.display = 'flex';
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = 'none';
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.style.display = 'block';
    
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000);
}

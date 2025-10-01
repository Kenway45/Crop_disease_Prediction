// Camera and prediction logic
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const captureBtn = document.getElementById('captureBtn');
const stopBtn = document.getElementById('stopBtn');
const loader = document.getElementById('loader');
const errorDiv = document.getElementById('error');
const resultsDiv = document.getElementById('results');
const predictionLabel = document.getElementById('predictionLabel');
const confidenceDiv = document.getElementById('confidence');
const topPredictionsDiv = document.getElementById('topPredictions');

let stream = null;

// Start camera
startBtn.addEventListener('click', async () => {
    try {
        errorDiv.style.display = 'none';
        
        // Request camera access
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment' // Use back camera on mobile
            } 
        });
        
        video.srcObject = stream;
        video.style.display = 'block';
        
        // Update button states
        startBtn.disabled = true;
        captureBtn.disabled = false;
        stopBtn.disabled = false;
        
    } catch (err) {
        console.error('Camera error:', err);
        showError('Failed to access camera. Please make sure you have granted camera permissions.');
    }
});

// Stop camera
stopBtn.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        video.style.display = 'none';
        
        // Update button states
        startBtn.disabled = false;
        captureBtn.disabled = true;
        stopBtn.disabled = true;
    }
});

// Capture and predict
captureBtn.addEventListener('click', async () => {
    try {
        errorDiv.style.display = 'none';
        resultsDiv.classList.remove('show');
        
        // Capture frame from video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Show loader
        loader.style.display = 'block';
        captureBtn.disabled = true;
        
        // Send to server
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (err) {
        console.error('Prediction error:', err);
        showError('Failed to get prediction. Please try again.');
    } finally {
        loader.style.display = 'none';
        captureBtn.disabled = false;
    }
});

// Display prediction results
function displayResults(result) {
    // Main prediction
    predictionLabel.textContent = result.prediction;
    confidenceDiv.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
    
    // Top predictions
    topPredictionsDiv.innerHTML = '';
    result.top_predictions.forEach(pred => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        const label = document.createElement('span');
        label.className = 'label';
        label.textContent = pred.class;
        
        const barContainer = document.createElement('div');
        barContainer.className = 'confidence-bar';
        
        const barFill = document.createElement('div');
        barFill.className = 'confidence-fill';
        barFill.style.width = '0%';
        barFill.textContent = `${(pred.confidence * 100).toFixed(1)}%`;
        
        barContainer.appendChild(barFill);
        item.appendChild(label);
        item.appendChild(barContainer);
        topPredictionsDiv.appendChild(item);
        
        // Animate bar
        setTimeout(() => {
            barFill.style.width = `${pred.confidence * 100}%`;
        }, 100);
    });
    
    // Show results
    resultsDiv.classList.add('show');
}

// Show error message
function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

// Check if browser supports required features
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError('Your browser does not support camera access. Please use a modern browser like Chrome, Firefox, or Safari.');
    startBtn.disabled = true;
}


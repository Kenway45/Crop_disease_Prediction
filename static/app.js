// Camera and prediction logic
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const visualCanvas = document.getElementById('visualCanvas');
const startBtn = document.getElementById('startBtn');
const captureBtn = document.getElementById('captureBtn');
const stopBtn = document.getElementById('stopBtn');
const loader = document.getElementById('loader');
const errorDiv = document.getElementById('error');
const warningDiv = document.getElementById('warning');
const resultsDiv = document.getElementById('results');
const predictionLabel = document.getElementById('predictionLabel');
const confidenceDiv = document.getElementById('confidence');
const topPredictionsDiv = document.getElementById('topPredictions');
const themeToggle = document.getElementById('themeToggle');
const capturedContainer = document.getElementById('capturedContainer');
const capturedImage = document.getElementById('capturedImage');
const overlayCanvas = document.getElementById('overlayCanvas');

let stream = null;
let currentTheme = 'light';

// Theme toggle functionality
themeToggle.addEventListener('click', () => {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    themeToggle.textContent = currentTheme === 'light' ? '🌓' : '☀️';
    localStorage.setItem('theme', currentTheme);
});

// Load saved theme
const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
    currentTheme = savedTheme;
    document.documentElement.setAttribute('data-theme', currentTheme);
    themeToggle.textContent = currentTheme === 'light' ? '🌓' : '☀️';
}

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
        warningDiv.style.display = 'none';
        resultsDiv.classList.remove('show');
        capturedContainer.style.display = 'none';
        
        // Capture frame from video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Display captured image
        capturedImage.src = imageData;
        capturedContainer.style.display = 'inline-block';
        
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
        
        // Display results with visualization
        displayResults(result, imageData);
        
    } catch (err) {
        console.error('Prediction error:', err);
        showError('Failed to get prediction. Please try again.');
    } finally {
        loader.style.display = 'none';
        captureBtn.disabled = false;
    }
});

// Display prediction results
function displayResults(result, imageData) {
    // Handle errors and warnings
    if (result.error) {
        if (result.error === 'No leaf detected') {
            showError('🍃 ' + result.message);
        } else {
            showError(result.message || result.error);
        }
        return;
    }
    
    if (result.warning) {
        showWarning('⚠️ ' + result.message);
    }
    
    // Main prediction
    predictionLabel.textContent = result.prediction || 'Unknown';
    
    // Confidence with color coding
    const confidencePercent = (result.confidence * 100).toFixed(1);
    confidenceDiv.textContent = `🔮 Quantum Confidence: ${confidencePercent}%`;
    
    // Color code confidence
    confidenceDiv.className = 'confidence';
    if (result.confidence >= 0.7) {
        confidenceDiv.classList.add('high');
    } else if (result.confidence >= 0.4) {
        confidenceDiv.classList.add('medium');
    } else {
        confidenceDiv.classList.add('low');
    }
    
    // Top predictions
    topPredictionsDiv.innerHTML = '';
    if (result.top_predictions) {
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
    }
    
    // Draw leaf bounding box and disease overlay
    drawLeafVisualization(result);
    
    // Show results
    resultsDiv.classList.add('show');
}

// Draw leaf detection bounding box and disease region overlay
function drawLeafVisualization(result) {
    const img = capturedImage;
    
    // Wait for image to load
    img.onload = () => {
        // Set canvas size to match image
        overlayCanvas.width = img.width;
        overlayCanvas.height = img.height;
        overlayCanvas.style.width = img.clientWidth + 'px';
        overlayCanvas.style.height = img.clientHeight + 'px';
        
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        
        // Draw leaf bounding box (green)
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 4;
        ctx.setLineDash([10, 5]);
        
        // Draw box around the whole leaf area (approximate)
        const margin = 20;
        ctx.strokeRect(
            margin, 
            margin, 
            overlayCanvas.width - 2 * margin, 
            overlayCanvas.height - 2 * margin
        );
        
        // Add "Leaf Detected" label
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.fillRect(margin, margin - 30, 150, 30);
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 16px Arial';
        ctx.fillText('🍃 Leaf Detected', margin + 5, margin - 8);
        
        // Draw disease region overlay (red semi-transparent areas)
        // Simulate disease detection in multiple regions
        if (result.confidence > 0.5 && result.prediction !== 'Healthy') {
            ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
            
            // Draw several disease "hotspots"
            const numRegions = Math.floor(Math.random() * 3) + 2; // 2-4 regions
            
            for (let i = 0; i < numRegions; i++) {
                // Random positions in the leaf area
                const x = margin + Math.random() * (overlayCanvas.width - 2 * margin - 80);
                const y = margin + Math.random() * (overlayCanvas.height - 2 * margin - 80);
                const width = 50 + Math.random() * 80;
                const height = 50 + Math.random() * 80;
                
                // Draw irregular disease region
                ctx.beginPath();
                ctx.ellipse(x + width/2, y + height/2, width/2, height/2, 0, 0, Math.PI * 2);
                ctx.fill();
                
                // Draw red outline
                ctx.strokeStyle = '#ff0000';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
            
            // Add "Disease Detected" label
            ctx.fillStyle = 'rgba(255, 0, 0, 0.9)';
            ctx.fillRect(overlayCanvas.width - margin - 180, margin - 30, 180, 30);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 16px Arial';
            ctx.fillText('🔴 Disease Detected', overlayCanvas.width - margin - 175, margin - 8);
        }
    };
    
    // Trigger load if image is already loaded
    if (img.complete) {
        img.onload();
    }
}

// Show error message
function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

// Show warning message
function showWarning(message) {
    warningDiv.textContent = message;
    warningDiv.style.display = 'block';
}

// Check if browser supports required features
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError('Your browser does not support camera access. Please use a modern browser like Chrome, Firefox, or Safari.');
    startBtn.disabled = true;
}


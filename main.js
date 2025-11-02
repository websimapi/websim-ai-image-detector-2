import { pipeline } from '@xenova/transformers';

// DOM Elements
const imageUploadInput = document.getElementById('image-upload-input');
const imagePreview = document.getElementById('image-preview');
const detectButton = document.getElementById('detect-button');
const statusMessage = document.getElementById('status-message');
const resultsContainer = document.getElementById('results-container');
const progressBar = document.getElementById('progress-bar');
const progressBarContainer = document.getElementById('progress-bar-container');

// State
let classifier = null;
let imageUrl = null;

// Zero-shot classification labels for AI detection
const candidateLabels = [
    'AI generated image',
    'artificial intelligence created image',
    'computer generated image',
    'synthetic image',
    'real photograph',
    'authentic photograph',
    'natural photograph',
    'camera captured image'
];

// Model Loading
class AIDetectorPipeline {
    static task = 'zero-shot-image-classification';
    static model = 'Xenova/clip-vit-base-patch32';
    static instance = null;

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, { progress_callback });
        }
        return this.instance;
    }
}

const updateLoadingStatus = (data) => {
    if (data.status === 'progress') {
        const percentage = (data.progress * 100).toFixed(2);
        progressBar.style.width = `${percentage}%`;
        statusMessage.textContent = `Loading ${data.file}... ${percentage}%`;
    } else if (data.status === 'done') {
        progressBar.style.width = '100%';
        statusMessage.textContent = 'Model loaded successfully! Ready to detect AI images.';
        setTimeout(() => {
            progressBarContainer.style.display = 'none';
            if(imageUrl) {
                statusMessage.textContent = 'Image ready. Click "Analyze Image" to begin!';
            } else {
                statusMessage.textContent = 'Please upload an image to begin.';
            }
        }, 1000);
    } else {
        statusMessage.textContent = `Loading model: ${data.status}...`;
    }
};

const initializeModel = async () => {
    statusMessage.textContent = 'Loading AI detection model... (first load may take 30-60s)';
    progressBarContainer.style.display = 'block';
    try {
        classifier = await AIDetectorPipeline.getInstance(updateLoadingStatus);
    } catch (error) {
        console.error('Error loading model:', error);
        statusMessage.textContent = 'Error loading model. Please refresh the page.';
    }
};

// Image Upload Handler
imageUploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        if (imageUrl) {
            URL.revokeObjectURL(imageUrl);
        }
        imageUrl = URL.createObjectURL(file);
        imagePreview.src = imageUrl;
        imagePreview.style.display = 'block';
        imagePreview.classList.add('has-image');
        
        detectButton.disabled = false;
        resultsContainer.innerHTML = '';
        if (classifier) {
            statusMessage.textContent = 'Image ready. Click "Analyze Image" to begin!';
        }
    }
});

// Detection Handler
detectButton.addEventListener('click', async () => {
    if (!imageUrl || !classifier) return;

    detectButton.disabled = true;
    statusMessage.textContent = 'Analyzing image for AI artifacts...';
    resultsContainer.innerHTML = '';

    try {
        const results = await classifier(imageUrl, candidateLabels);
        displayResults(results);
        statusMessage.textContent = 'Analysis complete! Upload another image to test.';
    } catch (error) {
        console.error('Error during classification:', error);
        statusMessage.textContent = 'Error analyzing image. Please try another image.';
        resultsContainer.innerHTML = '<p style="color: #dc3545; text-align: center;">Analysis failed. Please try a different image.</p>';
    } finally {
        detectButton.disabled = false;
    }
});

const displayResults = (results) => {
    resultsContainer.innerHTML = '';
    
    // Calculate AI vs Real scores
    let aiScore = 0;
    let realScore = 0;
    
    results.forEach(result => {
        const label = result.label.toLowerCase();
        if (label.includes('ai') || label.includes('artificial') || label.includes('computer') || label.includes('synthetic')) {
            aiScore += result.score;
        } else if (label.includes('real') || label.includes('authentic') || label.includes('natural') || label.includes('camera')) {
            realScore += result.score;
        }
    });

    // Normalize scores
    const total = aiScore + realScore;
    const aiPercentage = (aiScore / total) * 100;
    const realPercentage = (realScore / total) * 100;

    // Display verdict
    const verdictDiv = document.createElement('div');
    verdictDiv.className = 'ai-indicator';
    
    if (aiPercentage > 60) {
        verdictDiv.classList.add('likely-ai');
        verdictDiv.innerHTML = `🤖 Likely AI-Generated (${aiPercentage.toFixed(1)}% confidence)`;
    } else if (realPercentage > 60) {
        verdictDiv.classList.add('likely-real');
        verdictDiv.innerHTML = `📷 Likely Real Photo (${realPercentage.toFixed(1)}% confidence)`;
    } else {
        verdictDiv.classList.add('uncertain');
        verdictDiv.innerHTML = `🤔 Uncertain - Scores too close to determine`;
    }
    
    resultsContainer.appendChild(verdictDiv);

    // Group and display detailed scores
    const aiResults = results.filter(r => {
        const label = r.label.toLowerCase();
        return label.includes('ai') || label.includes('artificial') || label.includes('computer') || label.includes('synthetic');
    }).sort((a, b) => b.score - a.score);

    const realResults = results.filter(r => {
        const label = r.label.toLowerCase();
        return label.includes('real') || label.includes('authentic') || label.includes('natural') || label.includes('camera');
    }).sort((a, b) => b.score - a.score);

    // Display top AI indicators
    if (aiResults.length > 0) {
        const aiHeader = document.createElement('h3');
        aiHeader.style.cssText = 'margin-top: 20px; color: #667eea; font-size: 1rem;';
        aiHeader.textContent = '🤖 AI-Generated Indicators:';
        resultsContainer.appendChild(aiHeader);

        aiResults.slice(0, 3).forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            
            const label = document.createElement('span');
            label.textContent = result.label;
            
            const score = document.createElement('span');
            score.textContent = `${(result.score * 100).toFixed(2)}%`;

            resultItem.appendChild(label);
            resultItem.appendChild(score);
            resultsContainer.appendChild(resultItem);
        });
    }

    // Display top Real indicators
    if (realResults.length > 0) {
        const realHeader = document.createElement('h3');
        realHeader.style.cssText = 'margin-top: 20px; color: #667eea; font-size: 1rem;';
        realHeader.textContent = '📷 Real Photo Indicators:';
        resultsContainer.appendChild(realHeader);

        realResults.slice(0, 3).forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            
            const label = document.createElement('span');
            label.textContent = result.label;
            
            const score = document.createElement('span');
            score.textContent = `${(result.score * 100).toFixed(2)}%`;

            resultItem.appendChild(label);
            resultItem.appendChild(score);
            resultsContainer.appendChild(resultItem);
        });
    }
};

// Initialize on load
initializeModel();
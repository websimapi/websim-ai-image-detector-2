import { pipeline } from '@xenova/transformers';

// DOM Elements
const imageUploadInput = document.getElementById('image-upload-input');
const imagePreview = document.getElementById('image-preview');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const detectButton = document.getElementById('detect-button');
const statusMessage = document.getElementById('status-message');
const resultsContainer = document.getElementById('results-container');
const progressBar = document.getElementById('progress-bar');
const progressBarContainer = document.getElementById('progress-bar-container');

// State
let classifier = null;
let imageUrl = null;

// --- Model Loading ---

class AiImageDetectorPipeline {
    static task = 'image-classification';
    static model = 'Organika/AI-image-detector';
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
        statusMessage.textContent = `${data.file} (${percentage}%)`;
    } else if (data.status === 'done') {
        progressBar.style.width = '100%';
        statusMessage.textContent = 'Model loaded. Ready to detect!';
        setTimeout(() => {
            progressBarContainer.style.display = 'none';
            if(imageUrl) {
                 statusMessage.textContent = 'Ready to detect. Press the button!';
            } else {
                 statusMessage.textContent = 'Please upload an image to begin.';
            }
        }, 1000);
    } else {
        statusMessage.textContent = `Loading model: ${data.status}...`;
    }
};

const initializeModel = async () => {
    statusMessage.textContent = 'Loading AI model... (this may take a moment)';
    progressBarContainer.style.display = 'block';
    classifier = await AiImageDetectorPipeline.getInstance(updateLoadingStatus);
};

// --- UI Interaction ---

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
             statusMessage.textContent = 'Ready to detect. Press the button!';
        }
    }
});

detectButton.addEventListener('click', async () => {
    if (!imageUrl || !classifier) return;

    detectButton.disabled = true;
    statusMessage.textContent = 'Analyzing image...';
    resultsContainer.innerHTML = '';

    try {
        const results = await classifier(imageUrl);
        displayResults(results);
        statusMessage.textContent = 'Analysis complete. Upload another image?';
    } catch (error) {
        console.error('Error during classification:', error);
        statusMessage.textContent = 'An error occurred during analysis.';
    } finally {
        detectButton.disabled = false;
    }
});

const displayResults = (results) => {
    resultsContainer.innerHTML = '';
    
    // Sort results by score in descending order
    results.sort((a, b) => b.score - a.score);

    results.forEach(result => {
        const resultItem = document.createElement('div');
        resultItem.className = `result-item ${result.label.toLowerCase()}`;
        
        const label = document.createElement('span');
        label.textContent = result.label;
        
        const score = document.createElement('span');
        score.textContent = `${(result.score * 100).toFixed(2)}%`;

        resultItem.appendChild(label);
        resultItem.appendChild(score);
        resultsContainer.appendChild(resultItem);
    });
};

// --- Initialization ---

// Initialize the model as soon as the page loads.
initializeModel();
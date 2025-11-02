import { pipeline } from '@xenova/transformers';
import ExifReader from 'exifreader';

// DOM Elements
const imageUploadInput = document.getElementById('image-upload-input');
const imagePreview = document.getElementById('image-preview');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const detectButton = document.getElementById('detect-button');
const statusMessage = document.getElementById('status-message');
const resultsContainer = document.getElementById('results-container');
const metadataContainer = document.getElementById('metadata-container');
const progressBar = document.getElementById('progress-bar');
const progressBarContainer = document.getElementById('progress-bar-container');

// State
let classifier = null;
let imageUrl = null;
let imageFile = null;

// --- Model Loading ---

class AiImageDetectorPipeline {
    static task = 'image-classification';
    static model = 'Xenova/vit-base-patch16-224';
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
        imageFile = file; // Store the file object
        if (imageUrl) {
            URL.revokeObjectURL(imageUrl);
        }
        imageUrl = URL.createObjectURL(file);
        imagePreview.src = imageUrl;
        imagePreview.style.display = 'block';
        uploadPlaceholder.style.display = 'none'; // Explicitly hide placeholder
        
        detectButton.disabled = false;
        resultsContainer.innerHTML = '';
        metadataContainer.innerHTML = ''; // Clear old metadata
        metadataContainer.style.display = 'none'; // Hide metadata on new image
        if (classifier) {
             statusMessage.textContent = 'Ready to detect. Press the button!';
        }
    }
});

detectButton.addEventListener('click', async () => {
    if (!imageUrl || !classifier || !imageFile) return;

    detectButton.disabled = true;
    statusMessage.textContent = 'Analyzing image...';
    resultsContainer.innerHTML = '';
    metadataContainer.style.display = 'none';

    try {
        // Run classification and metadata extraction in parallel
        const [results, metadata] = await Promise.all([
            classifier(imageUrl),
            ExifReader.load(imageFile).catch(err => {
                console.warn("Could not read metadata:", err);
                return null; // Don't block analysis if metadata fails
            })
        ]);
        
        displayResults(results);
        displayMetadata(metadata);
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
    
    // Sort results by score and take the top 5
    const topResults = results.sort((a, b) => b.score - a.score).slice(0, 5);

    topResults.forEach(result => {
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
};

const displayMetadata = (metadata) => {
    metadataContainer.innerHTML = '';

    if (!metadata || Object.keys(metadata).length === 0) {
        metadataContainer.style.display = 'none';
        return;
    }

    const title = document.createElement('h3');
    title.textContent = 'Image Metadata';
    metadataContainer.appendChild(title);

    const table = document.createElement('table');
    const tbody = document.createElement('tbody');

    for (const key in metadata) {
        if (Object.prototype.hasOwnProperty.call(metadata, key)) {
            const value = metadata[key];
            if (value && typeof value.description !== 'undefined') {
                const row = document.createElement('tr');
                const keyCell = document.createElement('td');
                keyCell.textContent = key;
                const valueCell = document.createElement('td');
                valueCell.textContent = value.description;
                row.appendChild(keyCell);
                row.appendChild(valueCell);
                tbody.appendChild(row);
            }
        }
    }

    if (tbody.children.length > 0) {
        table.appendChild(tbody);
        metadataContainer.appendChild(table);
        metadataContainer.style.display = 'block';
    } else {
        const noData = document.createElement('p');
        noData.textContent = 'No readable metadata found in this image.';
        metadataContainer.appendChild(noData);
        metadataContainer.style.display = 'block';
    }
};

// --- Initialization ---

// Initialize the model as soon as the page loads.
initializeModel();
// --- DOM Elements ---
const imageUploadInput = document.getElementById('image-upload-input');
const imagePreview = document.getElementById('image-preview');
const detectButton = document.getElementById('detect-button');
const statusMessage = document.getElementById('status-message');
const progressBar = document.getElementById('progress-bar');
const progressBarContainer = document.getElementById('progress-bar-container');

// Results containers
const resultsContainer = document.getElementById('results-container');
const overallAssessment = document.getElementById('overall-assessment');
const metadataResults = document.getElementById('metadata-results');
const mlResults = document.getElementById('ml-results');
const technicalDetails = document.getElementById('technical-details');


// --- State & Config ---
let session;
let imageUrl = null;
let imageFile = null;
const modelUrl = "https://huggingface.co/Organika/sdxl-detector/resolve/refs%2Fpr%2F3/onnx/model.onnx";
const modelInputSize = 384;

// --- Model & Preprocessing ---

async function initializeModel() {
    statusMessage.textContent = 'Loading AI detection model... (this may take a moment)';
    progressBarContainer.style.display = 'block';
    progressBar.style.width = '30%';

    try {
        // ort.env.wasm.proxy = true; // For cross-origin workers
        session = await ort.InferenceSession.create(modelUrl);
        progressBar.style.width = '100%';
        statusMessage.textContent = 'Model loaded successfully! Ready for analysis.';
        setTimeout(() => {
            progressBarContainer.style.display = 'none';
            if(imageUrl) {
                statusMessage.textContent = 'Image ready. Click "Analyze Image" to begin!';
            } else {
                statusMessage.textContent = 'Please upload an image to begin.';
            }
        }, 1000);
    } catch (e) {
        console.error(`Failed to load the ONNX model: ${e}`);
        statusMessage.textContent = 'Error: Could not load the AI model. Please refresh.';
    }
}

async function preprocess(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = modelInputSize;
    canvas.height = modelInputSize;
    ctx.drawImage(image, 0, 0, modelInputSize, modelInputSize);
    
    const imageData = ctx.getImageData(0, 0, modelInputSize, modelInputSize);
    const data = imageData.data;
    
    // ImageNet mean and std
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    const tensorData = new Float32Array(3 * modelInputSize * modelInputSize);
    let dataIndex = 0;
    
    for (let c = 0; c < 3; ++c) {
        for (let h = 0; h < modelInputSize; ++h) {
            for (let w = 0; w < modelInputSize; ++w) {
                const pixelIndex = (h * modelInputSize + w) * 4;
                const value = data[pixelIndex + c] / 255;
                tensorData[dataIndex++] = (value - mean[c]) / std[c];
            }
        }
    }
    
    return new ort.Tensor('float32', tensorData, [1, 3, modelInputSize, modelInputSize]);
}

// --- Metadata Analysis ---

function analyzeMetadata(file, callback) {
    const findings = {
        hasAiMarkers: false,
        details: {},
        raw: "No EXIF data found.",
    };
    const aiKeywords = ['midjourney', 'dall-e', 'stable diffusion', 'adobe firefly', 'ai generated', 'synthes', 'artificial intelligence'];

    EXIF.getData(file, function() {
        const allTags = EXIF.getAllTags(this);
        if (Object.keys(allTags).length === 0) {
            callback(findings);
            return;
        }

        findings.raw = JSON.stringify(allTags, null, 2);
        
        const checkFields = ['Software', 'ImageDescription', 'UserComment', 'Artist', 'Copyright'];
        checkFields.forEach(field => {
            if (allTags[field]) {
                const value = allTags[field].toString().toLowerCase();
                findings.details[field] = allTags[field];
                if (aiKeywords.some(keyword => value.includes(keyword))) {
                    findings.hasAiMarkers = true;
                    findings.details[field] = `!! AI Marker Found: ${allTags[field]}`;
                }
            }
        });
        
        if(allTags.DateTime) findings.details['Creation Date'] = allTags.DateTime;

        callback(findings);
    });
}


// --- UI & Event Handlers ---

imageUploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        imageFile = file;
        if (imageUrl) URL.revokeObjectURL(imageUrl);
        
        imageUrl = URL.createObjectURL(file);
        imagePreview.src = imageUrl;
        imagePreview.style.display = 'block';
        imagePreview.classList.add('has-image');
        
        detectButton.disabled = false;
        resultsContainer.style.display = 'none';
        if (session) {
            statusMessage.textContent = 'Image ready. Click "Analyze Image" to begin!';
        }
    }
});


detectButton.addEventListener('click', async () => {
    if (!imageUrl || !session || !imageFile) return;

    detectButton.disabled = true;
    statusMessage.textContent = 'Analyzing... This may take a few seconds.';
    resultsContainer.style.display = 'block';
    
    // Clear previous results
    metadataResults.innerHTML = 'Running metadata scan...';
    mlResults.innerHTML = 'Preparing image for model...';
    overallAssessment.textContent = '';
    technicalDetails.textContent = '';

    // Stage 1: Metadata Analysis
    analyzeMetadata(imageFile, (metadata) => {
        displayMetadata(metadata);

        // Stage 2: ML Analysis
        runMLAnalysis().then(mlPrediction => {
            displayMLResults(mlPrediction);
            
            // Stage 3: Final Verdict
            createOverallAssessment(metadata, mlPrediction);
            
            statusMessage.textContent = 'Analysis complete!';
            detectButton.disabled = false;
        }).catch(err => {
            console.error('ML Analysis failed', err);
            mlResults.innerHTML = `<p style="color: red;">ML analysis failed.</p>`;
            statusMessage.textContent = 'Error during ML analysis.';
            detectButton.disabled = false;
        });
    });
});

async function runMLAnalysis() {
    const image = new Image();
    image.src = imageUrl;
    
    return new Promise((resolve, reject) => {
        image.onload = async () => {
            try {
                statusMessage.textContent = 'Preprocessing image for AI model...';
                const tensor = await preprocess(image);
                
                statusMessage.textContent = 'Running image through the neural network...';
                const feeds = { 'input': tensor };
                const results = await session.run(feeds);
                const output = results.output.data; // Model output: [real_score, artificial_score]
                
                // Softmax to get probabilities
                const expScores = output.map(score => Math.exp(score));
                const sumExpScores = expScores.reduce((a, b) => a + b, 0);
                const probabilities = expScores.map(score => score / sumExpScores);

                resolve({ real: probabilities[0], artificial: probabilities[1] });
            } catch(e) {
                reject(e);
            }
        };
        image.onerror = reject;
    });
}


function displayMetadata(data) {
    metadataResults.innerHTML = '';
    if (Object.keys(data.details).length === 0) {
        metadataResults.innerHTML = '<p>No relevant EXIF metadata found.</p>';
    } else {
        for(const [key, value] of Object.entries(data.details)) {
            const item = document.createElement('div');
            item.className = 'metadata-item';
            if (value.toString().startsWith('!!')) {
                item.classList.add('highlight');
            }
            item.innerHTML = `<span>${key}</span><span>${value.toString().replace('!! AI Marker Found: ', '')}</span>`;
            metadataResults.appendChild(item);
        }
    }
    technicalDetails.textContent = `--- RAW EXIF DATA ---\n${data.raw}`;
}


function displayMLResults(data) {
    mlResults.innerHTML = `
        <div class="ml-item">
            <span>AI-Generated Confidence</span>
            <span>${(data.artificial * 100).toFixed(2)}%</span>
        </div>
        <div class="ml-item">
            <span>Real Photo Confidence</span>
            <span>${(data.real * 100).toFixed(2)}%</span>
        </div>
    `;
}

function createOverallAssessment(metadata, mlPrediction) {
    const aiConfidence = mlPrediction.artificial;
    let verdict = '';
    let className = 'uncertain';

    if (metadata.hasAiMarkers) {
        verdict = 'Verdict: Strong indication of AI Generation (AI markers found in metadata).';
        className = 'likely-ai';
    } else if (aiConfidence > 0.85) {
        verdict = 'Verdict: Very Likely AI-Generated (High confidence from ML model).';
        className = 'likely-ai';
    } else if (aiConfidence > 0.6) {
        verdict = 'Verdict: Likely AI-Generated (Moderate confidence from ML model).';
        className = 'likely-ai';
    } else if (aiConfidence < 0.15) {
        verdict = 'Verdict: Very Likely a Real Photo (High confidence from ML model).';
        className = 'likely-real';
    } else if (aiConfidence < 0.4) {
        verdict = 'Verdict: Likely a Real Photo (Moderate confidence from ML model).';
        className = 'likely-real';
    } else {
        verdict = "Verdict: Uncertain (ML model is not confident, and no metadata markers were found).";
        className = 'uncertain';
    }
    
    overallAssessment.textContent = verdict;
    overallAssessment.className = '';
    overallAssessment.classList.add(className);
}

// --- Initialization ---
document.addEventListener('DOMContentLoaded', initializeModel);


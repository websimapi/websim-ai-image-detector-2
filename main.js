// DOM Elements
const imageUploadInput = document.getElementById('image-upload-input');
const imagePreview = document.getElementById('image-preview');
const detectButton = document.getElementById('detect-button');
const statusMessage = document.getElementById('status-message');
const resultsContainer = document.getElementById('results-container');
const progressBar = document.getElementById('progress-bar');
const progressBarContainer = document.getElementById('progress-bar-container');

// State
let session = null;
let imageFile = null;
let imageUrl = null;

// AI Detection Keywords for metadata
const AI_INDICATORS = [
    'midjourney', 'dall-e', 'dalle', 'stable diffusion', 'firefly',
    'openai', 'generative', 'ai generated', 'synthetic', 'artifical intelligence'
];

// ONNX Model Loading
const MODEL_URL = 'https://huggingface.co/Organika/sdxl-detector/resolve/main/onnx/model.onnx';

const loadModel = async () => {
    statusMessage.textContent = 'Loading AI detection model... (downloading ~354 MB, may take 2-3 minutes)';
    progressBarContainer.style.display = 'block';
    
    try {
        // Load ONNX Runtime
        const ort = window.ort;

        // Configure WASM for better cross-browser compatibility
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true; // Enable SIMD if available
        
        // Set proxy to CDN for better compatibility
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
        
        // Create session with progress tracking
        progressBar.style.width = '10%';
        statusMessage.textContent = 'Downloading model from HuggingFace...';
        
        session = await ort.InferenceSession.create(MODEL_URL, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'basic',
            executionMode: 'sequential',
            enableMemPattern: false,
            enableCpuMemArena: false
        });
        
        console.log('Model loaded. Input names:', session.inputNames);
        console.log('Model loaded. Output names:', session.outputNames);
        
        progressBar.style.width = '100%';
        statusMessage.textContent = 'Model loaded successfully!';
        
        setTimeout(() => {
            progressBarContainer.style.display = 'none';
            statusMessage.textContent = 'Please upload an image to begin.';
        }, 1000);
        
    } catch (error) {
        console.error('Error loading model:', error);
        statusMessage.textContent = 'Error loading model. Please refresh and try again.';
    }
};

// Extract and analyze image metadata
const analyzeMetadata = async (file) => {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                EXIF.getData(img, function() {
                    const allTags = EXIF.getAllTags(this);
                    const findings = {
                        software: null,
                        creator: null,
                        copyright: null,
                        aiIndicators: [],
                        hasC2PA: false,
                        creationDate: null,
                        allMetadata: allTags
                    };
                    
                    // Check common EXIF fields
                    if (allTags.Software) findings.software = allTags.Software;
                    if (allTags.Artist) findings.creator = allTags.Artist;
                    if (allTags.Copyright) findings.copyright = allTags.Copyright;
                    if (allTags.DateTime) findings.creationDate = allTags.DateTime;
                    
                    // Check for AI indicators in metadata
                    const metadataString = JSON.stringify(allTags).toLowerCase();
                    AI_INDICATORS.forEach(indicator => {
                        if (metadataString.includes(indicator)) {
                            findings.aiIndicators.push(indicator);
                        }
                    });
                    
                    // Check for C2PA/Content Credentials
                    if (allTags['C2PA'] || metadataString.includes('c2pa') || 
                        metadataString.includes('content credentials')) {
                        findings.hasC2PA = true;
                    }
                    
                    resolve(findings);
                });
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
};

// Preprocess image for Swin Transformer model (384x384, normalized)
const preprocessImage = async (imageUrl) => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            try {
                console.log("Image loaded for preprocessing. Dimensions:", img.width, "x", img.height);
                const canvas = document.createElement('canvas');
                canvas.width = 384;
                canvas.height = 384;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, 384, 384);
                console.log("Image drawn to 384x384 canvas.");
                
                const imageData = ctx.getImageData(0, 0, 384, 384);
                const pixels = imageData.data;
                console.log("Got image data from canvas. Pixel array length:", pixels.length);
                
                // Prepare tensor data [1, 3, 384, 384]
                const float32Data = new Float32Array(3 * 384 * 384);
                
                // ImageNet normalization values
                const mean = [0.485, 0.456, 0.406];
                const std = [0.229, 0.224, 0.225];
                
                for (let i = 0; i < 384 * 384; i++) {
                    // RGB channels
                    float32Data[i] = (pixels[i * 4] / 255 - mean[0]) / std[0];     // R
                    float32Data[384 * 384 + i] = (pixels[i * 4 + 1] / 255 - mean[1]) / std[1]; // G
                    float32Data[384 * 384 * 2 + i] = (pixels[i * 4 + 2] / 255 - mean[2]) / std[2]; // B
                }
                
                console.log("Image normalization complete. Tensor data length:", float32Data.length);
                resolve(float32Data);
            } catch (e) {
                console.error("Error during image preprocessing:", e);
                reject(new Error("Failed to process image data."));
            }
        };
        img.onerror = (err) => {
            console.error("Error loading image for preprocessing:", err);
            reject(new Error("Could not load the image. It might be corrupt or in an unsupported format."));
        };
        img.src = imageUrl;
    });
};

const initializeModel = async () => {
    await loadModel();
};

// Image Upload Handler
imageUploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        // Add file size warning for large images
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            statusMessage.textContent = 'Warning: Large file (>10MB). Analysis may be slow or fail.';
            statusMessage.style.color = '#ffc107';
        } else {
            statusMessage.style.color = ''; // Reset color
        }

        imageFile = file; // Store the file for metadata extraction
        if (imageUrl) {
            URL.revokeObjectURL(imageUrl);
        }
        imageUrl = URL.createObjectURL(file);
        imagePreview.src = imageUrl;
        imagePreview.style.display = 'block';
        imagePreview.classList.add('has-image');
        
        detectButton.disabled = false;
        resultsContainer.innerHTML = '';
        if (session) {
            statusMessage.textContent = 'Image ready. Click "Analyze Image" to begin!';
        }
    }
});

// Detection Handler
detectButton.addEventListener('click', async () => {
    if (!imageUrl || !session || !imageFile) return;

    detectButton.disabled = true;
    statusMessage.textContent = 'Step 1/2: Analyzing metadata...';
    resultsContainer.innerHTML = '';
    console.log("Starting analysis...");

    let tensor;
    try {
        // Attempt to trigger garbage collection before heavy processing
        if (window.gc) {
            window.gc();
            console.log("Garbage collection triggered.");
        }

        // Step 1: Analyze metadata
        console.log("Analyzing metadata for file:", imageFile.name);
        const metadata = await analyzeMetadata(imageFile);
        console.log("Metadata analysis complete:", metadata);
        
        // Step 2: ML-based detection
        statusMessage.textContent = 'Step 2/2: Running AI detection model...';
        console.log("Preprocessing image...");
        const preprocessed = await preprocessImage(imageUrl);
        console.log("Image preprocessing complete. Preprocessed data length:", preprocessed.length);
        
        console.log("Creating ONNX tensor...");
        tensor = new window.ort.Tensor('float32', preprocessed, [1, 3, 384, 384]);
        console.log("Tensor created:", tensor);
        
        const feeds = { pixel_values: tensor };
        console.log("Running model inference with feeds:", feeds);
        
        let results;
        try {
            results = await session.run(feeds);
        } catch (inferenceError) {
            console.error("Error during session.run():", inferenceError);
            if (inferenceError.message.includes('726881928')) { // Specific memory error code
                 throw new Error("Out of memory during model inference. Please try a smaller image or a device with more RAM.");
            }
            throw new Error("Model inference failed. The model may be incompatible with your browser's environment.");
        }

        console.log("Model inference complete. Results:", results);
        
        // FIX: The output name from this model is 'output_0', not 'logits'.
        const outputTensor = results[session.outputNames[0]]; 
        if (!outputTensor) {
            console.error("Could not find the expected output tensor in the results object.", results);
            throw new Error("Model output is not in the expected format.");
        }
        const logits = outputTensor.data;
        console.log("Extracted logits:", logits);
        
        // Softmax to get probabilities
        const exp = logits.map(x => Math.exp(x));
        const sumExp = exp.reduce((a, b) => a + b, 0);
        const probabilities = exp.map(x => x / sumExp);
        
        // Classes: 0 = artificial, 1 = real
        const artificialScore = probabilities[0];
        const realScore = probabilities[1];
        
        displayResults(metadata, artificialScore, realScore);
        statusMessage.textContent = 'Analysis complete! Upload another image to test.';
    } catch (error) {
        console.error('Error during analysis:', error);
        if (error.stack) {
            console.error("Stack trace:", error.stack);
        }
        statusMessage.textContent = 'Error analyzing image. Please try another image.';
        const errorMessage = error instanceof Error ? error.message : String(error);
        resultsContainer.innerHTML = '<p style="color: #dc3545; text-align: center;">Analysis failed: ' + errorMessage + '</p>';
    } finally {
        // Dispose of the tensor to free up memory
        if (tensor) {
            tensor.dispose();
            console.log("Tensor disposed.");
        }
        detectButton.disabled = false;
    }
});

const displayResults = (metadata, artificialScore, realScore) => {
    resultsContainer.innerHTML = '';
    
    const artificialPercentage = (artificialScore * 100).toFixed(1);
    const realPercentage = (realScore * 100).toFixed(1);
    
    // === METADATA ANALYSIS SECTION ===
    const metadataSection = document.createElement('div');
    metadataSection.style.cssText = 'background: #f8f9ff; border-radius: 10px; padding: 20px; margin-bottom: 20px; border-left: 4px solid #667eea;';
    
    const metadataTitle = document.createElement('h3');
    metadataTitle.style.cssText = 'color: #667eea; margin-bottom: 15px; font-size: 1.1rem;';
    metadataTitle.innerHTML = '🔍 Metadata Analysis';
    metadataSection.appendChild(metadataTitle);
    
    let metadataVerdict = 'No AI indicators found in metadata';
    let metadataIcon = '✅';
    let metadataColor = '#51cf66';
    
    if (metadata.aiIndicators.length > 0 || metadata.hasC2PA) {
        metadataVerdict = 'AI indicators detected in metadata!';
        metadataIcon = '⚠️';
        metadataColor = '#ff6b6b';
    }
    
    const metadataVerdictDiv = document.createElement('div');
    metadataVerdictDiv.style.cssText = `background: ${metadataColor}; color: white; padding: 12px; border-radius: 8px; margin-bottom: 15px; font-weight: 600;`;
    metadataVerdictDiv.textContent = `${metadataIcon} ${metadataVerdict}`;
    metadataSection.appendChild(metadataVerdictDiv);
    
    // Metadata details
    const detailsList = document.createElement('div');
    detailsList.style.cssText = 'font-size: 0.9rem; color: #555;';
    
    if (metadata.software) {
        detailsList.innerHTML += `<p><strong>Software:</strong> ${metadata.software}</p>`;
    }
    if (metadata.creator) {
        detailsList.innerHTML += `<p><strong>Creator:</strong> ${metadata.creator}</p>`;
    }
    if (metadata.creationDate) {
        detailsList.innerHTML += `<p><strong>Date:</strong> ${metadata.creationDate}</p>`;
    }
    if (metadata.hasC2PA) {
        detailsList.innerHTML += `<p><strong>C2PA:</strong> Content Credentials detected</p>`;
    }
    if (metadata.aiIndicators.length > 0) {
        detailsList.innerHTML += `<p><strong>AI Keywords Found:</strong> ${metadata.aiIndicators.join(', ')}</p>`;
    }
    if (!metadata.software && !metadata.creator && metadata.aiIndicators.length === 0) {
        detailsList.innerHTML += `<p style="color: #999;">No significant metadata found (may have been stripped)</p>`;
    }
    
    metadataSection.appendChild(detailsList);
    resultsContainer.appendChild(metadataSection);
    
    // === ML MODEL SECTION ===
    const mlSection = document.createElement('div');
    mlSection.style.cssText = 'background: #f8f9ff; border-radius: 10px; padding: 20px; margin-bottom: 20px; border-left: 4px solid #764ba2;';
    
    const mlTitle = document.createElement('h3');
    mlTitle.style.cssText = 'color: #764ba2; margin-bottom: 15px; font-size: 1.1rem;';
    mlTitle.innerHTML = '🤖 ML Model Analysis (Organika/sdxl-detector)';
    mlSection.appendChild(mlTitle);
    
    // ML Verdict
    const verdictDiv = document.createElement('div');
    verdictDiv.className = 'ai-indicator';
    
    if (artificialScore > 0.7) {
        verdictDiv.classList.add('likely-ai');
        verdictDiv.innerHTML = `🤖 Likely AI-Generated (${artificialPercentage}% confidence)`;
    } else if (realScore > 0.7) {
        verdictDiv.classList.add('likely-real');
        verdictDiv.innerHTML = `📷 Likely Real Photo (${realPercentage}% confidence)`;
    } else {
        verdictDiv.classList.add('uncertain');
        verdictDiv.innerHTML = `🤔 Uncertain - Scores: AI ${artificialPercentage}% / Real ${realPercentage}%`;
    }
    
    mlSection.appendChild(verdictDiv);
    
    // Confidence bars
    const confidenceBars = document.createElement('div');
    confidenceBars.style.cssText = 'margin-top: 15px;';
    
    confidenceBars.innerHTML = `
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 600;">AI-Generated</span>
                <span style="font-weight: 700; color: #ff6b6b;">${artificialPercentage}%</span>
            </div>
            <div style="background: #e9ecef; border-radius: 10px; height: 10px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #ff6b6b, #ee5a6f); width: ${artificialPercentage}%; height: 100%; border-radius: 10px;"></div>
            </div>
        </div>
        <div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 600;">Real Photo</span>
                <span style="font-weight: 700; color: #51cf66;">${realPercentage}%</span>
            </div>
            <div style="background: #e9ecef; border-radius: 10px; height: 10px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #51cf66, #37b24d); width: ${realPercentage}%; height: 100%; border-radius: 10px;"></div>
            </div>
        </div>
    `;
    
    mlSection.appendChild(confidenceBars);
    resultsContainer.appendChild(mlSection);
    
    // === FINAL VERDICT ===
    const finalSection = document.createElement('div');
    finalSection.style.cssText = 'background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 10px; padding: 20px; text-align: center;';
    
    const finalTitle = document.createElement('h3');
    finalTitle.style.cssText = 'margin-bottom: 10px; font-size: 1.2rem;';
    finalTitle.textContent = '📊 Overall Assessment';
    finalSection.appendChild(finalTitle);
    
    let finalVerdict = '';
    const metadataHasAI = metadata.aiIndicators.length > 0 || metadata.hasC2PA;
    const mlSaysAI = artificialScore > realScore;
    
    if (metadataHasAI && mlSaysAI) {
        finalVerdict = '⚠️ Strong Evidence: Both metadata and ML analysis indicate AI generation';
    } else if (metadataHasAI) {
        finalVerdict = '⚠️ Metadata indicates AI, but ML analysis is inconclusive';
    } else if (mlSaysAI && artificialScore > 0.8) {
        finalVerdict = '🤖 ML model strongly suggests AI generation (no metadata available)';
    } else if (mlSaysAI) {
        finalVerdict = '🤔 ML model leans toward AI generation, but confidence is moderate';
    } else {
        finalVerdict = '✅ Appears to be a real photograph based on available evidence';
    }
    
    const finalVerdictP = document.createElement('p');
    finalVerdictP.style.cssText = 'font-size: 1rem; line-height: 1.6;';
    finalVerdictP.textContent = finalVerdict;
    finalSection.appendChild(finalVerdictP);
    
    resultsContainer.appendChild(finalSection);
};

// Initialize on load
initializeModel();
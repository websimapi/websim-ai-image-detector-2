/* ...existing code... */
// --- Model Loading ---

class AiImageDetectorPipeline {
    static task = 'image-classification';
    static model = 'Xenova/vit-base-patch16-224';
    static instance = null;

    static async getInstance(progress_callback = null) {
/* ...existing code... */
/* ...existing code... */
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

// --- Initialization ---
/* ...existing code... */


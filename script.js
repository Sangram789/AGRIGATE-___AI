document.addEventListener('DOMContentLoaded', () => {

    const userImage = document.getElementById('user-image');
    const imageUpload = document.getElementById('image-upload');
    const btnReset = document.getElementById('btn-reset');
    const btnAnalyze = document.getElementById('btn-analyze');
    const labelContainer = document.getElementById('label-container');
    const diagnosisBox = document.getElementById('diagnosis');
    const statusBox = document.getElementById('status-box');

    let capturedImageBlob = null;

    // Upload image
    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            userImage.src = event.target.result;
            userImage.style.display = "block";
            btnReset.style.display = "inline-flex";
            btnAnalyze.style.display = "inline-flex";
            labelContainer.innerText = "Image Loaded Successfully";
            statusBox.innerText = "Ready to scan";
            capturedImageBlob = file;
        };
        reader.readAsDataURL(file);
    });

    // Analyze image
    btnAnalyze.addEventListener('click', async () => {
        if (!capturedImageBlob) {
            alert("Please select an image first!");
            return;
        }

        const formData = new FormData();
        formData.append('file', capturedImageBlob); // Ensure 'file' key matches Flask

        labelContainer.innerText = "Analyzing...";
        statusBox.innerText = "Processing...";
        btnAnalyze.disabled = true;
        btnAnalyze.innerText = "🔄 Analyzing...";

        try {
            const res = await fetch('/predict', { method: 'POST', body: formData });
            const data = await res.json();

            btnAnalyze.disabled = false;
            btnAnalyze.innerText = "🔍 Analyze Image";

            if (data.error) {
                labelContainer.innerText = "Analysis Failed!";
                statusBox.innerText = "Error";
                alert("Error: " + data.error);
            } else {
                labelContainer.innerText = `Detected: ${data.disease} (${data.confidence})`;
                diagnosisBox.innerText = `Action Plan for ${data.disease}:\n${data.solution}`;
                statusBox.innerText = "Analysis Complete";
                diagnosisBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        } catch (err) {
            console.error(err);
            btnAnalyze.disabled = false;
            btnAnalyze.innerText = "🔍 Analyze Image";
            labelContainer.innerText = "Ready to scan...";
            statusBox.innerText = "Error";
            alert("Analysis failed. Check console.");
        }
    });

    // Reset UI
    btnReset.addEventListener('click', () => {
        userImage.style.display = "none";
        userImage.src = "#";
        capturedImageBlob = null;
        btnReset.style.display = "none";
        btnAnalyze.style.display = "none";
        labelContainer.innerText = "Ready to scan...";
        diagnosisBox.innerText = "Detection results will appear here...";
        statusBox.innerText = "System Ready";
    });

});
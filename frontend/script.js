// Model TensorFlow.js tempatan (dieksport dari Teachable Machine -> "Tensorflow.js"
// -> Download, BUKAN shareable link). Fail model.json, metadata.json, weights.bin
// diletakkan dalam folder frontend/model/. Path relatif ni cuma berfungsi bila
// dibuka melalui HTTP (cth: Live Server) - bukan buka terus fail index.html.
const MODEL_URL = "./model/";

// URL backend FastAPI (jalankan backend/main.py dahulu)
const BACKEND_URL = "http://127.0.0.1:8000";

let model, webcam, maxPredictions;
let isWebcamActive = false;

// Memuatkan model AI (untuk mod webcam - inferens terus dalam browser)
async function init() {
    const modelURL = MODEL_URL + "model.json";
    const metadataURL = MODEL_URL + "metadata.json";

    try {
        model = await tmImage.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();
        console.log("Model AI Berjaya Dimuatkan!");
    } catch (e) {
        console.error("Gagal memuatkan model:", e);
    }
}

// Fungsi menjalankan Webcam (mod: Browser / TensorFlow.js, real-time)
async function startCamera() {
    if (!model) await init();
    if (isWebcamActive) return;

    resetApp();
    const flip = true;
    webcam = new tmImage.Webcam(400, 400, flip);
    await webcam.setup();
    await webcam.play();
    isWebcamActive = true;

    document.getElementById("webcam-container").appendChild(webcam.canvas);
    setSourceLabel("Browser (TensorFlow.js)");
    window.requestAnimationFrame(loop);
}

async function loop() {
    if (isWebcamActive) {
        webcam.update();
        const prediction = await model.predict(webcam.canvas);
        renderPrediction(pickHighest(prediction));
        window.requestAnimationFrame(loop);
    }
}

// Fungsi membaca Muat Naik Imej -> dihantar ke BACKEND FastAPI (/predict)
// supaya ujian deployment sebenar (seksyen 9.0 tugasan) dapat ditunjukkan.
document.getElementById("imageUpload").addEventListener("change", async function (e) {
    if (e.target.files.length === 0) return;

    const file = e.target.files[0];
    resetApp();

    const img = document.getElementById("image-preview");
    // "URL" di sini ialah Web API global (bukan pembolehubah kita - tu MODEL_URL).
    // Dulu kod guna nama "URL" untuk pembolehubah model, yang menimpa API ni dan
    // menyebabkan baris ini gagal senyap -> upload nampak "tak jadi apa-apa".
    img.src = URL.createObjectURL(file);
    img.style.display = "block";

    setSourceLabel("Server (FastAPI)");
    await predictViaBackend(file);
});

async function predictViaBackend(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${BACKEND_URL}/predict`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.detail || `HTTP ${response.status}`);
        }

        const result = await response.json();
        renderPrediction({
            className: result.prediction,
            probability: result.confidence,
        });
    } catch (err) {
        console.error("Ralat memanggil backend:", err);
        document.getElementById("prediction").innerText = "RALAT SERVER";
        document.getElementById("confidence").innerText = "-";
        document.getElementById("progress-bar").style.width = "0%";
        setSourceLabel(`Server tidak sambung (${err.message})`);
    }
}

// Cari prediction dengan probability tertinggi (untuk mod browser/webcam)
function pickHighest(predictionArray) {
    let highest = predictionArray[0];
    for (let i = 1; i < predictionArray.length; i++) {
        if (predictionArray[i].probability > highest.probability) {
            highest = predictionArray[i];
        }
    }
    return highest;
}

// Paparkan hasil ramalan pada UI, dengan pengendalian threshold confidence
const CONFIDENCE_THRESHOLD = 0.70;

function renderPrediction({ className, probability }) {
    const confidencePercent = (probability * 100).toFixed(2) + "%";

    if (probability >= CONFIDENCE_THRESHOLD) {
        document.getElementById("prediction").innerText = className;
        document.getElementById("progress-bar").style.backgroundColor = "#10b981";
    } else {
        document.getElementById("prediction").innerText = "UNKNOWN / UNCERTAIN";
        document.getElementById("progress-bar").style.backgroundColor = "#f59e0b";
    }

    document.getElementById("confidence").innerText = confidencePercent;
    document.getElementById("progress-bar").style.width = confidencePercent;
}

function setSourceLabel(text) {
    const el = document.getElementById("source-label");
    if (el) el.innerText = `Sumber: ${text}`;
}

// Fungsi Reset
function resetApp() {
    if (isWebcamActive && webcam) {
        webcam.stop();
        isWebcamActive = false;
    }
    document.getElementById("webcam-container").innerHTML = "";

    const img = document.getElementById("image-preview");
    img.src = "#";
    img.style.display = "none";

    document.getElementById("prediction").innerText = "-";
    document.getElementById("confidence").innerText = "0%";
    document.getElementById("progress-bar").style.width = "0%";
    document.getElementById("imageUpload").value = "";
    setSourceLabel("-");
}

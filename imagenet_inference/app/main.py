# app.py
import io
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel


# -------------------------
# Paths / Assets
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resnet50_imagenet_1k_final.onnx"
LABELS_PATH = BASE_DIR / "labels.txt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"ONNX model not found at: {MODEL_PATH}")

if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels file not found at: {LABELS_PATH}")


# -------------------------
# Model config
# -------------------------
INPUT_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Load labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS: List[str] = [line.strip() for line in f]

# Load ONNX model
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


# -------------------------
# Helpers
# -------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(INPUT_SIZE)

    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD

    # HWC -> CHW, add batch dim
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    # stable softmax
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def infer(image_bytes: bytes, top_k: int = 5) -> List[dict]:
    image = Image.open(io.BytesIO(image_bytes))
    x = preprocess_image(image)

    logits = session.run(None, {input_name: x})[0]  # shape (1, 1000)
    probs = softmax(logits)[0]

    idx = np.argsort(-probs)[:top_k]
    out = []
    for i in idx:
        out.append(
            {
                "index": int(i),
                "label": LABELS[i] if i < len(LABELS) else str(i),
                "probability": float(probs[i]),
            }
        )
    return out


# -------------------------
# API schema
# -------------------------
class PredictionResponse(BaseModel):
    predictions: List[dict]
    success: bool
    message: str


# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(
    title="ImageNet 1K Classifier",
    description="FastAPI + ONNXRuntime + custom HTML UI (Lambda-friendly)",
    version="1.0.0",
)

# In production, restrict origins to your UI domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/ui")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Multipart form-data:
      - field name: file
      - content: image bytes
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        preds = infer(image_bytes, top_k=5)
        return PredictionResponse(
            predictions=preds,
            success=True,
            message="Classification successful",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """
    A simple Gradio-like UI:
    - choose image
    - preview
    - submit -> POST /predict (multipart)
    - render top-5 predictions
    """
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ImageNet 1K Classifier</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 24px; }
    .wrap { max-width: 900px; margin: 0 auto; }
    .card { border: 1px solid rgba(0,0,0,.12); border-radius: 14px; padding: 18px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
    .btn { border: 0; border-radius: 10px; padding: 10px 14px; font-weight: 600; cursor: pointer; }
    .primary { background: #2563eb; color: white; }
    .secondary { background: rgba(0,0,0,.06); }
    .drop { border: 1px dashed rgba(0,0,0,.3); border-radius: 12px; padding: 12px; }
    .preview { width: 100%; max-height: 320px; object-fit: contain; border-radius: 12px; border: 1px solid rgba(0,0,0,.12); }
    .muted { opacity: .75; }
    .err { color: #dc2626; white-space: pre-wrap; }
    code { background: rgba(0,0,0,.06); padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h2 style="margin:0 0 8px 0;">ImageNet 1K Classifier</h2>
      <p class="muted" style="margin:0 0 16px 0;">
        Upload an image and get top-5 probabilities.
        (Lambda-friendly: single multipart request to <code>/predict</code>)
      </p>

      <div class="grid">
        <div>
          <div class="drop">
            <input id="file" type="file" accept="image/*" />
            <div style="margin-top:12px; display:flex; gap:10px;">
              <button class="btn primary" id="run">Submit</button>
              <button class="btn secondary" id="clear">Clear</button>
            </div>
          </div>
          <div id="status" class="muted" style="margin-top:10px;"></div>
          <div id="err" class="err" style="margin-top:10px;"></div>
        </div>

        <div>
          <img id="preview" class="preview" alt="Preview"/>
          <h3 style="margin:12px 0 8px 0;">Top-5 Predictions</h3>
          <ol id="out"></ol>
        </div>
      </div>

      <hr style="margin:18px 0; opacity:.25;">
      <p class="muted" style="margin:0;">
        API: <code>POST /predict</code> (multipart field: <code>file</code>) ·
        Health: <code>GET /health</code>
      </p>
    </div>
  </div>

<script>
const fileEl = document.getElementById("file");
const previewEl = document.getElementById("preview");
const outEl = document.getElementById("out");
const errEl = document.getElementById("err");
const statusEl = document.getElementById("status");
const runBtn = document.getElementById("run");
const clearBtn = document.getElementById("clear");

let objUrl = null;

fileEl.addEventListener("change", () => {
  errEl.textContent = "";
  outEl.innerHTML = "";
  statusEl.textContent = "";
  if (objUrl) URL.revokeObjectURL(objUrl);

  const f = fileEl.files && fileEl.files[0];
  if (!f) {
    previewEl.removeAttribute("src");
    return;
  }
  objUrl = URL.createObjectURL(f);
  previewEl.src = objUrl;
});

clearBtn.addEventListener("click", () => {
  errEl.textContent = "";
  outEl.innerHTML = "";
  statusEl.textContent = "";
  fileEl.value = "";
  if (objUrl) URL.revokeObjectURL(objUrl);
  previewEl.removeAttribute("src");
});

runBtn.addEventListener("click", async () => {
  errEl.textContent = "";
  outEl.innerHTML = "";

  const f = fileEl.files && fileEl.files[0];
  if (!f) {
    errEl.textContent = "Please choose an image first.";
    return;
  }

  statusEl.textContent = "Running inference…";
  runBtn.disabled = true;

  try {
    const form = new FormData();
    form.append("file", f);

    const res = await fetch("/predict", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      errEl.textContent = data?.detail ? JSON.stringify(data.detail, null, 2) : JSON.stringify(data, null, 2);
      statusEl.textContent = "";
      return;
    }

    const preds = data.predictions || [];
    preds.forEach(p => {
      const li = document.createElement("li");
      li.innerHTML = `<b>${p.label}</b> — ${(p.probability * 100).toFixed(2)}%`;
      outEl.appendChild(li);
    });

    statusEl.textContent = "Done.";
  } catch (e) {
    errEl.textContent = String(e);
    statusEl.textContent = "";
  } finally {
    runBtn.disabled = false;
  }
});
</script>
</body>
</html>
        """
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
import torch
import traceback
from src.model import load_model, get_preprocessor
from src.detection import finding_waldo
from typing import List, Optional
import os
from fastapi.responses import RedirectResponse


app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "waldo_detector_64x64_v1.0.pth")
model = load_model(model_path, device)
preprocessor = get_preprocessor()

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.post("/detect")
async def detect_waldo(
    file: UploadFile = File(...),
    n_show: int = Query(3, description="Number of detections to show"),
    threshold: float = Query(0.0, description="Confidence threshold"),
    red_threshold: float = Query(0.1, description="Red color threshold"),
    red_factor: float = Query(1.5, description="Red color factor"),
    stride: int = Query(16, description="Stride for sliding window"),
    scales: Optional[List[float]] = Query([1.0], description="List of scales for pyramid search")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    try:
        image_with_boxes, boxes, scores = finding_waldo(
            image_original=image,
            model=model,
            preprocessing_method=preprocessor,
            device=device,
            n_show=n_show,
            threshold=threshold,
            red_threshold=red_threshold,
            red_factor=red_factor,
            stride=stride,
            scales=scales
        )
    except Exception as e:
        tb_str = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb_str})

    buf = io.BytesIO()
    image_with_boxes.save(buf, format="PNG")
    buf.seek(0)

    detections = []
    for box, score in zip(boxes.tolist(), scores.tolist()):
        detections.append({
            "box": [int(coord) for coord in box],
            "score": float(score)
        })

    return StreamingResponse(buf, media_type="image/png")





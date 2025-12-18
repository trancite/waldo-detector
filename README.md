# 🕵️‍♂️ Waldo Detector

This project trains a convolutional neural network to detect Waldo in image patches and locate him in full-size images using a multi-scale sliding window approach.

---

## 📁 Project Structure

```
project_root/
├── data/
│   ├── original-images/       # Original full-size images (unmodified)
│   ├── raw_data/              # Patches extracted from original-images
│   ├── expanded_data/         # Dataset after adding/removing patches
│   ├── split_dataset/         # Training/testing set with augmentation applied
│   └── Validation_Set/        # Full-size images used for sliding window validation
│
├── notebooks/
│   ├── dataset_extension.ipynb         # Manual cropping of Waldo patches
│   ├── dataset_extension.pdf
│
│   ├── training_and_augmentation.ipynb # Model training with augmentation
│   ├── training_and_augmentation.pdf
│
│   └── sliding_window_detection.ipynb  # Sliding window inference
│   └── sliding_window_detection.pdf
│
├── src/
│   ├── model.py       # Model architecture and loading utilities
│   ├── detection.py   # Inference pipeline and sliding window logic
│   └── main.py        # FastAPI app entry point (asegúrate de que es así)
│
├── model/
│   └── waldo_detector_64x64.pth # Trained model weights
│
│
├── docs/
│   ├── Dataset_Preparation.md
│   ├── 64x64_Model_Design_and_Training.md
│   └── Sliding_Window_Validation.md
│
├── README.md
├── requirements.txt
└── Dockerfile

```

---

## 🧠 Methodology

1. **Dataset Preprocessing**: Manual dataset expansion and extensive augmentation  
2. **Patch Classification**: MobileNetV2-based CNN classifies 64×64 image patches  
3. **Pyramid of Scales Search**: Sliding window detection across multiple scales  
4. **Detection Refinement**: Non-Maximum Suppression (NMS) for precise localization  


---

## 📊 Dataset

Public Waldo dataset enhanced with manual patches and augmentation:

| Class      | Original | Modified | **Train Set**     |
|------------|----------|----------|------------------|
| Waldo      | 39       | 32       |  1,300 (×50)       |
| NotWaldo   | 5,337    | 5,337    | 12,807 (×3)      |


**Augmentation Techniques:**  
Rotation, flipping, brightness/contrast adjustment, and random cropping

---

## 🏋️‍♂️ Model Training

- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Input Processing**:  
  64×64 patches → Resize to 256×256 → Center crop to 224×224
- **Fine-tuning**: Last convolutional block + classification head
- **Loss**: Focal Loss (addresses class imbalance by down-weighting easy negatives)
- **Training**: 20 epochs | Batch size 32 | AdamW optimizer (lr=1e-3)

---

## 🚀 Inference Pipeline

Detects Waldo in full-resolution images through:

1. **Pyramid Selection**: Choose the set of scales (pyramid) for detection (default = [1.0])  
2. **Sliding Window**:  
   - Window size: 64×64 pixels (adjusted proportionally depending on the image scale)
   - Stride: 16 pixels (by default)
3. **Detection Filtering**:  
   - Confidence threshold: 0 (configurable; detects challenging Waldos)  
   - NMS with IoU threshold: 0 (configurable; removes duplicates)  
   - Red color filter: skips patches with insufficient red content to reduce false positives and speed up inference
    
4. **Output**:  
   - Original image with bounding box(es) around Waldo  
   - Zoomed crops of detections  
   - Confidence scores for each detection  

Example from the notebook:

![Detection Example](docs/Data Examples/example.png)

---

## ⚙️ FastAPI Web Service

This project includes a **FastAPI** web application that exposes the Waldo detection model through a simple REST API. The main features are:

- Accepts image uploads (PNG/JPG/JPEG) via a `/detect` POST endpoint  
- Supports configurable parameters like sliding window stride, scale pyramid, thresholds, and color filters  
- Returns the original image annotated with bounding boxes around detected Waldos  
- Optionally returns detection metadata (bounding boxes and confidence scores)  
- Includes an interactive Swagger UI at `/docs` for easy testing without coding  

The API is implemented in `src/main.py` and uses the trained model weights from `model/waldo_detector_64x64.pth`.  

**Start the server locally (requires FastAPI and Uvicorn installed in your Python environment):**

## ⚠️ Note

If you **don't have FastAPI or Uvicorn installed**, you **cannot run the API locally**.  
Instead, go to the **Docker section** below to run the API in a container with all dependencies pre-installed.

From the directory:

```bash
uvicorn src.main:app --reload
```

Access the docs here to test:

http://127.0.0.1:8000/docs

## 🐳 Docker Container

To simplify deployment and testing, the project includes a Dockerfile that builds a container with all dependencies installed:

- Uses `python:3.11-slim` as the base image  
- Installs required `Python` packages from requirements.txt  
- Copies source code and model weights into the container  
- Runs the FastAPI server on port 8000   

**Build the Docker image:**

```bash
docker build -t waldo-api .
```

**Run the container::**

```bash
docker run -p 8000:8000 waldo-api
```

Once running, the API is accessible on `http://localhost:8000` with the same endpoints and UI as the local version.

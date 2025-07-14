import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.ops import nms
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.model import load_model, get_preprocessor
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "waldo_detector_64x64_v1.0.pth")

Patch_Detector = load_model(model_path, device)
preprocesser = get_preprocessor()


def redness_dominance(pil_img, red_threshold=-1, red_factor=0):
    img = np.array(pil_img).astype(np.float32)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    red_dominant = (r > red_factor * g) & (r > red_factor * b)
    red_ratio = red_dominant.sum() / red_dominant.size
    return red_ratio > red_threshold


def sliding_window(
    image_original,
    model=Patch_Detector,
    preprocessing_method=preprocesser,
    device="cpu",
    stride=16,
    scales=[1],
    red_threshold=0.1,
    red_factor=1.5
):
    base_window_size = 64
    all_results = []
    width, height = image_original.size
    total_windows = 0
    sizes_per_scale = []
    counter = 1

    for scale in scales:
        window_size = int(base_window_size * 1/scale)
        stride_scaled = int(stride * 1/scale)
        rangex = (width - window_size) // stride_scaled + 1
        rangey = (height - window_size) // stride_scaled + 1
        total_windows += rangex * rangey
        sizes_per_scale.append((scale, window_size, stride_scaled, rangex, rangey))

    progress_bar = tqdm(total=total_windows, desc="Sliding Window Multi-Scale")

    for scale, window_size, stride_scaled, rangex, rangey in sizes_per_scale:
        for y in range(rangey):
            for x in range(rangex):
                x1 = x * stride_scaled
                y1 = y * stride_scaled
                x2 = x1 + window_size
                y2 = y1 + window_size

                if x2 > width:
                    x1 = width - window_size
                    x2 = width
                if y2 > height:
                    y1 = height - window_size
                    y2 = height

                crop = image_original.crop((x1, y1, x2, y2))
                crop_resized = crop.resize((base_window_size, base_window_size))

                if not redness_dominance(crop_resized, red_threshold, red_factor):
                    counter += 1
                    progress_bar.update(1)
                    continue

                input_tensor = preprocessing_method(crop_resized).unsqueeze(0).to(device)

                with torch.no_grad():
                    model.eval()
                    output = model(input_tensor)
                    prob = torch.sigmoid(output)[0].item()


                box_original_scale = [x1, y1, x2, y2]
                all_results.append([prob, box_original_scale, scale])

                progress_bar.update(1)

    progress_bar.close()
    all_results.sort(key=lambda x: x[0], reverse=True)
    print(counter)

    return all_results



from PIL import ImageDraw

def finding_waldo(
    image_original,
    model=Patch_Detector,
    preprocessing_method=preprocesser,
    device='cpu',
    n_show=5,
    threshold=0,
    red_threshold=0.1,
    red_factor=1.5,
    stride=16,
    scales=[1]
):
    sliding_results = sliding_window(
        image_original,
        model,
        preprocessing_method,
        device,
        stride,
        scales,
        red_threshold,
        red_factor
    )
    if len(sliding_results) == 0:
        raise ValueError("No valid patches found during sliding window inference.")

    boxes = torch.tensor([res[1] for res in sliding_results], dtype=torch.float32)
    scores = torch.tensor([res[0] for res in sliding_results])

    keep_indices = nms(boxes, scores, threshold)
    top_indices = keep_indices[:n_show]

    final_boxes = boxes[top_indices]
    final_scores = scores[top_indices]

    image_with_boxes = image_original.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    for box, score in zip(final_boxes, final_scores):
        x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=6)
        draw.rectangle([x1+3, y1+3, x2-3, y2-3], outline="darkred", width=2)
        draw.text((x1, y1 - 15), f"{score:.2f}", fill="red")

    output_path = "detections_output.png"
    image_with_boxes.save(output_path)
    print(f"Image saved in {output_path}")
    
    return image_with_boxes, final_boxes, final_scores




# Sliding Window Detection Strategy

## Detection Pipeline Overview

To locate Waldo in full-resolution images, we implemented a **sliding window detection pipeline** enhanced with a **scale pyramid**, **red color filtering**, and **Non-Maximum Suppression (NMS)**. This combination provides a scalable and efficient method for locating Waldo while remaining consistent with the model’s training distribution.

---

## Sliding Window with Scale Pyramid

The detector systematically scans the original image using a fixed-size window that moves horizontally and vertically across the entire area.

- The **base window size** is `64×64`, matching the patch size used during model training.
- To detect Waldos of varying apparent sizes, we apply a **scale pyramid**: the window size is scaled (e.g., `64`, `32`, `43` pixels), while the image itself remains unchanged.

> ⚠️ **Why scale the window and not the image?**  
> Resizing the full image alters its visual structure (e.g., texture, edge sharpness, proportions), potentially confusing the model. Scaling only the window keeps the image intact while allowing multi-scale detection.

---

## Red Color Filtering

To reduce computational overhead and improve efficiency, we include a **red color-based filter**:

- Before inference, each patch is evaluated for **red color dominance**, using adjustable parameters (`red_threshold`, `red_factor`).
- Patches that do not meet the red threshold—typically areas without visual resemblance to Waldo’s outfit—are skipped.

This filter leverages domain-specific knowledge to discard uninformative regions while preserving recall in likely areas.

---

## Non-Maximum Suppression (NMS)

Since overlapping windows are processed at multiple scales, multiple detections may cluster around the same object. To reduce redundancy:

- We apply **Non-Maximum Suppression (NMS)** with a configurable **IoU threshold** (e.g., `0.1`), keeping only the highest-scoring box within overlapping regions.

This step ensures that final detections are clean and localized, improving interpretability and post-processing.

---

## Summary of Components

| Component              | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| **Sliding Window**     | Systematically scan full-resolution images using fixed-size windows     |
| **Scale Pyramid**      | Adjust window size to detect Waldos appearing at different scales       |
| **Red Filter**         | Skip low-red-content patches to reduce inference time and false positives |
| **NMS**                | Eliminate overlapping detections, keeping only the strongest candidates  |

---

## Detection Examples

Below are several examples of Waldo detection using the full pipeline:

### ✅ Successful Detection – Top-Ranked Box

In the following example, Waldo is correctly detected in the **top-ranked** bounding box, with a confidence score above 0.95. 

![Detection Success 1](Data Examples/example_1.png)

---

### ✅ Successful Detection – Lower-Ranked Box

Here, Waldo is detected in the **second-ranked** prediction. 

![Detection Success 2](Data Examples/example_3.png)

---

### ❌ Missed Detection

Here, Waldo was detected in the **fifth-ranked** position, which resulted in a failure.

![Detection Failure 1](Data Examples/example_4.png)

# ✅ Validation Results

## 5.2. Summary of Sliding Window Validation

| Image ID  | Detection | Notes                                                       |
|-----------|-----------|-------------------------------------------------------------|
| image1    | ✅         | —                                                           |
| image2    | ✅         | —                                                           |
| image3    | ✅         | —                                                           |
| image4    | ✅         | —                                                           |
| image5    | ✅         | —                                                           |
| image6    | ✅         | —                                                           |
| image7    | ✅         | —                                                           |
| image8    | ✅         | —                                                           |
| image9    | ❌         | Waldo not detected                                          |
| image10   | ✅         | Red threshold adjusted manually                             |
| image11   | ✅         | —                                                           |
| image12   | ✅         | Red threshold adjusted manually                             |
| image13   | ✅         | —                                                           |
| image14   | ✅         | —                                                           |
| image15   | ❌         | Detected at the 15th position in ranked sliding windows     |

---

## 📈 Detection Performance Overview

- 🟢 **Successful detections**: **13 / 15 images**
- 🔴 **Missed detections**: `image9`, `image15`
- ⚙️ **Manual red filter adjustments applied**: `image10`, `image12`



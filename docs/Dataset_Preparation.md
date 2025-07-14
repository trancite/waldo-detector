# Preprocessing and Model Selection

## Dataset Preprocessing

### Dataset Structure

The dataset contains full images and labeled patches extracted from them. However, among the patch folders, some are duplicated versions where one is simply a grayscale (black and white) version of the other. To avoid redundancy, we will focus exclusively on the colored patches.

| Folder Name         | Description                          | Number of Images |
|---------------------|--------------------------------------|------------------|
| `full_images/`      | Original full-size images            | 19               |
| `patches_64x64/`    | Cropped 64×64 patches                | 5384             |
| `patches_128x128/`  | Cropped 128×128 patches              | *n*              |
| `patches_256x256/`  | Cropped 256×256 patches              | *m*              |

First of all, let's examine the 64×64 patches. The class distribution is the following:

- `notwaldo`: 5337  
- `waldo`: 39

This represents an extreme class imbalance, which would likely harm the model's performance if left unaddressed. Additionally, many of the existing **waldo** patches are not perfectly centered on Waldo. While this may help the network generalize better by learning to detect Waldo in off-center or challenging contexts, it also introduces noise.


**Full Image Example**  
![Full Image](Data_Examples/example_full_image.png)

**Patch Samples (64×64)**  
![Waldo Patch](Data_Examples/10_15_4.jpg)
![NotWaldo Patch](DataExamples/example_not_waldo.png)



To improve the dataset, we manually created a few additional 64×64 **waldo** patches by carefully cropping them from the original full images using the script `dataset_extension.ipynb`.

At the same time, **a large number of low-quality or ambiguous 64×64 patches were manually discarded**, especially from the `notwaldo` class, to avoid false positives and improve overall data quality.

After this manual curation, the class distribution becomes:

- `notwaldo`: 5337  
- `waldo`: 32

Although the imbalance remains significant, these new samples improve the dataset's quality and diversity.

---

### Data Augmentation for Class Balance

At this stage, our dataset of 64×64 patches remains imbalanced. To improve the model’s ability to learn from limited **waldo** samples, we implement **data augmentation** aggressively.

Data augmentation is a technique used to artificially increase the size of the dataset by generating new images through transformations of existing ones. These transformations help the model generalize better and reduce overfitting.

Since Waldo typically appears with consistent colors and orientations, we apply only **mild augmentations** to avoid distorting the visual patterns that identify him. These include:

- Horizontal flips  
- Rotations  
- Slight changes in brightness, contrast, or saturation  
- Minor scaling and translations

The augmentation is **applied to disk**, not dynamically in memory. This requires caution: the dataset must be cleaned before re-generating, to avoid duplicated samples.

The **augmentation ratio is asymmetric**:

- For the **waldo** class: ×50
- For the **notwaldo** class: ×3

This significantly increases the presence of **waldo** in the training data, helping the model to learn rare but crucial features.



---

## Model Architecture and Fine-Tuning

We use **MobileNetV2** with default pretrained weights. Fine-tuning is applied to the last two layers of the network:

- A **convolutional layer**, whose weights are updated during training.
- A **fully connected (linear) layer**, which is replaced to output two logits: **waldo** and **notwaldo**.

This allows us to leverage the pretrained feature extraction capabilities while adapting the output to our binary classification task.

We restrict training to these layers due to:

1. **Hardware constraints**
2. **Transfer learning benefits** from low- and mid-level pretrained filters

---

## Loss Function

To handle the extreme class imbalance, we use the **Focal Loss**, instead of the standard CrossEntropyLoss.

Focal Loss is designed to focus training on hard-to-classify examples. It is especially useful in cases of class imbalance, as it down-weights easy examples and gives more importance to misclassified ones — in our case, **waldo** patches.

This choice drastically improves training stability and performance when dealing with the rare class.

---

## Final Preprocessing Step

MobileNetV2 expects input images of size (3, 224, 224). Our patches are (3, 64, 64), so we must resize them accordingly.

We perform a **two-step transformation**:

1. Resize to 256×256
2. Center crop to 224×224

This prevents geometric distortions and preserves spatial relationships important for classification.

This preprocessing ensures compatibility with MobileNetV2 while preserving crucial visual features in the input.


# Patch Detector: Training and Evaluation

## 📁 Train-Test Split

To evaluate our classifier reliably, we divide the **augmented dataset** into two distinct subsets:

- **Training set**
- **Testing set**

This division is guided by the following principles:

### ❗ Key Requirements

1. **The test set must not contain any augmented images.**  
   Augmentation is used to improve model training and reduce overfitting, but these images are artificially generated and do not reflect real-world conditions. For an honest evaluation of generalization, only **original, unaugmented** images should be included in the test set.

2. **Class distribution must be preserved.**  
   Since we have only 32 original **waldo** patches, it’s crucial to maintain a similar class balance in both the training and test sets.  
   We achieve this using `StratifiedShuffleSplit` from `sklearn.model_selection`, which ensures that the ratio between **waldo** and **notwaldo** remains consistent in both subsets.

---

## 🧠 Model Training

We train the model for **20 epochs**, tracking the training loss after each step to monitor learning progression.


### Training Results

- ⏱️ **Total time:** 7204.46 seconds

---

## 🧪 Model Testing

The test set contains **1,074 images**, of which only **6** belong to the **waldo** class.

### 🔍 Classification Threshold

To make a binary decision from model probabilities, we apply a threshold p:

- If the predicted probability for **waldo** is greater than p = 0.5, classify the patch as **waldo**.
- Otherwise, classify it as **notwaldo**.

This lower threshold helps **counteract the severe class imbalance**, making the model more sensitive to **waldo** cases.

---

## 📊 Evaluation Metrics

Not all metrics are equally useful when working with highly imbalanced datasets. Here’s how we interpret them in this context:

### ✅ Accuracy
> Proportion of total correct predictions.

⚠️ **Not reliable alone** here — the model could get high accuracy simply by always predicting the majority class (**notwaldo**).

### ✅ Recall
> Ability to correctly identify actual instances of a class.

- **Class-specific recall** helps assess how well each class is detected.
- **Weighted recall** averages recalls across classes, weighted by class frequency — more meaningful in imbalanced datasets.

---

### Final Results

#### 🧪 Test Set Limitations

First of all, it is important to note the small size of the test set (around 6 images of the **Waldo** class). While this is clearly a limitation when assessing the model’s performance, we chose to keep it small for two main reasons:

1. The model will also be validated on full-size images using a sliding window approach, which provides a more realistic evaluation.
2. The number of available **Waldo** samples is very limited, and we want the model to be trained on as many of them as possible.

Despite its size, this minimal test set still serves as a basic sanity check for patch-level classification performance.

| Metric                 | Value       |
|------------------------|-------------|
| **Accuracy**           | 99.44%      |
| **Weighted Recall**    | 99.44%      |
| **Recall (waldo)**     | 83.33%      |
| **Recall (notwaldo)**  | 99.53%      |

> 🟢 These results demonstrate strong overall performance.  
> The **waldo** class recall, while slightly lower, still reaches **83.33%**, which is a clear success considering the original class imbalance and dataset limitations.

⚠️ **We are aware of the small size of the test set. However, we do not need specific metrics here, as the important behavior will be evaluated on the full images.**







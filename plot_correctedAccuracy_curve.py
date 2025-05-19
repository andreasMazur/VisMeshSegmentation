import numpy as np
import random

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deepview import DeepView
from relabel_helpers.deepview_label_corrections import DeepViewLabelRevisit
from relabel_helpers.recommendation_functions import recommend_KNN_based

def apply_label_noise(y, noise_level, num_classes, seed=None):
    """
    Applies label noise by randomly flipping a percentage of the labels.

    Parameters:
        y: np.array - labels
        noise_level: float - percentage of labels to corrupt (0.0 to 1.0)
        num_classes: int - number of unique classes
        seed: int or None - for reproducibility

    Returns:
        y_noisy: np.array - noisy labels
    """
    if seed is not None:
        np.random.seed(seed)

    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_level * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    for idx in noisy_indices:
        original_label = y[idx]
        new_label = random.choice([l for l in range(num_classes) if l != original_label])
        y_noisy[idx] = new_label

    return y_noisy



# Load and normalize data
data, target = load_digits(return_X_y=True)
data = data.reshape(len(data), -1) / 255.
classes = np.arange(10)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42, stratify=target)

# Introduce label noise
noise_percent = 50
noise_level = noise_percent / 100.0
y_train_noisy = apply_label_noise(y_train, noise_level, num_classes=len(classes), seed=42)

# Initial prediction on noisy labels
n_trees = 100
model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
model.fit(X_train, y_train_noisy)
y_pred = model.predict(X_train)

# Percentages of label corrections to evaluate
correction_percentages = np.arange(0, 110, 10)  # 0% to 100% inclusive
average_accuracies = []

pred_wrapper = DeepView.create_simple_wrapper(model.predict_proba)

 # --- Deep View Parameters ----
batch_size = 32
max_samples = 100000
data_shape = (64,)
resolution = 100
N = 10
lam = 1
cmap = 'tab10'
# to make shure deepview.show is blocking,
# disable interactive mode
interactive = False
title = 'Forest - MNIST'

deepview = DeepViewLabelRevisit(pred_wrapper, classes, max_samples, batch_size, data_shape,
	N, lam, resolution, cmap, interactive, title, disc_dist=False)

deepview.add_samples(X_train, y_train_noisy)

# Repeat for each correction percentage
for percent in correction_percentages:
    test_scores = []
    for _ in range(10):  # Repeat 10 times for average
        knn_corrected_labels, _, _ = recommend_KNN_based(
            X_train, y_train_noisy, deepview.background_at, n_neighbors=5, recommendation_percentage=percent)

        model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        model.fit(X_train, knn_corrected_labels)
        score = model.score(X_test, y_test)
        test_scores.append(score)

    avg_score = np.mean(test_scores)
    average_accuracies.append(avg_score)
    print(f"Correction {percent}%: Avg accuracy = {avg_score:.4f}")

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(correction_percentages, average_accuracies, marker='o')
plt.xlabel('Percentage of Noisy Labels Corrected')
plt.ylabel('Average Accuracy on Test Set')
plt.title('Effect of KNN-based Label Correction on Accuracy')
plt.grid(True)
plt.show()

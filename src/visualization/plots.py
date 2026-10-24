"""Small plotting helpers shared by the analysis notebooks."""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def normalized_confusion_matrix(y_true, y_pred, class_labels=None, ax=None):
    """Plot a row-normalized confusion matrix for the requested class order."""
    if class_labels is None:
        class_labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=class_labels, normalize="true")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_labels)), class_labels, rotation=90)
    ax.set_yticks(range(len(class_labels)), class_labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Row-normalized confusion matrix (diagonal = recall)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    threshold_value = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            valor = cm[i, j]
            if valor > 0.005:
                ax.text(
                    j,
                    i,
                    f"{valor:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if valor > threshold_value else "black",
                )
    return ax

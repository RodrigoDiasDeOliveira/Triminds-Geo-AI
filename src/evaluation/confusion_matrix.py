import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names
):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.show()
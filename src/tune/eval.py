import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_classification(
    y_true: list[str],
    y_preds: list[str],
    labels: list[str],
    save_folder: str | None = None,
) -> None:
    """
    Evaluate a classification task and plot the confusion matrix.

    Args:
        y_true (list): A list of true labels.
        y_preds (list): A list of predicted labels.
        labels (list): A list of unique labels.
        save_folder (str): The folder to save the confusion matrix plot and classification report.
    """
    report = classification_report(y_true, y_preds, labels=labels)
    cm = confusion_matrix(y_true, y_preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(
            f"{save_folder}/confusion_matrix_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        )
        with open(
            f"{save_folder}/classification_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt",
            "w",
        ) as f:
            f.write(report)

        return

    print(report)
    plt.show()

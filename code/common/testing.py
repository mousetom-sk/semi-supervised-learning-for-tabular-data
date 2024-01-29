from typing import List, Tuple
from numpy.typing import NDArray

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

import matplotlib.pyplot as plt


Estimator = MLPClassifier | RandomForestClassifier | LabelPropagation | LabelSpreading | SelfTrainingClassifier


class ModelTester:

    _X_test: NDArray = None
    _y_test: NDArray = None
    _display_labels: List[str] = None
    _titles: List[str] = None

    _labels_mapping: List[Tuple[str, str]] = None
    _mertics: List[str] = ["Precision", "Recall"]

    def __init__(
        self,
        X_test: NDArray, y_test: NDArray,
        display_labels: List[str], titles: List[str]
    ):
        self._X_test = X_test
        self._y_test = y_test
        self._display_labels = display_labels
        self._titles = titles

        self._labels_mapping = list(zip(range(len(display_labels)), display_labels))

    def test(self, models: List[Estimator]) -> None:
        figs = [plt.figure(m) for m in self._mertics]
        axes = list()
        
        for i in range(len(self._mertics)):
            axes.append(figs[i].subplots(ncols=len(models)))

            for j in range(len(models)):
                axes[-1][j].set_title(self._titles[j], fontdict={"fontsize": 12}, pad=30)

        for i, m in enumerate(models):
            y_pred = m.predict(self._X_test)

            report = classification_report(self._y_test, y_pred, digits=4)
            conf_mat = confusion_matrix(self._y_test, y_pred)

            print(f"Model {i+1}")
            print()
            print(f"Labels Mapping: {self._labels_mapping}")
            print()
            print(report)
            print(conf_mat)
            print()
            
            conf_prec = ConfusionMatrixDisplay.from_predictions(
                self._y_test, y_pred,
                normalize="pred",
                display_labels=self._display_labels,
                ax=axes[0][i]
            )
            conf_prec.ax_.set_xticklabels(self._display_labels, rotation=45, ha="right")

            conf_recall = ConfusionMatrixDisplay.from_predictions(
                self._y_test, y_pred,
                normalize="true",
                display_labels=self._display_labels,
                ax=axes[1][i]
            )
            conf_recall.ax_.set_xticklabels(self._display_labels, rotation=45, ha="right")

        plt.show()

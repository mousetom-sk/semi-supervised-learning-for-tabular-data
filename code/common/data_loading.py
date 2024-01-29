from typing import Tuple, List
from numpy.typing import NDArray

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from . import config


class CoverDataLoader:

    display_labels: List[str] = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas Fir", "Krummholz"]
    cases: List[str] = ["All Labeled", "10% Labeled"]
    non_categoric_features: slice = slice(10)

    _balanced: bool = None

    def __init__(self, balanced: bool=False) -> None:
        self._balanced = balanced

    def load_subsets(self) -> Tuple[List[NDArray], List[NDArray], NDArray, NDArray, NDArray, NDArray]:
        Xs, ys, X_val, y_val, X_test, y_test = self.load_unlabaled()

        Xs_new, ys_new = [], []
        for X, y in zip(Xs, ys):
            unlab = y == -1
            Xs_new.append(X[~unlab,:])
            ys_new.append(y[~unlab])

        return Xs_new, ys_new, X_val, y_val, X_test, y_test
    
    def load_unlabaled(self) -> Tuple[List[NDArray], List[NDArray], NDArray, NDArray, NDArray, NDArray]:
        common_dir = os.path.dirname(__file__)
        code_dir = os.path.dirname(common_dir)
        project_dir = os.path.dirname(code_dir)
        data_dir = os.path.join(project_dir, "data")

        if self._balanced:
            df_train = pd.read_csv(os.path.join(data_dir, "covtype-train-balanced.csv"))
        else:
            df_train = pd.read_csv(os.path.join(data_dir, "covtype-train.csv"))
        
        data = df_train.values

        X_train = data[:, 1:-1]

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(data[:, -1])

        df_val = pd.read_csv(os.path.join(data_dir, "covtype-val.csv"))
        data = df_val.values

        X_val = data[:, 1:-1]
        y_val = encoder.transform(data[:, -1])

        df_test = pd.read_csv(os.path.join(data_dir, "covtype-test.csv"))
        data = df_test.values

        X_test = data[:, 1:-1]
        y_test = encoder.transform(data[:, -1])

        Xs = [X_train]
        ys = [y_train]

        for keep in [0.1]:
            df_subset = df_train.groupby(
                "Cover_Type", group_keys=False
            ).apply(lambda x:
                x.sample(frac=keep, random_state=config.RANDOM_STATE)
            )

            mask = np.array([df_train["id"].values[i] not in df_subset["id"].values
                             for i in range(len(df_train))])
            y_masked = np.copy(y_train)
            y_masked[mask] = -1

            Xs.append(X_train)
            ys.append(y_masked)

        return Xs, ys, X_val, y_val, X_test, y_test

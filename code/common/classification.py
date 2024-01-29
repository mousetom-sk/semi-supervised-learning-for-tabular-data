from __future__ import annotations
from typing import Dict, List, Tuple, Any
from numpy.typing import NDArray

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.base import clone


PseudoLabeler = LabelPropagation | LabelSpreading | Pipeline
Estimator = MLPClassifier | RandomForestClassifier | Pipeline


class KNNDistanceKernel:

    _nn: NearestNeighbors = None
    _gamma: float = None
    _last_fitted_X: NDArray = None

    def __init__(self, n_neighbors: int, gamma: float):
        self._nn = NearestNeighbors(n_neighbors=n_neighbors)
        self._gamma = gamma

    def _should_fit_first(self, X: NDArray) -> bool:
        return (self._last_fitted_X is None 
                or self._last_fitted_X.shape != X.shape
                or np.any(~np.equal(self._last_fitted_X, X)))

    def __call__(self, X: NDArray, Y: NDArray) -> NDArray:
        if self._should_fit_first(X): 
            self._nn.fit(X)

        dist, ind = self._nn.kneighbors(Y)
        weights = np.zeros((Y.shape[0], X.shape[0]))
        rows = np.arange(Y.shape[0])
        weights[rows[:, np.newaxis], ind] = np.exp(-self._gamma * (dist**2))

        return weights.T


class PseudoLabelingEnhancedClassifier:
    
    _pseudo_labeler: PseudoLabeler = None
    _base_estimator: Estimator = None
    _min_confidence: float = None

    def __init__(
        self,
        pseudo_labeler: PseudoLabeler, base_estimator: Estimator, min_confidence: float
    ):
        self._pseudo_labeler = pseudo_labeler
        self._base_estimator = base_estimator
        self._min_confidence = min_confidence

    def fit(self, X: NDArray, y: NDArray) -> Estimator:
        base_estimator = clone(self._base_estimator)
        has_label = y != -1

        if np.all(has_label):
            base_estimator.fit(X, y)

            return base_estimator
        
        pseudo_labeler = clone(self._pseudo_labeler)
        pseudo_labeler.fit(X, y)

        y_prob = pseudo_labeler.predict_proba(X)
        y_pred = np.argmax(y_prob, axis=1)
        confident = np.max(y_prob, axis=1) > self._min_confidence
        add_label = ~has_label & confident

        print(f"Adding {np.sum(add_label)} unlabeled example(s).")

        y_ext = np.copy(y)
        y_ext[add_label] = y_pred[add_label]
        has_label_ext = y_ext != -1

        base_estimator.fit(X[has_label_ext], y_ext[has_label_ext])

        return base_estimator


class ContinuousSelfTrainingClassifier:

    _base_estimator: Estimator = None
    _max_iter: int = None
    _min_confidence: float = None

    def __init__(self, base_estimator: Estimator, max_iter: int, min_confidence: float):
        self._base_estimator = base_estimator
        self._max_iter = max_iter
        self._min_confidence = min_confidence

    def fit(self, X: NDArray, y: NDArray) -> Estimator:
        base_estimator = clone(self._base_estimator)

        if isinstance(base_estimator, Pipeline):
            params_prefix = "estimator__"
        else:
            params_prefix = ""

        base_estimator.set_params(**{params_prefix + "warm_start": True})

        has_label = y != -1
        X_lab = X[has_label]
        y_lab = y[has_label]

        base_estimator.fit(X_lab, y_lab)

        if np.all(has_label):
            return base_estimator
        
        if isinstance(base_estimator, Pipeline):
            true_estimator = base_estimator.named_steps["estimator"]
        else:
            true_estimator = base_estimator

        if isinstance(true_estimator, MLPClassifier):
            current_rate = base_estimator.get_params()[params_prefix + "learning_rate_init"]
            base_estimator.set_params(**{params_prefix + "learning_rate_init": current_rate / 2})
        elif isinstance(true_estimator, RandomForestClassifier):
            sample_weights = np.ones_like(y_lab)

        X_unlab = X[~has_label]

        it = 0
        added = 0

        while not np.all(has_label) and it < self._max_iter:
            y_prob = base_estimator.predict_proba(X_unlab)
            y_pred = np.argmax(y_prob, axis=1)
            confident = np.max(y_prob, axis=1) > self._min_confidence

            if np.all(~confident):
                break
            
            X_lab = np.concatenate((X_lab, X_unlab[confident]))
            y_lab = np.concatenate((y_lab, y_pred[confident]))
            X_unlab = X_unlab[~confident]

            if isinstance(true_estimator, MLPClassifier):
                base_estimator.fit(X_lab, y_lab)
            else:
                sample_weights = np.append(sample_weights, np.repeat(0.5, np.sum(confident)))
                current_estimators = base_estimator.get_params()[params_prefix + "n_estimators"]
                base_estimator.set_params(**{params_prefix + "n_estimators": current_estimators + 20})
                base_estimator.fit(X_lab, y_lab, sample_weights)

            it += 1
            added += np.sum(confident)

        print(f"Added {added} unlabeled examples in total.")
        
        return base_estimator


class CoTrainingClassifier:

    _base_estimator1: Estimator = None
    _base_estimator2: Estimator = None
    _max_iter: int = None
    _min_confidence: float = None

    def __init__(self, base_estimator1: Estimator, base_estimator2: Estimator, max_iter: int, min_confidence: float):
        self._base_estimator1 = clone(base_estimator1)
        self._base_estimator2 = clone(base_estimator2)
        self._max_iter = max_iter
        self._min_confidence = min_confidence

    def fit(self, X: NDArray, y: NDArray) -> CoTrainingClassifier:
        has_label1, has_label2 = y != -1, y != -1
        X_lab1, X_lab2 = X[has_label1], X[has_label1]
        y_lab1, y_lab2 = y[has_label2], y[has_label2]

        self._base_estimator1.fit(X_lab1, y_lab1)
        self._base_estimator2.fit(X_lab2, y_lab2)

        if np.all(has_label1):
            return self

        X_unlab1, X_unlab2 = X[~has_label1], X[~has_label2]

        it = 0

        while len(X_unlab1) + len(X_unlab2) > 0 and it < self._max_iter:
            y_prob1 = self._base_estimator1.predict_proba(X_unlab2)
            y_pred1 = np.argmax(y_prob1, axis=1)
            confident1 = np.max(y_prob1, axis=1) > self._min_confidence

            y_prob2 = self._base_estimator2.predict_proba(X_unlab1)
            y_pred2 = np.argmax(y_prob2, axis=1)
            confident2 = np.max(y_prob2, axis=1) > self._min_confidence

            if np.all(~confident1) and np.all(~confident2):
                break
            
            X_lab1 = np.concatenate((X_lab1, X_unlab1[confident2]))
            y_lab1 = np.concatenate((y_lab1, y_pred2[confident2]))
            X_unlab1 = X_unlab1[~confident2]
            
            X_lab2 = np.concatenate((X_lab2, X_unlab2[confident1]))
            y_lab2 = np.concatenate((y_lab2, y_pred1[confident1]))
            X_unlab2 = X_unlab2[~confident1]

            self._base_estimator1.fit(X_lab1, y_lab1)
            self._base_estimator2.fit(X_lab2, y_lab2)

            it += 1
        
        print(f"Left {len(X_unlab1)} unlabeled examples not used by estimator 1.")
        print(f"left {len(X_unlab2)} unlabeled examples not used by estimator 2.")
        
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        y_prob = self.predict_proba(X)
        y_pred = np.argmax(y_prob, axis=1)

        return y_pred

    def predict_proba(self, X: NDArray) -> NDArray:
        y_prob1 = self._base_estimator1.predict_proba(X)
        y_prob2 = self._base_estimator2.predict_proba(X)

        return (y_prob1 + y_prob2) / 2

    def score(self, X: NDArray, y: NDArray) -> float:
        y_pred = self.predict(X)

        return accuracy_score(y, y_pred)


BasicEstimator = MLPClassifier | RandomForestClassifier | LabelPropagation |\
                 LabelSpreading | SelfTrainingClassifier


def make_pipeline(
    prerpocessing_steps: List[Tuple[str, Any]],
    estimator_type: BasicEstimator, **estimator_args: Dict[str, Any]
) -> Pipeline:
    estimator = estimator_type(**estimator_args)
    pipeline = Pipeline(
        prerpocessing_steps + [("estimator", estimator)]
    )

    return pipeline

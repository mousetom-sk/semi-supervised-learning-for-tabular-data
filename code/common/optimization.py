from typing import Type, Dict, List, Any
from numpy.typing import NDArray

from abc import ABC, abstractmethod

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.model_selection import ParameterSampler, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from .classification import PseudoLabelingEnhancedClassifier, ContinuousSelfTrainingClassifier, CoTrainingClassifier
from .config import RANDOM_STATE


Estimator = MLPClassifier | RandomForestClassifier | LabelPropagation |\
            LabelSpreading | SelfTrainingClassifier | PseudoLabelingEnhancedClassifier |\
            ContinuousSelfTrainingClassifier | CoTrainingClassifier |\
            Pipeline


class HyperparameterSearch(ABC):

    _estimator_type: Type[Estimator] = None
    _static_params: Dict[str, Any] = None
    _param_distributions: Dict[str, Any] = None
    _n_samples: int = None
    _random_state: int = None

    def __init__(
        self,
        estimator_type: Type[Estimator],
        static_params: Dict[str, Any],
        param_distributions: Dict[str, Any],
        n_samples: int,
        random_state: int = RANDOM_STATE
    ):
        self._estimator_type = estimator_type
        self._static_params = static_params
        self._param_distributions = param_distributions
        self._n_samples = n_samples
        self._random_state = random_state

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X: NDArray, y: NDArray, X_val: NDArray, y_val: NDArray) -> Estimator:
        sampler = self._get_param_sampler()

        best_score = None
        best_model = None

        for params in sampler:
            all_params = self._static_params | params
            if self._estimator_type in {MLPClassifier, RandomForestClassifier}:
                all_params["random_state"] = self._random_state

            model = self._estimator_type(**all_params)
            model = model.fit(X, y)

            score = model.score(X_val, y_val)
            print(f"Configuration: {params}, Score: {score}")

            if best_score is None or score > best_score:
                best_score = score
                best_model = model

        print(f"Best Score: {best_score}")

        return best_model
    
    @abstractmethod
    def _get_param_sampler(self) -> ParameterSampler | ParameterGrid:
        pass


class RandomSearch(HyperparameterSearch):

    def __init__(
        self,
        estimator_type: Type[Estimator],
        static_params: Dict[str, Any],
        param_distributions: Dict[str, Any],
        n_samples: int,
        random_state: int = RANDOM_STATE
    ):
        super().__init__(estimator_type, static_params, param_distributions, n_samples, random_state)
    
    def _get_param_sampler(self) -> ParameterSampler | ParameterGrid:
        return ParameterSampler(
            self._param_distributions,
            n_iter=self._n_samples,
            random_state=self._random_state
        )


class GridSearch(HyperparameterSearch):

    def __init__(
        self,
        estimator_type: Type[Estimator],
        static_params: Dict[str, Any],
        param_distributions: Dict[str, Any],
        random_state: int = RANDOM_STATE
    ):
        super().__init__(estimator_type, static_params, param_distributions, 0, random_state)
    
    def _get_param_sampler(self) -> ParameterSampler | ParameterGrid:
        return ParameterGrid(self._param_distributions)


class BasicEstimatorWrapper:

    _estimator_type: Type[Estimator] = None
    _static_params: Dict[str, Any] = None
    _random_state: int = None

    def __init__(
        self,
        estimator_type: Type[Estimator],
        static_params: Dict[str, Any] = {},
        random_state: int = RANDOM_STATE
    ):
        self._estimator_type = estimator_type
        self._static_params = static_params
        self._random_state = random_state

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X: NDArray, y: NDArray, X_val: NDArray, y_val: NDArray) -> Estimator:
        all_params = dict(**self._static_params)
        if self._estimator_type in {MLPClassifier, RandomForestClassifier}:
            all_params["random_state"] = self._random_state
        
        model = self._estimator_type(**all_params)
        model = model.fit(X, y)

        score = model.score(X_val, y_val)
        print(f"Configuration: {self._static_params}, Score: {score}")

        return model


class BatchOptimizer:

    _estimator: BasicEstimatorWrapper | HyperparameterSearch = None
    _cases: List[str] = None

    def __init__(
        self,
        estimator: BasicEstimatorWrapper | HyperparameterSearch,
        cases: List[str]
    ):
        self._estimator = estimator
        self._cases = cases
    
    def fit(
        self,
        Xs: List[NDArray], ys: List[NDArray],
        X_val: NDArray, y_val: NDArray
    ) -> List[Estimator]:
        models = []

        for i, (X, y) in enumerate(zip(Xs, ys)):
            print(f"Training set {i + 1}: {self._cases[i]}")
            models.append(self._estimator.fit(X, y, X_val, y_val))
            print()

        return models

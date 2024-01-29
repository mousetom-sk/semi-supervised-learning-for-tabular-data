from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import StandardScaler, FunctionTransformer, add_dummy_feature
from sklearn.compose import ColumnTransformer

from common.data_loading import CoverDataLoader
from common.classification import PseudoLabelingEnhancedClassifier, KNNDistanceKernel, make_pipeline
from common.optimization import GridSearch, BatchOptimizer
from common.testing import ModelTester
from common.config import RANDOM_STATE


# Load Data

data_loader = CoverDataLoader() # for balanced training set: CoverDataLoader(True)
Xs, ys, X_val, y_val, X_test, y_test = data_loader.load_unlabaled()

# Define Model Hyper-Parameters

scaler1 = ColumnTransformer(
    [("scaler", StandardScaler(), data_loader.non_categoric_features)],
    remainder="passthrough"
)
scaler2 = ColumnTransformer(
    [("scaler", StandardScaler(), data_loader.non_categoric_features)],
    remainder="passthrough"
)
intercept = FunctionTransformer(add_dummy_feature)

static_params = {
    "pseudo_labeler": make_pipeline(
        prerpocessing_steps=[("scaler", scaler1)],
        estimator_type=lambda n_neighbors, gamma: LabelPropagation(
            kernel=KNNDistanceKernel(n_neighbors, gamma), max_iter=200
        ),
        n_neighbors=13, # for balanced training set: 11
        gamma=0.5 # for balanced training set: 1
    ),
    "base_estimator": make_pipeline(
        prerpocessing_steps=[("scaler", scaler2), ("intercept", intercept)],
        estimator_type=MLPClassifier,
        hidden_layer_sizes=(64,), # for balanced training set: (64,)
        activation="tanh",
        learning_rate_init=0.001,
        learning_rate="adaptive",
        random_state=RANDOM_STATE
    )
}

distributions = {
    "min_confidence": [0.85, 0.9, 0.925, 0.95]
}

# Train (Optimize)

est = GridSearch(PseudoLabelingEnhancedClassifier, static_params, distributions)
opt = BatchOptimizer(est, data_loader.cases)
models = opt.fit(Xs, ys, X_val, y_val)

# Test

tester = ModelTester(X_test, y_test, data_loader.display_labels, data_loader.cases)
tester.test(models)

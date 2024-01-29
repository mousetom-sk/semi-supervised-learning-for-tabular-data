from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer, add_dummy_feature
from sklearn.compose import ColumnTransformer

from common.data_loading import CoverDataLoader
from common.classification import make_pipeline
from common.optimization import RandomSearch, BatchOptimizer
from common.testing import ModelTester
from common.config import RANDOM_STATE


# Load Data

data_loader = CoverDataLoader() # for balanced training set: CoverDataLoader(True)
Xs, ys, X_val, y_val, X_test, y_test = data_loader.load_subsets()

# Define Model Hyper-Parameters

scaler = ColumnTransformer(
    [("scaler", StandardScaler(), data_loader.non_categoric_features)],
    remainder="passthrough"
)
intercept = FunctionTransformer(add_dummy_feature)

static_params = {
    "prerpocessing_steps": [("scaler", scaler), ("intercept", intercept)],
    "estimator_type": MLPClassifier,
    "learning_rate": "adaptive",
    "random_state": RANDOM_STATE
}

distributions = {
    "hidden_layer_sizes": [(16,), (32,), (64,)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.01, 0.001]
}

# Train (Optimize)

est = RandomSearch(make_pipeline, static_params, distributions, 8)
opt = BatchOptimizer(est, data_loader.cases)
models = opt.fit(Xs, ys, X_val, y_val)

# Test

tester = ModelTester(X_test, y_test, data_loader.display_labels, data_loader.cases)
tester.test(models)

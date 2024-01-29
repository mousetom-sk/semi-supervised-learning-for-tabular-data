from sklearn.ensemble import RandomForestClassifier

from common.data_loading import CoverDataLoader
from common.classification import ContinuousSelfTrainingClassifier
from common.optimization import GridSearch, BatchOptimizer
from common.testing import ModelTester
from common.config import RANDOM_STATE


# Load Data

data_loader = CoverDataLoader() # for balanced training set: CoverDataLoader(True)
Xs, ys, X_val, y_val, X_test, y_test = data_loader.load_unlabaled()

# Define Model Hyper-Parameters

static_params = {
    "base_estimator": RandomForestClassifier(
        class_weight="balanced", # for balanced training set: None
        random_state=RANDOM_STATE
    ),
    "max_iter": 10
}

distributions = {
    "min_confidence": [0.85, 0.9, 0.925, 0.95]
}

# Train (Optimize)

est = GridSearch(ContinuousSelfTrainingClassifier, static_params, distributions)
opt = BatchOptimizer(est, data_loader.cases)
models = opt.fit(Xs, ys, X_val, y_val)

# Test

tester = ModelTester(X_test, y_test, data_loader.display_labels, data_loader.cases)
tester.test(models)

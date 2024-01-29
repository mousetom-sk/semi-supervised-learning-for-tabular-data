from sklearn.ensemble import RandomForestClassifier

from common.data_loading import CoverDataLoader
from common.optimization import GridSearch, BatchOptimizer
from common.testing import ModelTester


# Load Data

data_loader = CoverDataLoader()
Xs, ys, X_val, y_val, X_test, y_test = data_loader.load_subsets()

# Define Model Hyper-Parameters

static_params = {}

distributions = {
    "class_weight": [None, "balanced"]
}

# Train (Optimize)

est = GridSearch(RandomForestClassifier, static_params, distributions)
opt = BatchOptimizer(est, data_loader.cases)
models = opt.fit(Xs, ys, X_val, y_val)

# Test

tester = ModelTester(X_test, y_test, data_loader.display_labels, data_loader.cases)
tester.test(models)

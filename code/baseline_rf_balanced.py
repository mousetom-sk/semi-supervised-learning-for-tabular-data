import numpy as np

from sklearn.ensemble import RandomForestClassifier

from common.data_loading import CoverDataLoader
from common.optimization import GridSearch, BatchOptimizer
from common.testing import ModelTester


# Load Data

data_loader = CoverDataLoader(True)
Xs, ys, X_val, y_val, X_test, y_test = data_loader.load_subsets()

# Define Model Hyper-Parameters

bins = np.unique(y_val)
bins = np.append(bins, bins[-1] + 1)
hist = np.histogram(y_val, bins=bins)
class_weight = hist[0] / np.min(hist[0])

static_params = {}

distributions = {
    "class_weight": [None, dict(zip(hist[1], class_weight))]
}

# Train (Optimize)

est = GridSearch(RandomForestClassifier, static_params, distributions)
opt = BatchOptimizer(est, data_loader.cases)
models = opt.fit(Xs, ys, X_val, y_val)

# Test

tester = ModelTester(X_test, y_test, data_loader.display_labels, data_loader.cases)
tester.test(models)

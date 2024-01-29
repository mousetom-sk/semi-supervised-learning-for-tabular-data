import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

from common.data_loading import CoverDataLoader
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
    "max_iter": 25
}

distributions = {
    "threshold": [0.9, 0.95, 0.98, 0.99]
}

# Train (Optimize)

est = GridSearch(SelfTrainingClassifier, static_params, distributions)
opt = BatchOptimizer(est, data_loader.cases)
models = opt.fit(Xs, ys, X_val, y_val)

# Test

tester = ModelTester(X_test, y_test, data_loader.display_labels, data_loader.cases)
tester.test(models)

# Calculate Statistics for Trained Labels

for i, m in enumerate(models, start=1):
    trained_labels_hist = np.histogram(
        ys[0][m.labeled_iter_ > 0],
        bins=np.arange(len(data_loader.display_labels))
    )

    print(f"Model {i}")
    print()
    print(f"Labeled examples: {trained_labels_hist}")
    print()

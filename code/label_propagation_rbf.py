from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from common.data_loading import CoverDataLoader
from common.classification import make_pipeline
from common.optimization import GridSearch, BatchOptimizer
from common.testing import ModelTester


# Load Data

data_loader = CoverDataLoader() # for balanced training set: CoverDataLoader(True)
Xs, ys, X_val, y_val, X_test, y_test = data_loader.load_unlabaled()

# Define Model Hyper-Parameters

scaler = ColumnTransformer(
    [("scaler", StandardScaler(), data_loader.non_categoric_features)],
    remainder="passthrough"
)

static_params = {
    "prerpocessing_steps": [("scaler", scaler)],
    "estimator_type": LabelPropagation,
    "max_iter": 200
}

distributions = {
    "gamma": [0.1, 1, 1.5, 2, 5]
}

# Train (Optimize)

est = GridSearch(make_pipeline, static_params, distributions)
opt = BatchOptimizer(est, data_loader.cases)
models = opt.fit(Xs, ys, X_val, y_val)

# Test

tester = ModelTester(X_test, y_test, data_loader.display_labels, data_loader.cases)
tester.test(models)

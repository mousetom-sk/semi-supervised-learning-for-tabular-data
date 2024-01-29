from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from common.data_loading import CoverDataLoader
from common.classification import KNNDistanceKernel, make_pipeline
from common.optimization import RandomSearch, BatchOptimizer
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
    "estimator_type": lambda n_neighbors, gamma, alpha: LabelSpreading(
        kernel=KNNDistanceKernel(n_neighbors, gamma), alpha=alpha
    )
}

distributions = {
    "gamma": [0.1, 0.5, 1, 1.5, 2, 5],
    "n_neighbors": [11, 13, 15, 17],
    "alpha": [0.1, 0.2, 0.3, 0.4]
}

# Train (Optimize)

est = RandomSearch(make_pipeline, static_params, distributions, 10)
opt = BatchOptimizer(est, data_loader.cases)
models = opt.fit(Xs, ys, X_val, y_val)

# Test

tester = ModelTester(X_test, y_test, data_loader.display_labels, data_loader.cases)
tester.test(models)

import enum
from typing import NamedTuple, List, Any, Dict, Optional, Union, Callable

import numpy as np
import pytest
from sklearn import cluster
from sklearn import compose
from sklearn import datasets
from sklearn import decomposition
from sklearn import ensemble
from sklearn import feature_extraction
from sklearn import feature_selection
from sklearn import impute
from sklearn import linear_model
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm

from alumni import estimators


class EstimatorKind(enum.Enum):
    transform = 1
    predict = 2
    predict_proba = 3


class EstimatorSample(NamedTuple):
    estimator_class: type
    estimator_init_kwargs: Union[Dict[str, Any], Callable]
    estimator_kind: Optional[EstimatorKind]
    fit_param_names: List[str]
    X: Any
    y: Any

    @property
    def name(self):
        return self.estimator_class.__name__

    @property
    def kind(self):
        if self.estimator_kind is None:
            return None
        return self.estimator_kind.name


ESTIMATORS = [
    EstimatorSample(
        preprocessing.OneHotEncoder,
        dict(handle_unknown="ignore"),
        EstimatorKind.transform,
        ["categories_", "drop_idx_"],
        [["Male", 1], ["Female", 3], ["Female", 2]],
        None,
    ),
    EstimatorSample(
        preprocessing.PolynomialFeatures,
        dict(degree=2),
        EstimatorKind.transform,
        ["n_input_features_", "n_output_features_"],
        np.arange(6).reshape(3, 2),
        None,
    ),
    EstimatorSample(
        preprocessing.StandardScaler,
        dict(),
        EstimatorKind.transform,
        ["scale_", "mean_"]  # used in fit
        + ["var_", "n_samples_seen_"],  # used in partial_fit
        [[0, 0], [0, 0], [1, 1], [1, 1]],
        None,
    ),
    EstimatorSample(
        svm.LinearSVC,
        dict(random_state=0, tol=1e-5),
        EstimatorKind.predict,
        ["coef_", "intercept_"] + ["classes_"],
        *datasets.make_classification(n_features=4, random_state=0),
    ),
    EstimatorSample(
        neighbors.KNeighborsClassifier,
        dict(n_neighbors=3),
        EstimatorKind.predict_proba,
        ["classes_", "effective_metric_", "outputs_2d_", "_y"],
        [[0], [1], [2], [3]],
        [0, 0, 1, 1],
    ),
    EstimatorSample(
        decomposition.PCA,
        dict(n_components=2),
        EstimatorKind.transform,
        ["components_", "mean_"],
        np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]),
        None,
    ),
    EstimatorSample(
        preprocessing.RobustScaler,
        dict(),
        EstimatorKind.transform,
        ["center_", "scale_"],
        [[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]],
        None,
    ),
    EstimatorSample(
        svm.NuSVR,
        dict(gamma="scale", C=1.0, nu=0.1),
        EstimatorKind.predict,
        ["_sparse", "_gamma"],
        np.arange(50).reshape((10, 5)),
        np.arange(10),
    ),
    EstimatorSample(
        neighbors.RadiusNeighborsRegressor,
        dict(radius=1.0),
        EstimatorKind.predict,
        ["_y"],
        [[0], [1], [2], [3]],
        [0, 0, 1, 1],
    ),
    EstimatorSample(
        linear_model.ElasticNetCV,
        dict(cv=5, random_state=0),
        EstimatorKind.predict,
        [
            "mse_path_",
            "l1_ratio_",
            "alphas_",
            "coef_",
            "intercept_",
            "dual_gap_",
            "n_iter_",
        ],
        *datasets.make_regression(n_features=2, random_state=0),
    ),
    EstimatorSample(
        impute.SimpleImputer,
        dict(missing_values=np.nan, strategy="mean"),
        EstimatorKind.transform,
        ["statistics_", "indicator_"],
        [[7, 2, 3], [4, np.nan, 6], [10, 5, 9]],
        None,
    ),
    EstimatorSample(
        feature_selection.SelectKBest,
        dict(score_func=feature_selection.chi2, k=20),
        EstimatorKind.transform,
        ["scores_", "pvalues_"],
        *datasets.load_digits(return_X_y=True),
    ),
    EstimatorSample(
        feature_extraction.text.HashingVectorizer,
        dict(n_features=2 ** 4),
        EstimatorKind.transform,
        [],  # no params are fitted
        [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ],
        None,
    ),
    EstimatorSample(
        cluster.SpectralBiclustering,
        dict(n_clusters=2, random_state=0),
        None,
        ["row_labels_", "column_labels_", "rows_", "columns_"],
        np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]]),
        None,
    ),
    EstimatorSample(
        pipeline.Pipeline,
        lambda: dict(
            steps=[
                ("onehotencoder", preprocessing.OneHotEncoder(handle_unknown="ignore"))
            ]
        ),
        EstimatorKind.transform,
        [],  # all fitted params are deep
        [["Male", 1], ["Female", 3], ["Female", 2]],
        None,
    ),
    EstimatorSample(
        pipeline.Pipeline,
        lambda: dict(
            steps=[
                (
                    "pipeline",
                    pipeline.make_pipeline(
                        pipeline.make_pipeline(
                            preprocessing.OneHotEncoder(handle_unknown="ignore")
                        )
                    ),
                )
            ]
        ),
        EstimatorKind.transform,
        [],  # all fitted params are deep
        [["Male", 1], ["Female", 3], ["Female", 2]],
        None,
    ),
    EstimatorSample(
        compose.ColumnTransformer,
        lambda: dict(
            transformers=[
                ("norm1", preprocessing.Normalizer(norm="l1"), [0, 1]),
                ("norm2", preprocessing.Normalizer(norm="l1"), slice(2, 4)),
            ]
        ),
        EstimatorKind.transform,
        ["sparse_output_", "transformers_", "_columns", "_remainder", "_n_features"],
        np.array([[0.0, 1.0, 2.0, 2.0], [1.0, 1.0, 0.0, 1.0]]),
        None,
    ),
    EstimatorSample(
        ensemble.AdaBoostRegressor,
        dict(random_state=0, n_estimators=100),
        EstimatorKind.predict,
        ["estimators_", "estimator_weights_", "estimator_errors_"],
        *datasets.make_regression(
            n_features=4, n_informative=2, random_state=0, shuffle=False
        ),
    ),
    EstimatorSample(
        feature_selection.RFE,
        lambda: dict(
            estimator=svm.SVR(kernel="linear"), n_features_to_select=5, step=1
        ),
        EstimatorKind.predict,  # defined by the svm.SVR estimator
        ["n_features_", "support_", "ranking_", "estimator_"],
        *datasets.make_friedman1(n_samples=50, n_features=10, random_state=0),
    ),
]


def get_estimator(estimator_sample):
    if callable(estimator_sample.estimator_init_kwargs):
        init_kwargs = estimator_sample.estimator_init_kwargs()
    else:
        init_kwargs = estimator_sample.estimator_init_kwargs
    estimator = estimator_sample.estimator_class(**init_kwargs)
    estimator.fit(estimator_sample.X, estimator_sample.y)
    return estimator


@pytest.mark.parametrize("estimator_sample", ESTIMATORS)
def test_fit_param_names(estimator_sample):
    estimator = get_estimator(estimator_sample)
    # check type
    if estimator_sample.estimator_kind is not None:
        estimator_func = getattr(estimator, estimator_sample.estimator_kind.name)
        estimator_func(estimator_sample.X)
    # check fit_param_names
    fit_param_names = set(estimator_sample.fit_param_names)
    estimator_fit_attr_names = set(estimators.get_fit_params_dict(estimator))
    if not fit_param_names.issubset(estimator_fit_attr_names):
        # if it's not a subset, this assert is False, it's here to allow easier readability in pytest
        assert fit_param_names == estimator_fit_attr_names

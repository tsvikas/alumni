import enum
from typing import NamedTuple, List, Any, Dict, Optional

import numpy as np
import pytest
from sklearn.cluster import SpectralBiclustering
from sklearn.datasets import load_digits
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVR

from alumni import estimators


class EstimatorKind(enum.Enum):
    transform = 1
    predict = 2
    predict_proba = 3


class EstimatorSample(NamedTuple):
    estimator_class: type
    estimator_init_kwargs: Dict[str, Any]
    estimator_kind: Optional[EstimatorKind]
    fit_param_names: List[str]
    X: Any
    y: Any

    @property
    def name(self):
        return self.estimator_class.__name__


ESTIMATORS = [
    EstimatorSample(
        OneHotEncoder,
        dict(handle_unknown="ignore"),
        EstimatorKind.transform,
        ["categories_", "drop_idx_"],
        [["Male", 1], ["Female", 3], ["Female", 2]],
        None,
    ),
    EstimatorSample(
        PolynomialFeatures,
        dict(degree=2),
        EstimatorKind.transform,
        ["n_input_features_", "n_output_features_"],
        np.arange(6).reshape(3, 2),
        None,
    ),
    EstimatorSample(
        StandardScaler,
        dict(),
        EstimatorKind.transform,
        ["scale_", "mean_"]  # used in fit
        + ["var_", "n_samples_seen_"],  # used in partial_fit
        [[0, 0], [0, 0], [1, 1], [1, 1]],
        None,
    ),
    EstimatorSample(
        LinearSVC,
        dict(random_state=0, tol=1e-5),
        EstimatorKind.predict,
        ["coef_", "intercept_"] + ["classes_"],
        *make_classification(n_features=4, random_state=0),
    ),
    EstimatorSample(
        KNeighborsClassifier,
        dict(n_neighbors=3),
        EstimatorKind.predict_proba,
        ["classes_", "effective_metric_", "outputs_2d_", "_y"],
        [[0], [1], [2], [3]],
        [0, 0, 1, 1],
    ),
    EstimatorSample(
        PCA,
        dict(n_components=2),
        EstimatorKind.transform,
        ["components_", "mean_"],
        np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]),
        None,
    ),
    EstimatorSample(
        RobustScaler,
        dict(),
        EstimatorKind.transform,
        ["center_", "scale_"],
        [[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]],
        None,
    ),
    EstimatorSample(
        NuSVR,
        dict(gamma="scale", C=1.0, nu=0.1),
        EstimatorKind.predict,
        ["_sparse", "_gamma"],
        np.arange(50).reshape((10, 5)),
        np.arange(10),
    ),
    EstimatorSample(
        RadiusNeighborsRegressor,
        dict(radius=1.0),
        EstimatorKind.predict,
        ["_y"],
        [[0], [1], [2], [3]],
        [0, 0, 1, 1],
    ),
    EstimatorSample(
        ElasticNetCV,
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
        *make_regression(n_features=2, random_state=0),
    ),
    EstimatorSample(
        SimpleImputer,
        dict(missing_values=np.nan, strategy="mean"),
        EstimatorKind.transform,
        ["statistics_", "indicator_"],
        [[7, 2, 3], [4, np.nan, 6], [10, 5, 9]],
        None,
    ),
    EstimatorSample(
        SelectKBest,
        dict(score_func=chi2, k=20),
        EstimatorKind.transform,
        ["scores_", "pvalues_"],
        *load_digits(return_X_y=True),
    ),
    EstimatorSample(
        HashingVectorizer,
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
        SpectralBiclustering,
        dict(n_clusters=2, random_state=0),
        None,
        ["row_labels_", "column_labels_", "rows_", "columns_"],
        np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]]),
        None,
    ),
]


def get_estimator(estimator_sample):
    estimator = estimator_sample.estimator_class(
        **estimator_sample.estimator_init_kwargs
    )
    estimator.fit(estimator_sample.X, estimator_sample.y)
    return estimator


@pytest.mark.parametrize("estimator_sample", ESTIMATORS)
def test_fit_param_names(estimator_sample):
    estimator = get_estimator(estimator_sample)
    fit_param_names = set(estimator_sample.fit_param_names)
    estimator_fit_attr_names = set(estimators.get_fit_params_dict(estimator))
    if not fit_param_names.issubset(estimator_fit_attr_names):
        # if it's not a subset, this assert is False, it's here to allow easier readability in pytest
        assert fit_param_names == estimator_fit_attr_names

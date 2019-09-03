import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

from alumni import estimators


def get_all_estimators():
    return [
        get_onehotencoder(),
        get_polynomialfeatures(),
        get_standardscaler(),
        get_linearsvc(),
        get_kneighborsclassifier(),
        get_pca(),
    ]


def get_onehotencoder():
    enc = OneHotEncoder(handle_unknown="ignore")
    data = [["Male", 1], ["Female", 3], ["Female", 2]]
    enc.fit(data)
    return (
        enc,
        ["categories", "drop", "sparse", "dtype", "handle_unknown"]
        + ["n_values", "categorical_features"],
        ["categories_", "drop_idx_"],
    )


def get_polynomialfeatures():
    poly = PolynomialFeatures(2)
    data = np.arange(6).reshape(3, 2)
    poly.fit(data)
    return (
        poly,
        ["degree", "interaction_only", "include_bias", "order"],
        ["n_input_features_", "n_output_features_"],
    )


def get_standardscaler():
    scaler = StandardScaler()
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    scaler.fit(data)
    return (
        scaler,
        ["copy", "with_mean", "with_std"],
        ["scale_", "mean_"]  # used in fit
        + ["var_", "n_samples_seen_"],  # used in partial_fit
    )


def get_linearsvc():
    X, y = make_classification(n_features=4, random_state=0)
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X, y)
    return (
        clf,
        [
            "penalty",
            "loss",
            "dual",
            "tol",
            "C",
            "multi_class",
            "fit_intercept",
            "intercept_scaling",
            "class_weight",
            "verbose",
            "random_state",
            "max_iter",
        ],
        ["coef_", "intercept_"] + ["classes_"],
    )


def get_kneighborsclassifier():
    neigh = KNeighborsClassifier(n_neighbors=3)
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh.fit(X, y)
    return (
        neigh,
        [
            "n_neighbors",
            "weights",
            "algorithm",
            "leaf_size",
            "p",
            "metric",
            "metric_params",
            "n_jobs",
        ],
        ["classes_", "effective_metric_", "outputs_2d_", "_y"],
    )


def get_pca():
    pca = PCA(n_components=2)
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca.fit(data)
    return (
        pca,
        [
            "n_components",
            "copy",
            "whiten",
            "svd_solver",
            "tol",
            "iterated_power",
            "random_state",
        ],
        ["components_", "mean_"],
    )


@pytest.mark.parametrize("estimator, attr_names, fit_attr_names", get_all_estimators())
def test_param_names(estimator, attr_names, fit_attr_names):
    assert set(attr_names) == set(estimators.get_params_dict(estimator))


@pytest.mark.parametrize("estimator, attr_names, fit_attr_names", get_all_estimators())
def test_fit_param_names(estimator, attr_names, fit_attr_names):
    fit_param_names = set(fit_attr_names)
    estimator_fit_attr_names = set(estimators.get_fit_params_dict(estimator))
    if not fit_param_names.issubset(estimator_fit_attr_names):
        # if it's not a subset, this assert is False, it's here to allow easier readability in pytest
        assert fit_param_names == estimator_fit_attr_names

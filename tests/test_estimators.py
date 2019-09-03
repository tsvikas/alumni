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


def get_all_estimators():
    return [
        get_onehotencoder(),
        get_polynomialfeatures(),
        get_standardscaler(),
        get_linearsvc(),
        get_kneighborsclassifier(),
        get_pca(),
        get_robustscaler(),
        get_nusvr(),
        get_radiusneighborsregressor(),
        get_elasticnetcv(),
        get_simpleimputer(),
        get_selectkbest(),
        get_hashingvectorizer(),
        get_spectralbiclustering(),
    ]


def get_onehotencoder():
    enc = OneHotEncoder(handle_unknown="ignore")
    data = [["Male", 1], ["Female", 3], ["Female", 2]]
    enc.fit(data)
    return enc, ["categories_", "drop_idx_"]


def get_polynomialfeatures():
    poly = PolynomialFeatures(2)
    data = np.arange(6).reshape(3, 2)
    poly.fit(data)
    return poly, ["n_input_features_", "n_output_features_"]


def get_standardscaler():
    scaler = StandardScaler()
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    scaler.fit(data)
    return (
        scaler,
        ["scale_", "mean_"]  # used in fit
        + ["var_", "n_samples_seen_"],  # used in partial_fit
    )


def get_linearsvc():
    X, y = make_classification(n_features=4, random_state=0)
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X, y)
    return clf, ["coef_", "intercept_"] + ["classes_"]


def get_kneighborsclassifier():
    neigh = KNeighborsClassifier(n_neighbors=3)
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh.fit(X, y)
    return neigh, ["classes_", "effective_metric_", "outputs_2d_", "_y"]


def get_pca():
    pca = PCA(n_components=2)
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca.fit(data)
    return pca, ["components_", "mean_"]


def get_robustscaler():
    transformer = RobustScaler()
    data = [[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]]
    transformer.fit(data)
    return transformer, ["center_", "scale_"]


def get_nusvr():
    n_samples, n_features = 10, 5
    np.random.seed(0)
    y = np.random.randn(n_samples)
    X = np.random.randn(n_samples, n_features)
    clf = NuSVR(gamma="scale", C=1.0, nu=0.1)
    clf.fit(X, y)
    return clf, ["_sparse", "_gamma"]


def get_radiusneighborsregressor():
    neigh = RadiusNeighborsRegressor(radius=1.0)
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh.fit(X, y)
    return neigh, ["_y"]


def get_elasticnetcv():
    regr = ElasticNetCV(cv=5, random_state=0)
    X, y = make_regression(n_features=2, random_state=0)
    regr.fit(X, y)
    return (
        regr,
        [
            "mse_path_",
            "l1_ratio_",
            "alphas_",
            "coef_",
            "intercept_",
            "dual_gap_",
            "n_iter_",
        ],
    )


def get_simpleimputer():
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    data = [[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]
    imp_mean.fit(data)
    return imp_mean, ["statistics_", "indicator_"]


def get_selectkbest():
    sel = SelectKBest(chi2, k=20)
    X, y = load_digits(return_X_y=True)
    sel.fit(X, y)
    return sel, ["scores_", "pvalues_"]


def get_hashingvectorizer():
    vectorizer = HashingVectorizer(n_features=2 ** 4)
    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    vectorizer.fit(corpus)
    return vectorizer, []


def get_spectralbiclustering():
    clustering = SpectralBiclustering(n_clusters=2, random_state=0)
    data = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
    clustering.fit(data)
    return clustering, ["row_labels_", "column_labels_", "rows_", "columns_"]


@pytest.mark.parametrize("estimator, fit_attr_names", get_all_estimators())
def test_fit_param_names(estimator, fit_attr_names):
    fit_param_names = set(fit_attr_names)
    estimator_fit_attr_names = set(estimators.get_fit_params_dict(estimator))
    if not fit_param_names.issubset(estimator_fit_attr_names):
        # if it's not a subset, this assert is False, it's here to allow easier readability in pytest
        assert fit_param_names == estimator_fit_attr_names

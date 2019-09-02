import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from alumni import alumni


def get_all_estimators():
    return [get_onehotencoder(), get_polynomialfeatures(), get_standardscaler()]


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
        ["n_input_features_", "n_output_features_"] + ["powers_"],
    )


def get_standardscaler():
    scaler = StandardScaler()
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    scaler.fit(data)
    return (
        scaler,
        ["copy", "with_mean", "with_std"],
        ["scale_", "mean_", "var_", "n_samples_seen_"],
    )


@pytest.mark.parametrize("estimator, attr_names, fit_attr_names", get_all_estimators())
def test_param_names(estimator, attr_names, fit_attr_names):
    assert set(attr_names) == set(alumni.get_params_dict(estimator))


@pytest.mark.parametrize("estimator, attr_names, fit_attr_names", get_all_estimators())
def test_fit_param_names(estimator, attr_names, fit_attr_names):
    assert set(fit_attr_names) == set(alumni.get_fit_params_dict(estimator))

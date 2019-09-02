from collections.abc import Iterable, Mapping

import numpy as np
import pytest
import tables
from sklearn.preprocessing import OneHotEncoder

from alumni import __version__, alumni


def test_version():
    assert __version__ == "0.0.0"


def is_equal(a, b):
    if a is None:
        return b is None
    try:
        return bool(a == b)
    except ValueError as e:  # trying to compare numpy arrays
        if type(a) != type(b):
            return False
        if isinstance(a, np.ndarray):
            return np.all(a == b)
        elif isinstance(a, Mapping):
            return (list(a.keys()) == list(b.keys())) and all(
                is_equal(a[k], b[k]) for k in a
            )
        elif isinstance(a, Iterable):
            return all(is_equal(a0, b0) for (a0, b0) in zip(a, b))
        else:
            raise RuntimeError("unexpected compare") from e


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


@pytest.mark.parametrize("estimator, attr_names, fit_attr_names", [get_onehotencoder()])
def test_param_names(estimator, attr_names, fit_attr_names):
    assert set(attr_names) == set(alumni.get_params_dict(estimator))


@pytest.mark.parametrize("estimator, attr_names, fit_attr_names", [get_onehotencoder()])
def test_fit_param_names(estimator, attr_names, fit_attr_names):
    assert set(fit_attr_names) == set(alumni.get_fit_params_dict(estimator))


@pytest.mark.parametrize("estimator, attr_names, fit_attr_names", [get_onehotencoder()])
def test_save(tmp_path, estimator, attr_names, fit_attr_names):
    fn = tmp_path / "est.hdf5"
    alumni.save_estimator(fn, estimator, fitted=True)

    with tables.open_file(str(fn), "r") as h:
        # check root
        root_attrs = h.root._v_attrs
        assert root_attrs["protocol_name"] == alumni.PROTOCOL_NAME
        assert root_attrs["protocol_version"] == alumni.PROTOCOL_VERSION

        # check estimator attrs
        est_attrs = h.root[alumni.ESTIMATOR_GROUP]._v_attrs
        assert (
            est_attrs["__class_name__"]
            == f"{estimator.__class__.__module__}.{estimator.__class__.__name__}"
        )
        assert est_attrs["__type__"] == alumni.GroupType.ESTIMATOR.name
        for attr_name in attr_names:
            assert is_equal(est_attrs[attr_name], getattr(estimator, attr_name))

        # check fit attrs
        fit_attrs = h.root[alumni.ESTIMATOR_GROUP][alumni.FIT_GROUP]._v_attrs
        assert fit_attrs["__type__"] == alumni.GroupType.FITTED_ATTRIBUTES.name
        for attr_name in fit_attr_names:
            assert is_equal(fit_attrs[attr_name], getattr(estimator, attr_name))


@pytest.mark.parametrize("estimator, attr_names, fit_attr_names", [get_onehotencoder()])
def test_load(tmp_path, estimator, attr_names, fit_attr_names):
    fn = tmp_path / "est.hdf5"
    alumni.save_estimator(fn, estimator, fitted=True)
    loaded_est = alumni.load_estimator(fn, fitted=True)
    assert type(loaded_est) == type(estimator)
    for attr_name in attr_names + fit_attr_names:
        assert is_equal(getattr(loaded_est, attr_name), getattr(estimator, attr_name))

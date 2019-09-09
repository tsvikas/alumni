from typing import Optional, Union, Iterable

import numpy as np
import pytest
import sklearn.base
import tables

from alumni import __version__, alumni
from tests.test_estimators import ESTIMATORS, get_estimator


def test_version():
    assert __version__ == "0.0.0"


@pytest.mark.parametrize("estimator_sample", ESTIMATORS)
def test_save(tmp_path, estimator_sample):
    # create estimator & save to hdf
    estimator = get_estimator(estimator_sample)
    fn = tmp_path / "est.hdf5"
    alumni.save_estimator(fn, estimator, fitted=True)

    # open hdf and check content
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
        for attr_name in estimator.get_params(deep=False):
            orig_param = getattr(estimator, attr_name)
            if alumni.is_list_of_named_estimators(orig_param):
                # recursive estimator - only check for existence. skip check for identity
                assert attr_name in h.root[alumni.ESTIMATOR_GROUP]
            else:
                saved_param = est_attrs[attr_name]
                assert_equal(saved_param, orig_param)

        # check fit attrs
        fit_attrs = h.root[alumni.ESTIMATOR_GROUP][alumni.FIT_GROUP]._v_attrs
        assert fit_attrs["__type__"] == alumni.GroupType.FITTED_ATTRIBUTES.name
        for attr_name in estimator_sample.fit_param_names:
            assert_equal(fit_attrs[attr_name], getattr(estimator, attr_name))


@pytest.mark.parametrize("estimator_sample", ESTIMATORS)
def test_load(tmp_path, estimator_sample):
    # create estimator & save to hdf
    estimator = get_estimator(estimator_sample)
    fn = tmp_path / "est.hdf5"
    alumni.save_estimator(fn, estimator, fitted=True)

    # load estimator and check it
    fit_attr_names = estimator_sample.fit_param_names
    attr_names = list(estimator.get_params(deep=False))
    loaded_est = alumni.load_estimator(fn)
    assert type(loaded_est) == type(estimator)
    for attr_name in attr_names + fit_attr_names:
        assert_equal(getattr(loaded_est, attr_name), getattr(estimator, attr_name))


def assert_equal(
    actual: Union[np.ndarray, Iterable, int, float],
    desired: Union[np.ndarray, Iterable, int, float],
    err_msg: Optional[str] = "",
    verbose: Optional[bool] = True,
) -> None:
    __tracebackhide__ = True  # Hide traceback for py.test
    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k, i in desired.items():
            if k not in actual:
                raise AssertionError(repr(k))
            assert_equal(actual[k], desired[k], "key=%r\n%s" % (k, err_msg), verbose)
        return
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k in range(len(desired)):
            assert_equal(actual[k], desired[k], "item=%r\n%s" % (k, err_msg), verbose)
        return
    if isinstance(desired, sklearn.base.BaseEstimator):
        if not type(actual) == type(desired):
            raise AssertionError(repr(type(actual)))
        assert_equal(actual.__dict__, desired.__dict__, err_msg, verbose)
        return

    np.testing.assert_equal(actual, desired, err_msg, verbose)

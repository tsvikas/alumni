from pathlib import Path

import pytest
import tables

from alumni import __version__, alumni
from alumni.utils import assert_equal
from tests.test_estimators import ESTIMATORS, get_estimator, EstimatorSample


def test_version():
    assert __version__ == "0.0.0"


@pytest.mark.parametrize("estimator_sample", ESTIMATORS)
def test_save(tmp_path: Path, estimator_sample: EstimatorSample):
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
        est_group = h.root[alumni.ESTIMATOR_GROUP]
        est_attrs = est_group._v_attrs
        assert (
            est_attrs["__class_name__"]
            == f"{estimator.__class__.__module__}.{estimator.__class__.__name__}"
        )
        assert est_attrs["__type__"] == alumni.GroupType.ESTIMATOR.name
        for attr_name in estimator.get_params(deep=False):
            orig_param = getattr(estimator, attr_name)
            assert_param_saved(orig_param, attr_name, est_group)

        # check fit attrs
        fit_group = est_group[alumni.FIT_GROUP]
        fit_attrs = fit_group._v_attrs
        assert fit_attrs["__type__"] == alumni.GroupType.FITTED_ATTRIBUTES.name
        for attr_name in estimator_sample.fit_param_names:
            orig_param = getattr(estimator, attr_name)
            assert_param_saved(orig_param, attr_name, fit_group)


def assert_param_saved(param, param_name: str, group: tables.Group):
    if (
        alumni.is_list_of_named_estimators(param)
        or alumni.is_list_of_estimators(param)
        or alumni.is_estimator(param)
    ):
        # recursive estimator - only check for existence. skip check for identity
        assert param_name in group
    else:
        assert_equal(group._v_attrs[param_name], param)


@pytest.mark.parametrize("estimator_sample", ESTIMATORS)
def test_load(tmp_path: Path, estimator_sample: EstimatorSample):
    # create estimator & save to hdf
    estimator = get_estimator(estimator_sample)
    fn = tmp_path / "est.hdf5"
    alumni.save_estimator(fn, estimator, fitted=True)

    # load estimator
    fit_attr_names = estimator_sample.fit_param_names
    attr_names = list(estimator.get_params(deep=False))
    loaded_est = alumni.load_estimator(fn)
    # check estimator params
    assert type(loaded_est) == type(estimator)
    for attr_name in attr_names + fit_attr_names:
        assert_equal(getattr(loaded_est, attr_name), getattr(estimator, attr_name))
    # check estimator func
    if estimator_sample.kind is not None:
        assert_equal(
            getattr(loaded_est, estimator_sample.kind)(estimator_sample.X),
            getattr(estimator, estimator_sample.kind)(estimator_sample.X),
        )

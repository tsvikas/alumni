import numpy as np
import tables
from sklearn.preprocessing import OneHotEncoder

from alumni import __version__, alumni


def test_version():
    assert __version__ == "0.0.0"


def test_save(tmp_path):
    est = OneHotEncoder(handle_unknown="ignore")
    X = [["Male", 1], ["Female", 3], ["Female", 2]]
    est.fit(X)
    fn = tmp_path / "est.hdf5"
    alumni.save_estimator(fn, est, fitted=True)

    with tables.open_file(str(fn), "r") as h:
        root_attrs = h.root._v_attrs
        assert root_attrs["protocol_name"] == alumni.PROTOCOL_NAME
        assert root_attrs["protocol_version"] == alumni.PROTOCOL_VERSION

        est_attrs = h.root[alumni.ESTIMATOR_GROUP]._v_attrs
        assert (
            est_attrs["__class_name__"]
            == "sklearn.preprocessing._encoders.OneHotEncoder"
        )
        assert est_attrs["__type__"] == alumni.GroupType.ESTIMATOR.name
        assert est_attrs["handle_unknown"] == "ignore"

        fit_attrs = h.root[alumni.ESTIMATOR_GROUP][alumni.FIT_GROUP]._v_attrs
        assert fit_attrs["__type__"] == alumni.GroupType.FITTED_ATTRIBUTES.name
        categories_0, categories_1 = fit_attrs["categories_"]
        assert (categories_0 == np.array(["Female", "Male"], dtype=object)).all()
        assert (categories_1 == np.array([1, 2, 3], dtype=object)).all()

        assert fit_attrs["drop_idx_"] is None


def test_load(tmp_path):
    est = OneHotEncoder(handle_unknown="ignore")
    X = [["Male", 1], ["Female", 3], ["Female", 2]]
    est.fit(X)
    fn = tmp_path / "est.hdf5"
    alumni.save_estimator(fn, est, fitted=True)
    loaded_est = alumni.load_estimator(fn, fitted=True)
    assert type(loaded_est) == type(est)
    assert loaded_est.handle_unknown == est.handle_unknown
    assert len(loaded_est.categories_) == len(est.categories_)
    assert np.all(loaded_est.categories_[0] == est.categories_[0])
    assert np.all(loaded_est.categories_[1] == est.categories_[1])

from typing import Any

import numpy as np
import sklearn.base
import sklearn.tree.tree


def assert_equal(
    actual: Any, desired: Any, err_msg: str = "", verbose: bool = True
) -> None:
    # recursively call itself when needed (copied from np.testing.assert_equal)
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
    # BaseEstimator doesn't define an __eq__ function, compare it's properties
    if isinstance(desired, sklearn.base.BaseEstimator):
        if not type(actual) == type(desired):
            raise AssertionError(repr(type(actual)))
        assert_equal(actual.__dict__, desired.__dict__, err_msg, verbose)
        return
    # Tree doesn't define an __eq__ function, compare it's properties
    if isinstance(desired, sklearn.tree.tree.Tree):
        if not type(actual) == type(desired):
            raise AssertionError(repr(type(actual)))
        actual_tree = {
            k: getattr(actual, k)
            for k in dir(actual)
            if not callable(getattr(actual, k))
        }
        desired_tree = {
            k: getattr(desired, k)
            for k in dir(desired)
            if not callable(getattr(desired, k))
        }
        assert_equal(actual_tree, desired_tree, err_msg, verbose)
        return
    # scipy.sparse.csr.csr_matrix doesn't define an __eq__ function, convert to np.array
    if (
        type(desired).__module__ == "scipy.sparse.csr"
        and type(desired).__name__ == "csr_matrix"
    ):
        if not type(actual) == type(desired):
            raise AssertionError(repr(type(actual)))
        assert_equal(actual.toarray(), desired.toarray(), err_msg, verbose)
        return
    np.testing.assert_equal(actual, desired, err_msg, verbose)

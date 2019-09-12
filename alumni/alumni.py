import enum
from pathlib import Path
from typing import Tuple, Any, List

import tables
from sklearn.base import BaseEstimator

from alumni import utils
from alumni.estimators import get_params_dict, get_fit_params_dict

PROTOCOL_NAME = "sklearn-hdf5"
PROTOCOL_VERSION = "0.1"

HDF_TITLE = "alumni, a python package to save complex scikit-learn estimators"
ESTIMATOR_GROUP = "_estimator"
FIT_GROUP = "_fit"
VALIDATION_GROUP = "_validation"


class GroupType(enum.Enum):
    ESTIMATOR = 1
    FITTED_ATTRIBUTES = 2
    LIST_OF_NAMED_ESTIMATORS = 3
    LIST_OF_ESTIMATORS = 4
    # TODO: KERAS_REGRESSOR/CLASSIFIER/HISTORY


def save_estimator(
    filename: Path, estimator: BaseEstimator, *, validation=None, fitted: bool = True
):
    if Path(filename).exists():
        raise ValueError(f"file {filename} exists")
    with tables.File(str(filename), mode="w", title=HDF_TITLE) as hdf_file:
        # save metadata
        hdf_file.set_node_attr("/", "protocol_name", PROTOCOL_NAME)
        hdf_file.set_node_attr("/", "protocol_version", PROTOCOL_VERSION)
        # save estimator
        group = hdf_file.create_group("/", ESTIMATOR_GROUP)
        _save_estimator_to_group(hdf_file, group, estimator, fitted=fitted)
        # save validation data
        if validation is not None:
            validation_func, validation_X = validation
            group = hdf_file.create_group("/", VALIDATION_GROUP)
            _save_validation_to_group(
                hdf_file, group, estimator, validation_func, validation_X
            )


def _save_validation_to_group(
    hdf_file, group, estimator, validation_func, validation_data
):
    hdf_file.set_node_attr(group, "validation_func", validation_func)
    hdf_file.set_node_attr(group, "X", validation_data)
    hdf_file.set_node_attr(
        group, "y", getattr(estimator, validation_func)(validation_data)
    )


def _save_estimator_to_group(
    hdf_file: tables.File, group: tables.Group, estimator: BaseEstimator, fitted: bool
):
    # save estimator metadata
    class_name = estimator.__class__.__module__ + "." + estimator.__class__.__name__
    module_version = getattr(__import__(estimator.__class__.__module__), "__version__")
    hdf_file.set_node_attr(group, "__class_name__", class_name)
    hdf_file.set_node_attr(group, "__module_version__", module_version)
    hdf_file.set_node_attr(group, "__type__", GroupType.ESTIMATOR.name)

    # save params
    params_dict = get_params_dict(estimator)
    # one would expect that those params are not fitted, and fitted can be set to Flase
    # but some of them (for example pipeline.Pipeline.steps) do includes fitted estimators.
    _save_params_to_group(hdf_file, group, params_dict, fitted=fitted)

    if fitted:
        # create fit group
        fit_group = hdf_file.create_group(group, FIT_GROUP)
        hdf_file.set_node_attr(fit_group, "__type__", GroupType.FITTED_ATTRIBUTES.name)
        # save fit params
        fit_params_dict = get_fit_params_dict(estimator)
        _save_params_to_group(hdf_file, fit_group, fit_params_dict, fitted)


def _save_params_to_group(
    hdf_file: tables.File, group: tables.Group, params_dict: dict, fitted: bool
):
    for param_name, param_value in params_dict.items():
        if is_estimator(param_value):
            param_group = hdf_file.create_group(group, param_name)
            _save_estimator_to_group(hdf_file, param_group, param_value, fitted)
        elif is_list_of_named_estimators(param_value):
            param_group = hdf_file.create_group(group, param_name)
            _save_list_of_named_estimators(hdf_file, param_group, param_value, fitted)
        elif is_list_of_estimators(param_value):
            param_group = hdf_file.create_group(group, param_name)
            _save_list_of_estimators(hdf_file, param_group, param_value, fitted)
        else:
            hdf_file.set_node_attr(group, param_name, param_value)


def is_estimator(param_value):
    return hasattr(param_value, "_get_param_names")


def is_list_of_named_estimators(param_value):
    return (
        isinstance(param_value, list)
        and param_value
        and isinstance(param_value[0], tuple)
        and len(param_value[0]) >= 2
        and is_estimator(param_value[0][1])
    )


def is_list_of_estimators(param_value):
    return (
        isinstance(param_value, list) and param_value and is_estimator(param_value[0])
    )


def _save_list_of_estimators(
    hdf_file: tables.File,
    group: tables.Group,
    estimator_list: List[BaseEstimator],
    fitted: bool,
):
    hdf_file.set_node_attr(group, "__type__", GroupType.LIST_OF_ESTIMATORS.name)
    hdf_file.set_node_attr(group, "len", len(estimator_list))
    for i, estimator in enumerate(estimator_list):
        sub_group = hdf_file.create_group(group, f"item_{i}")
        _save_estimator_to_group(hdf_file, sub_group, estimator, fitted)


def _save_list_of_named_estimators(
    hdf_file: tables.File,
    group: tables.Group,
    estimator_list: List[Tuple[str, BaseEstimator, Any]],
    fitted: bool,
):
    hdf_file.set_node_attr(group, "__type__", GroupType.LIST_OF_NAMED_ESTIMATORS.name)
    hdf_file.set_node_attr(group, "names", [n for (n, e, *r) in estimator_list])
    hdf_file.set_node_attr(group, "rests", [r for (n, e, *r) in estimator_list])
    for (name, estimator, *_rest) in estimator_list:
        sub_group = hdf_file.create_group(group, name)
        _save_estimator_to_group(hdf_file, sub_group, estimator, fitted)


def load_estimator(filename: Path):
    with tables.File(str(filename), mode="r") as hdf_file:
        # check metadata
        assert hdf_file.get_node_attr("/", "protocol_name") == PROTOCOL_NAME
        assert hdf_file.get_node_attr("/", "protocol_version") == PROTOCOL_VERSION
        # load estimator
        group = hdf_file.get_node("/")[ESTIMATOR_GROUP]
        estimator = _load_estimator_from_group(group)
        # load validation
        if VALIDATION_GROUP in hdf_file.root:
            group = hdf_file.get_node("/")[VALIDATION_GROUP]
            validation_func, validation_X, validation_y = _load_validation_from_group(
                group
            )
            utils.assert_equal(
                validation_y, getattr(estimator, validation_func)(validation_X)
            )
        return estimator


def _load_validation_from_group(group: tables.Group):
    user_attrs = _get_user_attrs(group)
    return user_attrs["validation_func"], user_attrs["X"], user_attrs["y"]


def check_version(module_name: str, module_version: str):
    assert module_version == getattr(__import__(module_name), "__version__")


# noinspection PyProtectedMember
def _get_user_attrs(group: tables.Group):
    attrs = group._v_attrs
    user_attrs = {k: attrs[k] for k in attrs._f_list("user")}
    return user_attrs


def _load_estimator_from_group(group: tables.Group):
    user_attrs = _get_user_attrs(group)

    group_type = GroupType[user_attrs.pop("__type__")]
    if group_type == GroupType.ESTIMATOR:
        module_name, class_name = user_attrs.pop("__class_name__").rsplit(".", 1)
        module_version = user_attrs.pop("__module_version__")
        check_version(module_name, module_version)

        for name, subgroup in group._v_groups.items():
            if name != FIT_GROUP:
                user_attrs[name] = _load_estimator_from_group(subgroup)

        assert not [p for p in user_attrs if p.startswith("__")]
        mod = __import__(module_name, fromlist=[class_name])
        klass = getattr(mod, class_name)
        estimator = klass(**user_attrs)

        if FIT_GROUP in group._v_groups:
            fit_group = group[FIT_GROUP]
            fit_user_attrs = _get_user_attrs(fit_group)
            assert (
                GroupType[fit_user_attrs.pop("__type__")] == GroupType.FITTED_ATTRIBUTES
            )
            for name, subgroup in fit_group._v_groups.items():
                fit_user_attrs[name] = _load_estimator_from_group(subgroup)

            assert not [p for p in fit_user_attrs if p.startswith("__")]
            for k, v in fit_user_attrs.items():
                try:
                    setattr(estimator, k, v)
                except AttributeError:
                    # some attributes might be read only
                    pass

        return estimator

    elif group_type == GroupType.FITTED_ATTRIBUTES:
        raise ValueError("_load_estimator_from_group got a group with fitting data")

    elif group_type == GroupType.LIST_OF_NAMED_ESTIMATORS:
        list_of_names_estimators = []
        for name, rest in zip(user_attrs["names"], user_attrs["rests"]):
            list_of_names_estimators.append(
                (name, _load_estimator_from_group(group[name]), *rest)
            )
        return list_of_names_estimators

    elif group_type == GroupType.LIST_OF_ESTIMATORS:
        list_of_estimators = []
        for i in range(user_attrs["len"]):
            list_of_estimators.append(_load_estimator_from_group(group[f"item_{i}"]))
        return list_of_estimators

    raise NotImplementedError(
        f"HDF group type {group_type.name} loading not implemented"
    )

import enum
from pathlib import Path

import tables

PROTOCOL_NAME = "sklearn-hdf5"
PROTOCOL_VERSION = "0.1"

HDF_TITLE = "alumni, a python package to save complex scikit-learn estimators"
ESTIMATOR_GROUP = "_estimator"
FIT_GROUP = "_fit"


class GroupType(enum.Enum):
    ESTIMATOR = 1
    FITTED_ATTRIBUTES = 2
    LIST_OF_NAMED_ESTIMATORS = 3
    # TODO: KERAS_REGRESSOR/CLASSIFIER/HISTORY


def save_estimator(filename, estimator, fitted=True):
    if Path(filename).exists():
        raise ValueError(f"file {filename} exists")
    with tables.File(str(filename), mode="w", title=HDF_TITLE) as hdf_file:
        # save metadata
        hdf_file.set_node_attr("/", "protocol_name", PROTOCOL_NAME)
        hdf_file.set_node_attr("/", "protocol_version", PROTOCOL_VERSION)
        # save estimator
        group = hdf_file.create_group("/", ESTIMATOR_GROUP)
        _save_estimator_to_group(hdf_file, group, estimator, fitted=fitted)
        # TODO: save train / test / validation data


def _save_estimator_to_group(hdf_file, group, estimator, fitted):
    # save estimator metadata
    class_name = estimator.__class__.__module__ + "." + estimator.__class__.__name__
    module_version = getattr(__import__(estimator.__class__.__module__), "__version__")
    hdf_file.set_node_attr(group, "__class_name__", class_name)
    hdf_file.set_node_attr(group, "__module_version__", module_version)
    hdf_file.set_node_attr(group, "__type__", GroupType.ESTIMATOR.name)

    # save params
    params_dict = get_params_dict(estimator)
    _save_params_to_group(hdf_file, group, params_dict, fitted=False)  # TODO: check

    if fitted:
        # create fit group
        fit_group = hdf_file.create_group(group, FIT_GROUP)
        hdf_file.set_node_attr(fit_group, "__type__", GroupType.FITTED_ATTRIBUTES.name)
        # save fit params
        fit_params_dict = get_fit_params_dict(estimator)
        _save_params_to_group(hdf_file, fit_group, fit_params_dict, fitted)


def get_params_dict(estimator):
    params_dict = estimator.get_params(deep=False)
    return params_dict


def get_fit_params_dict(estimator):
    fit_param_names = [
        p for p in dir(estimator) if p.endswith("_") and not p.endswith("__")
    ]
    assert not [p for p in fit_param_names if p.startswith("_")]

    fit_params_dict = {}
    for param_name in fit_param_names:
        try:
            fit_params_dict[param_name] = getattr(estimator, param_name)
        except AttributeError:
            # some attributes might exist (since they are properties) but be uninitialized
            pass
    return fit_params_dict


def _save_params_to_group(hdf_file, group, params_dict, fitted):
    for param_name, param_value in params_dict.items():
        if is_estimator(param_value):
            param_group = hdf_file.create_group(group, param_name)
            _save_estimator_to_group(hdf_file, param_group, param_value, fitted)
        elif is_list_of_named_estimators(param_value):
            param_group = hdf_file.create_group(group, param_name)
            _save_list_of_named_estimators(hdf_file, param_group, param_value, fitted)
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


def _save_list_of_named_estimators(hdf_file, group, estimator_list, fitted):
    hdf_file.set_node_attr(group, "__type__", GroupType.LIST_OF_NAMED_ESTIMATORS.name)
    hdf_file.set_node_attr(group, "names", [n for (n, e, *r) in estimator_list])
    hdf_file.set_node_attr(group, "rests", [r for (n, e, *r) in estimator_list])
    for (name, estimator, *_rest) in estimator_list:
        sub_group = hdf_file.create_group(group, name)
        _save_estimator_to_group(hdf_file, sub_group, estimator, fitted)


def load_estimator(filename, fitted=True):
    with tables.File(str(filename), mode="r") as hdf_file:
        # check metadata
        assert hdf_file.get_node_attr("/", "protocol_name") == PROTOCOL_NAME
        assert hdf_file.get_node_attr("/", "protocol_version") == PROTOCOL_VERSION
        # load estimator
        group = hdf_file.get_node("/")[ESTIMATOR_GROUP]
        estimator = _load_estimator_from_group(hdf_file, group, fitted=fitted)
        return estimator


def check_version(module_name, module_version):
    assert module_version == getattr(__import__(module_name), "__version__")


# noinspection PyProtectedMember
def _get_user_attrs(group):
    attrs = group._v_attrs
    user_attrs = {k: attrs[k] for k in attrs._f_list("user")}
    return user_attrs


def _load_estimator_from_group(hdf_file, group, fitted):
    user_attrs = _get_user_attrs(group)

    group_type = GroupType[user_attrs.pop("__type__")]
    if group_type == GroupType.ESTIMATOR:
        module_name, class_name = user_attrs.pop("__class_name__").rsplit(".", 1)
        module_version = user_attrs.pop("__module_version__")
        check_version(module_name, module_version)

        # TODO: add subgroups to user_attrs

        assert not [p for p in user_attrs if p.startswith("_")]
        mod = __import__(module_name, fromlist=[class_name])
        klass = getattr(mod, class_name)
        estimator = klass(**user_attrs)

        if fitted:
            fit_group = group[FIT_GROUP]
            fit_user_attrs = _get_user_attrs(fit_group)
            assert (
                GroupType[fit_user_attrs.pop("__type__")] == GroupType.FITTED_ATTRIBUTES
            )
            # TODO: add subgroups to fit_user_attrs

            assert not [p for p in fit_user_attrs if p.startswith("_")]
            for k, v in fit_user_attrs.items():
                assert not k.startswith("_")
                try:
                    setattr(estimator, k, v)
                except AttributeError:
                    # some attributes might be read only
                    pass

        return estimator

    elif group_type == GroupType.FITTED_ATTRIBUTES:
        raise ValueError("_load_estimator_from_group got a group with fitting data")

    elif group_type == GroupType.LIST_OF_NAMED_ESTIMATORS:
        # TODO: code here
        raise NotImplementedError("list of tuples")

    raise NotImplementedError("unrecognized group type")

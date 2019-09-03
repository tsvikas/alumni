import warnings

EXTRA_ATTRS_TO_SAVE = {"KNeighborsClassifier": ["_y"]}


def get_params_dict(estimator):
    params_dict = estimator.get_params(deep=False)
    return params_dict


def get_fit_params_dict(estimator):
    fit_param_names = [
        p for p in dir(estimator) if p.endswith("_") and not p.endswith("__")
    ] + EXTRA_ATTRS_TO_SAVE.get(estimator.__class__.__name__, [])
    assert not [p for p in fit_param_names if p.startswith("__")]

    fit_params_dict = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        for param_name in fit_param_names:
            try:
                fit_params_dict[param_name] = getattr(estimator, param_name)
            except AttributeError:
                # some attributes might exist (since they are properties) but be uninitialized
                pass
    return fit_params_dict

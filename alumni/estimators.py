from sklearn.base import BaseEstimator


def get_params_dict(estimator: BaseEstimator):
    params_dict = estimator.get_params(deep=False)
    return params_dict


def get_fit_params_dict(estimator: BaseEstimator):
    params_dict = get_params_dict(estimator)
    return {k: v for (k, v) in estimator.__dict__.items() if k not in params_dict}

from typing import Any

from sklearn.linear_model import LinearRegression, PoissonRegressor, LogisticRegression, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Add XGBoost

def learnerFromString(learner_string : Any) -> Any:
    """
    Retrieve a scikit-learn learner based on a string specification.

    Parameters
    ----------
    learner_string : str
        The string specifying a supported scikit-learn model.

    Returns
    -------
    sklearn.base.BaseEstimator
        An instance of a scikit-learn estimator corresponding to the
        specified string. This object will be used as the learner.
    """

    match learner_string:
        case "LinearRegression":
            return LinearRegression()
        case "LogisticRegression":
            return LogisticRegression()
        case "Ridge":
            return Ridge()
        case "Lasso":
            return Lasso()
        case "PoissonRegressor":
            return PoissonRegressor()
        case "HuberRegressor":
            return HuberRegressor()
        case "DecisionTreeRegressor":
            return DecisionTreeRegressor()
        case "RandomForestRegressor":
            return RandomForestRegressor()
        case "GradientBoostingRegressor":
            return GradientBoostingRegressor()
        case "AdaBoostRegressor":
            return AdaBoostRegressor()
        case "SVR":
            return SVR()
        case "KNeighborsRegressor":
            return KNeighborsRegressor()
        case _:
            raise ValueError(
                "The specified learner string does not correspond to any "\
                "supported learner.\nConsider passing the appropriate "\
                "scikit-learn object directly as an argument.\n"\
                "The accepted strings arguments are:"\
                "\n- LinearRegression"\
                "\n- Ridge"\
                "\n- Lasso"\
                "\n- PoissonRegressor"\
                "\n- DecisionTreeRegressor"\
                "\n- RandomForestRegressor"\
                "\n- GradientBoostingRegressor"\
                "\n- AdaBoostRegressor"\
                "\n- SVR"\
                "\n- KNeighborsRegressor"
            )
import pyAgrum as gum
import pyAgrum.causal as csl

from copy import deepcopy

import pandas as pd
import numpy as np



from sklearn.linear_model import LinearRegression, PoissonRegressor, LogisticRegression, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor



from sklearn.kernel_ridge import KernelRidge

import xgboost as xgb
from sklearn.model_selection import train_test_split

from scipy.interpolate import interp1d

from typing import Any

class estimator:
    """
    """
    def __init__(self, df : pd.DataFrame,
                       cslbn : csl.CausalModel,
                       treatment : str,
                       outcome : str,
                       covariates : set[str],
                       conditional : dict[str : int | float] = None) -> None:
        """
        """
        self.df = df
        self.cslbn = cslbn

        self.T = treatment
        self.Y = outcome
        self.X = covariates
        self.cond = conditional

        self.e = None


    

    def Rlearner2(self, learner : any):
        """
        """
        m = copy.deepcopy(learner)
        m.fit(X=self.df[[*self.X]], y=self.df[self.Y])
        m_pred = m.predict(self.df[[*self.X]])

        e = self.e if self.e is not None else self.propensityScoreFunc()
        e_pred = e.predict_proba(self.df[[*self.X]])[:,1]

        X = self.df[[*self.X]].to_numpy()
        T = self.df[self.T].to_numpy()
        Y = self.df[self.Y].to_numpy()

        Y_tilde = Y - m_pred
        T_tilde = T - e_pred

        if self.cond == None:
            df = self.df[[*self.X]]
        else:
            df = pd.DataFrame(columns=[*self.X], index=[0], data=self.cond)

        lasso = Lasso(1e-4)
        lasso.fit(X=T_tilde.reshape(-1,1) * X, y=Y_tilde)
        return (df.to_numpy() @ lasso.coef_).mean()

    

    def AIPW(self, learner : any):
        """
        """

        mu0 = copy.deepcopy(learner)
        mu1 = copy.deepcopy(learner)

        df0 = self.df[self.df[self.T] == 0]
        df1 = self.df[self.df[self.T] == 1]

        mu0.fit(X=df0[[*self.X]], y=df0[self.Y])
        mu1.fit(X=df1[[*self.X]], y=df1[self.Y])

        e = self.e if self.e is not None else self.propensityScoreFunc()

        if self.cond == None:
            df = self.df[[*self.X]]
        else:
            df = pd.DataFrame(columns=[*self.X], index=[0], data=self.cond)

        e_pred = e.predict_proba(df)[:,1]

        mu0_pred = mu0.predict(df)
        mu1_pred = mu1.predict(df)

        v_func = np.vectorize(lambda e, t, y, mu0, mu1: (t*y - (t-e)*mu1)/e - ((1-t)*y - (t-e)*mu0)/(1-e))
        tau_list = v_func(e_pred, self.df[self.T], self.df[self.Y], mu0_pred, mu1_pred)

        return tau_list.mean()
























def _learnerFromString(learner_string : Any) -> Any:
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


class SLearner:
    """
    A basic implementation of the S-learner based on Kunzel et al. (2018)
    (see https://arxiv.org/abs/1706.03461).
    """

    def __init__(self, learner : str | Any | None = None) -> None:
        """
        Initialize an S-learner.

        Parameters
        ----------
        learner (optional): str | object | None
            Base estimator for all learners.
            If not provided, defaults to LinearRegression.
        """

        if isinstance(learner, str):
            self.learner = _learnerFromString(learner)
        elif learner is None:
            self.learner = _learnerFromString("LinearRegression")
        else:
            self.learner = deepcopy(learner)

    def fit(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series,
            y : np.ndarray | pd.Series,
        ) -> None:
        """
        Fit the inference model.

        Parameters
        ----------
        X : np.matrix | np.ndarray | pd.DataFrame
            The covariate matrix.
        treatment : np.ndarray | pd.Series
            The treatment assignment vector.
        y : np.ndarray | pd.Series,
            The outcome vector.
        """

        self.learner.fit(
            X=pd.concat(
                [pd.DataFrame(X), pd.DataFrame(treatment)],
                axis=1
            ),
            y=np.array(y)
        )

    def predict(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
        )-> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).

        Note: For an S-learner, the ITE is constant and corresponds to the
        Average Treatment Effect (ATE) of the fitted groups, due to the
        use of a single linear model.

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series
            The vector of outcomes.

        Returns
        -------
        np.ndarray
            An array containing the predicted ITE.
        """

        X_control = pd.concat(
            [
                pd.DataFrame(X),
                pd.DataFrame(
                    {self.learner.feature_names_in_[-1]: np.zeros(len(X))},
                    index=pd.DataFrame(X).index
                )
            ], axis=1
        )

        X_treatment = pd.concat(
            [
                pd.DataFrame(X),
                pd.DataFrame(
                    {self.learner.feature_names_in_[-1]: np.ones(len(X))},
                    index=pd.DataFrame(X).index
                )
            ], axis=1
        )

        mu0 = self.learner.predict(X=X_control)
        mu1 = self.learner.predict(X=X_treatment)

        return mu1 - mu0

    def estimate_ate(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
            pretrain : bool = True
        ) -> float:
        """
        Predicts the Average Treatment Effect (ATE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series
            The vector of outcomes.

        Returns
        -------
        float
            The value of the ATE.
        """

        return self.predict(X, treatment, y).mean()

class TLearner:
    """
    A basic implementation of the T-learner based on Kunzel et al. (2018)
    (see https://arxiv.org/abs/1706.03461).
    """

    def __init__(
            self,
            learner : str | Any | None = None,
            control_learner : str | Any | None = None,
            treatment_learner : str | Any | None = None
        ) -> None:
        """
        Initialize an T-learner.

        Parameters
        ----------
        learner (optional): str | object | None
            Base estimator for all learners.
            If not provided, defaults to LinearRegression.
        control_learner (optional): str | object | None
            Estimator for control group outcome.
            Overrides `learner` if specified.
        treatment_learner (optional): str | object | None
            Estimator for treatment group outcome.
            Overrides `learner` if specified.
        """

        if learner is None:
            self.control_learner = _learnerFromString("LinearRegression")
            self.treatment_learner = _learnerFromString("LinearRegression")
        elif isinstance(learner, str):
            self.control_learner = _learnerFromString(learner)
            self.treatment_learner = _learnerFromString(learner)
        else:
            self.treatment_learner = deepcopy(learner)
            self.control_learner = deepcopy(learner)

        if isinstance(control_learner, str):
            self.control_learner = _learnerFromString(control_learner)
        elif control_learner is not None:
            self.control_learner = deepcopy(control_learner)

        if isinstance(treatment_learner, str):
            self.treatment_learner = _learnerFromString(treatment_learner)
        elif treatment_learner is not None:
            self.treatment_learner = deepcopy(treatment_learner)

    def fit(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series,
            y : np.ndarray | pd.Series,
        ) -> None:
        """
        Fit the inference model.

        Parameters
        ----------
        X : np.matrix | np.ndarray | pd.DataFrame
            The covariate matrix.
        treatment : np.ndarray | pd.Series
            The treatment assignment vector.
        y : np.ndarray | pd.Series,
            The outcome vector.
        """

        self.control_learner.fit(X=X[treatment == 0], y=y[treatment == 0])
        self.treatment_learner.fit(X=X[treatment == 1], y=y[treatment == 1])

    def predict(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
        )-> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series | None
            The vector of outcomes.

        Returns
        -------
        np.ndarray
            An array containing the predicted ITE.
        """

        mu0 = self.control_learner.predict(X=X)
        mu1 = self.treatment_learner.predict(X=X)

        return mu1 - mu0

    def estimate_ate(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
            pretrain : bool = True
        ) -> float:
        """
        Predicts the Average Treatment Effect (ATE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series
            The vector of outcomes.

        Returns
        -------
        float
            The value of the ATE.
        """

        return self.predict(X, treatment, y).mean()

class XLearner:
    """
    A basic implementation of the X-learner based on Kunzel et al. (2018)
    (see https://arxiv.org/abs/1706.03461).
    """

    def __init__(
            self,
            learner : str | Any | None = None,
            control_outcome_learner : str | Any | None = None,
            treatment_outcome_learner : str | Any | None = None,
            control_effect_learner : str | Any | None = None,
            treatment_effect_learner : str | Any | None = None,
            propensity_score_learner : str | Any | None = None
        ) -> None:
        """
        Initialize an X-learner.

        Parameters
        ----------
        learner (optional): str | object | None
            Base estimator for all learners.
            If not provided, defaults to LinearRegression.
        control_outcome_learner (optional): str | object | None
            Estimator for control group outcome.
            Overrides `learner` if specified.
        treatment_outcome_learner (optional): str | object | None
            Estimator for treatment group outcome.
            Overrides `learner` if specified.
        control_effect_learner (optional): str | object | None
            Estimator for control group effect.
            Overrides `learner` if specified.
        treatment_effect_learner (optional): str | object | None
            Estimator for treatment group effect.
            Overrides `learner` if specified.
        propensity_score_learner (optional): str | object | None
            Estimator for propensity score.
            If not provided, defaults to LogisticRegression.

        """

        if learner is None:
            self.control_outcome_learner = _learnerFromString("LinearRegression")
            self.treatment_outcome_learner = _learnerFromString("LinearRegression")
            self.control_effect_learner = _learnerFromString("LinearRegression")
            self.treatment_effect_learner = _learnerFromString("LinearRegression")
        elif isinstance(learner, str):
            self.control_outcome_learner = _learnerFromString(learner)
            self.treatment_outcome_learner = _learnerFromString(learner)
            self.control_effect_learner = _learnerFromString(learner)
            self.treatment_effect_learner = _learnerFromString(learner)
        else:
            self.control_outcome_learner = deepcopy(learner)
            self.treatment_outcome_learner = deepcopy(learner)
            self.control_effect_learner = deepcopy(learner)
            self.treatment_effect_learner = deepcopy(learner)

        if isinstance(control_outcome_learner, str):
            self.control_outcome_learner = _learnerFromString(control_outcome_learner)
        elif control_outcome_learner is not None:
            self.control_outcome_learner = deepcopy(control_outcome_learner)

        if isinstance(treatment_outcome_learner, str):
            self.treatment_outcome_learner = _learnerFromString(treatment_outcome_learner)
        elif treatment_outcome_learner is not None:
            self.treatment_outcome_learner = deepcopy(treatment_outcome_learner)

        if isinstance(control_effect_learner, str):
            self.control_effect_learner = _learnerFromString(control_effect_learner)
        elif control_effect_learner is not None:
            self.control_effect_learner = deepcopy(control_effect_learner)

        if isinstance(treatment_effect_learner, str):
            self.treatment_effect_learner = _learnerFromString(treatment_effect_learner)
        elif treatment_effect_learner is not None:
            self.treatment_effect_learner = deepcopy(treatment_effect_learner)

        if propensity_score_learner is None:
            self.propensity_score_learner = _learnerFromString("LogisticRegression")
        elif isinstance(propensity_score_learner, str):
            self.propensity_score_learner = _learnerFromString(propensity_score_learner)
        else:
            self.propensity_score_learner = deepcopy(propensity_score_learner)


    def fit(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series,
            y : np.ndarray | pd.Series,
        ) -> None:
        """
        Fit the inference model.

        Parameters
        ----------
        X : np.matrix | np.ndarray | pd.DataFrame
            The covariate matrix.
        treatment : np.ndarray | pd.Series
            The treatment assignment vector.
        y : np.ndarray | pd.Series,
            The outcome vector.
        """

        self.control_outcome_learner.fit(X=X[treatment == 0], y=y[treatment == 0])
        self.treatment_outcome_learner.fit(X=X[treatment == 1], y=y[treatment == 1])

        Delta0 = self.treatment_outcome_learner.predict(X=X[treatment == 0]) \
            - y[treatment == 0]
        Delta1 = y[treatment == 1] \
            - self.control_outcome_learner.predict(X=X[treatment == 1])

        self.control_effect_learner.fit(X=X[treatment == 0], y=Delta0)
        self.treatment_effect_learner.fit(X=X[treatment == 1], y=Delta1)

        self.propensity_score_learner.fit(X=X, y=treatment)


    def predict(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
        )-> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series | None
            The vector of outcomes.

        Returns
        -------
        np.ndarray
            An array containing the predicted ITE.
        """

        tau0 = self.control_effect_learner.predict(X)
        tau1 = self.treatment_effect_learner.predict(X)
        e = self.propensity_score_learner.predict_proba(X)

        v_func = np.vectorize(lambda e0, e1, t0, t1: e0*t0 + e1*t1)

        return v_func(e[:,0], e[:,1], tau0, tau1)

    def estimate_ate(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
            pretrain : bool = True
        ) -> float:
        """
        Predicts the Average Treatment Effect (ATE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series
            The vector of outcomes.

        Returns
        -------
        float
            The value of the ATE.
        """

        return self.predict(X, treatment, y).mean()

class PStratification:
    """
    A basic implementation of Propensity Stratification estimator
    based on Lunceford et al. (2004)
    (see https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.1903).
    """

    def __init__(
            self,
            propensity_score_learner : str | Any | None = None,
        ) -> None:
        """
        Initialize an P-Stratification estimator.

        Parameters
        ----------
        propensity_score_learner (optional): str | object | None
            Estimator for propensity score.
            If not provided, defaults to LogisticRegression.
        """
        if propensity_score_learner is None:
            self.propensity_score_learner = _learnerFromString("LogisticRegression")
        elif isinstance(propensity_score_learner, str):
            self.propensity_score_learner = _learnerFromString(propensity_score_learner)
        else:
            self.propensity_score_learner = deepcopy(propensity_score_learner)

    def fit(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series,
            y : np.ndarray | pd.Series,
        ) -> None:
        """
        Fit the inference model.

        Parameters
        ----------
        X : np.matrix | np.ndarray | pd.DataFrame
            The covariate matrix.
        treatment : np.ndarray | pd.Series
            The treatment assignment vector.
        y : np.ndarray | pd.Series,
            The outcome vector.
        """
        self.propensity_score_learner.fit(X=X, y=treatment)

    def predict(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
            num_strata : int = None,
        )-> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series | None
            The vector of outcomes.
        num_strata (optional): int
            The number of strata.
            Default is the lenght of X over 1000.
        Returns
        -------
        np.ndarray
            An array containing the predicted ITE.
        """

        if num_strata == None:
            num_strata = len(X)//1000

        e = self.propensity_score_learner.predict_proba(X)[:,1]
        e = pd.DataFrame({"e": e}).sort_values("e")

        data = pd.concat([treatment, y], axis=1).reindex(e.index)

        strata = np.array_split(data, num_strata, axis=0)

        res = np.apply_along_axis(lambda df: print(df[0], df[1]), 0, strata)


        return res

    def estimate_ate(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
            pretrain : bool = True
        ) -> float:
        """
        Predicts the Average Treatment Effect (ATE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series
            The vector of outcomes.

        Returns
        -------
        float
            The value of the ATE.
        """

        return self.predict(X, treatment, y)#.mean()





    def Pstrat(self, num_strata : int = None):
        """
        """
        if num_strata == None:
            num_strata = len(self.df)//1000

        e = self.e if self.e is not None else self.propensityScoreFunc()
        e_pred = e.predict_proba(self.df[[*self.X]])[:,1]
        e_pred = pd.DataFrame({"e":e_pred}).sort_values("e")

        df = self.df.reindex(e_pred.index)
        strata = np.array_split(df, num_strata)
        tau_list = [len(Y)*(Y[Y[self.T] == 1][self.Y].mean() - Y[Y[self.T] == 0][self.Y].mean()) for Y in strata]

        if self.cond == None:
            return sum(tau_list)/len(df)
        else:
            cond_df = pd.DataFrame(columns=[*self.X], index=[0], data=self.cond)
            e_index = e_pred["e"].searchsorted(e.predict_proba(cond_df))[0,1]
            strata_index = -1 if e_index == len(df) else df.index[e_index]//num_strata
            Y = strata[strata_index]
            return Y[Y[self.T] == 1][self.Y].mean() - Y[Y[self.T] == 0][self.Y].mean()

class IPW:
    """
    A basic implementation of the Inverse Propensity Score Weighting (IPW) estimator
    based on Lunceford et al. (2004)
    (see https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.1903).
    """

    def __init__(
            self,
            propensity_score_learner : str | Any | None = None,
        ) -> None:
        """
        Initialize an IPW estimator.

        Parameters
        ----------
        propensity_score_learner (optional): str | object | None
            Estimator for propensity score.
            If not provided, defaults to LogisticRegression.
        """
        if propensity_score_learner is None:
            self.propensity_score_learner = _learnerFromString("LogisticRegression")
        elif isinstance(propensity_score_learner, str):
            self.propensity_score_learner = _learnerFromString(propensity_score_learner)
        else:
            self.propensity_score_learner = deepcopy(propensity_score_learner)

    def fit(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series,
            y : np.ndarray | pd.Series,
        ) -> None:
        """
        Fit the inference model.

        Parameters
        ----------
        X : np.matrix | np.ndarray | pd.DataFrame
            The covariate matrix.
        treatment : np.ndarray | pd.Series
            The treatment assignment vector.
        y : np.ndarray | pd.Series,
            The outcome vector.
        """
        self.propensity_score_learner.fit(X=X, y=treatment)

    def predict(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
        )-> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series | None
            The vector of outcomes.

        Returns
        -------
        np.ndarray
            An array containing the predicted ITE.
        """

        e = self.propensity_score_learner.predict_proba(X)[:,1]
        v_func = np.vectorize(lambda e, t, y: (t/e - (1-t)/(1-e))*y)

        return v_func(e, treatment, y)

    def estimate_ate(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
            pretrain : bool = True
        ) -> float:
        """
        Predicts the Average Treatment Effect (ATE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series
            The vector of outcomes.

        Returns
        -------
        float
            The value of the ATE.
        """

        return self.predict(X, treatment, y).mean()

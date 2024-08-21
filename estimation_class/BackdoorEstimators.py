import pandas as pd
import numpy as np

from typing import Any

from sklearn.base import clone

from estimation_class.Learners import learnerFromString

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
            self.learner = learnerFromString(learner)
        elif learner is None:
            self.learner = learnerFromString("LinearRegression")
        else:
            self.learner = clone(learner)

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
            self.control_learner = learnerFromString("LinearRegression")
            self.treatment_learner = learnerFromString("LinearRegression")
        elif isinstance(learner, str):
            self.control_learner = learnerFromString(learner)
            self.treatment_learner = learnerFromString(learner)
        else:
            self.treatment_learner = clone(learner)
            self.control_learner = clone(learner)

        if isinstance(control_learner, str):
            self.control_learner = learnerFromString(control_learner)
        elif control_learner is not None:
            self.control_learner = clone(control_learner)

        if isinstance(treatment_learner, str):
            self.treatment_learner = learnerFromString(treatment_learner)
        elif treatment_learner is not None:
            self.treatment_learner = clone(treatment_learner)

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
            self.control_outcome_learner = learnerFromString("LinearRegression")
            self.treatment_outcome_learner = learnerFromString("LinearRegression")
            self.control_effect_learner = learnerFromString("LinearRegression")
            self.treatment_effect_learner = learnerFromString("LinearRegression")
        elif isinstance(learner, str):
            self.control_outcome_learner = learnerFromString(learner)
            self.treatment_outcome_learner = learnerFromString(learner)
            self.control_effect_learner = learnerFromString(learner)
            self.treatment_effect_learner = learnerFromString(learner)
        else:
            self.control_outcome_learner = clone(learner)
            self.treatment_outcome_learner = clone(learner)
            self.control_effect_learner = clone(learner)
            self.treatment_effect_learner = clone(learner)

        if isinstance(control_outcome_learner, str):
            self.control_outcome_learner = learnerFromString(control_outcome_learner)
        elif control_outcome_learner is not None:
            self.control_outcome_learner = clone(control_outcome_learner)

        if isinstance(treatment_outcome_learner, str):
            self.treatment_outcome_learner = learnerFromString(treatment_outcome_learner)
        elif treatment_outcome_learner is not None:
            self.treatment_outcome_learner = clone(treatment_outcome_learner)

        if isinstance(control_effect_learner, str):
            self.control_effect_learner = learnerFromString(control_effect_learner)
        elif control_effect_learner is not None:
            self.control_effect_learner = clone(control_effect_learner)

        if isinstance(treatment_effect_learner, str):
            self.treatment_effect_learner = learnerFromString(treatment_effect_learner)
        elif treatment_effect_learner is not None:
            self.treatment_effect_learner = clone(treatment_effect_learner)

        if propensity_score_learner is None:
            self.propensity_score_learner = learnerFromString("LogisticRegression")
        elif isinstance(propensity_score_learner, str):
            self.propensity_score_learner = learnerFromString(propensity_score_learner)
        else:
            self.propensity_score_learner = clone(propensity_score_learner)


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
            self.propensity_score_learner = learnerFromString("LogisticRegression")
        elif isinstance(propensity_score_learner, str):
            self.propensity_score_learner = learnerFromString(propensity_score_learner)
        else:
            self.propensity_score_learner = clone(propensity_score_learner)

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
            self.propensity_score_learner = learnerFromString("LogisticRegression")
        elif isinstance(propensity_score_learner, str):
            self.propensity_score_learner = learnerFromString(propensity_score_learner)
        else:
            self.propensity_score_learner = clone(propensity_score_learner)

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

def __AIPW(self, learner : any):
        """
        """

        mu0 = clone(learner)
        mu1 = clone(learner)

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
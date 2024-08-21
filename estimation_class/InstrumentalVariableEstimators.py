import pandas as pd
import numpy as np

from typing import Any

from sklearn.base import clone

from estimation_class.Learners import learnerFromString

class LATE:
    """
    A basic implementation of the Wald estimand which computes the
    Local Average Treatment Effect.
    """

    def __init__(
            self,
            learner : str | Any | None = None
        ) -> None:
        """
        Initialize an TSLS estimator.

        Parameters
        ----------
        learner (optional): str | object | None
            Base estimator for all learners.
        """
        self.w = None
        self.T = None
        self.y = None


    def fit(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series,
            y : np.ndarray | pd.Series,
            w : np.matrix | np.ndarray | pd.DataFrame
        ) -> None:
        """
        Fit the inference model.

        Parameters
        ----------
        X : np.matrix | np.ndarray | pd.DataFrame
            The covariate matrix.
        w : np.matrix | np.ndarray | pd.DataFrame
            The instrument variable.
        treatment : np.ndarray | pd.Series
            The treatment assignment vector.
        y : np.ndarray | pd.Series,
            The outcome vector.
        """

        self.w = w if isinstance(w, pd.Series) else w.iloc[:,0]

        if set(self.w.unique()) != {0,1}:
            raise ValueError(
                "Instrument must be binary with values 0 and 1 for LATE estimation."
            )

        self.T = treatment
        self.y = y

    def predict(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame = None,
            treatment : np.ndarray | pd.Series = None,
            y : np.ndarray | pd.Series = None,
            w : np.matrix | np.ndarray | pd.DataFrame = None,
        )-> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).

        Parameters
        ----------
        X : np.matrix | np.ndarray | pd.DataFrame
            The covariate matrix.
        w : np.matrix | np.ndarray | pd.DataFrame
            The instrument variable.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series | None
            The vector of outcomes.

        Returns
        -------
        np.ndarray
            An array containing the predicted ITE.
        """

        return [
            (self.y[self.w == 1] - self.y[self.w == 0])/\
            (self.T[self.w == 1] - self.T[self.w == 0])
        ] * len(w)

    def estimate_ate(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame = None,
            treatment : np.ndarray | pd.Series = None,
            y : np.ndarray | pd.Series = None,
            w : np.matrix | np.ndarray | pd.DataFrame = None,
            pretrain : bool = True
        ) -> float:
        """
        Predicts the Average Treatment Effect (ATE).

        Parameters
        ----------
        X : np.matrix | np.ndarray | pd.DataFrame
            The covariate matrix.
        w : np.matrix | np.ndarray | pd.DataFrame
            The instrument variable.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series
            The vector of outcomes.

        Returns
        -------
        float
            The value of the ATE.
        """

        return (self.y[self.w == 1] - self.y[self.w == 0])/\
            (self.T[self.w == 1] - self.T[self.w == 0])

class TSLS:
    """
    A basic implementation of the Two Stage Least-Squares Estimator.
    (see https://scholar.harvard.edu/imbens/files/wo-stage_least_squares_estimation_of_average_causal_effects_in_models_with_variable_treatment_intensity.pdf)
    """

    def __init__(
            self,
            learner : str | Any | None = None,
            treatment_learner : str | Any | None = None,
            outcome_learner : str | Any | None = None
        ) -> None:
        """
        Initialize an TSLS estimator.

        Parameters
        ----------
        learner (optional): str | object | None
            Base estimator for all learners.
            If not provided, defaults to LinearRegression.
        treatment_learner (optional): str | object | None
            Estimator for treatment assignment.
            Overrides `learner` if specified.
        outcome_learner (optional): str | object | None
            Estimator for outcome.
            Overrides `learner` if specified.
        """

        if learner is None:
            self.treatment_learner = learnerFromString("LinearRegression")
            self.outcome_learner = learnerFromString("LinearRegression")
        elif isinstance(learner, str):
            self.treatment_learner = learnerFromString(learner)
            self.outcome_learner = learnerFromString(learner)
        else:
            self.treatment_learner = clone(learner)
            self.outcome_learner = clone(learner)

        if isinstance(treatment_learner, str):
            self.treatment_learner = learnerFromString(treatment_learner)
        elif treatment_learner is not None:
            self.treatment_learner = clone(treatment_learner)

        if isinstance(outcome_learner, str):
            self.outcome_learner = learnerFromString(outcome_learner)
        elif outcome_learner is not None:
            self.outcome_learner = clone(outcome_learner)

    def fit(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series,
            y : np.ndarray | pd.Series,
            w : np.matrix | np.ndarray | pd.DataFrame,
        ) -> None:
        """
        Fit the inference model.

        Parameters
        ----------
        X : np.matrix | np.ndarray | pd.DataFrame
            The covariate matrix.
        w : np.matrix | np.ndarray | pd.DataFrame
            The instrument variable.
        treatment : np.ndarray | pd.Series
            The treatment assignment vector.
        y : np.ndarray | pd.Series,
            The outcome vector.
        """

        self.treatment_learner.fit(
            X=pd.concat([pd.DataFrame(X), pd.DataFrame(w)], axis=1),
            y=treatment
        )

    def predict(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series,
            y : np.ndarray | pd.Series,
            w : np.matrix | np.ndarray | pd.DataFrame,
        )-> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        w : np.matrix | np.ndarray | pd.DataFrame
            The instrument variable.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series
            The vector of outcomes.

        Returns
        -------
        np.ndarray
            An array containing the predicted ITE.
        """

        T_hat = self.treatment_learner.predict(
            X=pd.concat([pd.DataFrame(X), pd.DataFrame(w)], axis=1)
        )
        self.outcome_learner.fit(
            X=pd.concat([pd.DataFrame(X), pd.DataFrame({"T":T_hat})], axis=1),
            y=y
        )

        return np.repeat(self.outcome_learner.coef_, len(X))

    def estimate_ate(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series,
            y : np.ndarray | pd.Series = None,
            w : np.matrix | np.ndarray | pd.DataFrame = None,
            pretrain : bool = True
        ) -> float:
        """
        Predicts the Average Treatment Effect (ATE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        w : np.matrix | np.ndarray | pd.DataFrame
            The instrument variable.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series
            The vector of outcomes.

        Returns
        -------
        float
            The value of the ATE.
        """

        return self.predict(
            X=X,
            treatment=treatment,
            y=y,
            w=w
        ).mean()

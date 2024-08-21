import pandas as pd
import numpy as np

from typing import Any

from sklearn.base import clone

from estimation_class.Learners import learnerFromString

class FrontdoorSLearner:
    """
    Uses the Frontdoor Adjustment Formula, Pearl (1995),
    to derive a S-Learner estimator.
    (see https://www.jstor.org/stable/2337329).
    """

    def __init__(
            self,
            learner : str | Any | None = None,
            propensity_learner : str | Any | None = None
        ) -> None:
        """
        Initialize the Frontdoor Adjustment estimator.

        Parameters
        ----------
        learner (optional): str | object | None
            Estimator for outcome variable.
            If not provided, defaults to LinearRegression.
        propensity_learner (optional): str | object | None
            Estimator for treatment proability.
            If not provided, defaults to LogisticRegression.
        """

        if learner is None:
            self.learner = learnerFromString("LinearRegression")
        elif isinstance(learner, str):
            self.learner = learnerFromString(learner)
        else:
            self.learner = clone(learner)

        if propensity_learner is None:
            self.propensity_learner = learnerFromString("LogisticRegression")
        elif isinstance(propensity_learner, str):
            self.propensity_learner = learnerFromString(propensity_learner)
        else:
            self.propensity_learner = clone(propensity_learner)

        self.treatment_probability = None

    def fit(
            self,
            M : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series,
            y : np.ndarray | pd.Series,
        ) -> None:
        """
        Fit the inference model.

        Parameters
        ----------
        M : np.matrix | np.ndarray | pd.DataFrame
            The mediator matrix.
        treatment : np.ndarray | pd.Series
            The treatment assignment vector.
        y : np.ndarray | pd.Series,
            The outcome vector.
        """

        self.conditional_outcome_learner.fit(
            X=pd.concat(
                [pd.DataFrame(M), pd.DataFrame(treatment)],
                axis=1
            ),
            y=np.array(y)
        )

        self.propensity_learner.fit(
            X=pd.DataFrame(M),
            y=treatment
        )

        self.treatment_probability = treatment.sum() / treatment.count()


    def predict(
            self,
            M : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
        )-> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).

        Parameters
        ----------
        M: np.matrix | np.ndarray | pd.DataFrame
            The matrix of mediators.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series
            The vector of outcomes.

        Returns
        -------
        np.ndarray
            An array containing the predicted ITE.
        """

        M_control = pd.concat(
            [
                pd.DataFrame(M),
                pd.DataFrame(
                    {
                        self.conditional_outcome_learner.feature_names_in_[-1]: \
                            np.zeros(len(M))
                    },
                    index=pd.DataFrame(M).index
                )
            ], axis=1
        )

        M_treatment = pd.concat(
            [
                pd.DataFrame(M),
                pd.DataFrame(
                    {
                        self.conditional_outcome_learner.feature_names_in_[-1]: \
                            np.ones(len(M))
                    },
                    index=pd.DataFrame(M).index
                )
            ], axis=1
        )

        mu0 = self.conditional_outcome_learner.predict(X=M_control)
        mu1 = self.conditional_outcome_learner.predict(X=M_treatment)

        e = self.propensity_learner.predict_proba(X=M)[:,1]
        p = self.treatment_probability
        n = len(M)

        return (e/p - (1-e)/(1-p)) * (mu1*p + mu0*(1-p))

    def estimate_ate(
            self,
            M : np.matrix | np.ndarray | pd.DataFrame,
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

        return self.predict(M, treatment, y).mean()

class FrontdoorTLearner:
    """
    Uses the Frontdoor Adjustment Formula, Pearl (1995),
    to derive a T-Learner estimator.
    (see https://www.jstor.org/stable/2337329).
    """

    def __init__(
            self,
            learner : str | Any | None = None,
            control_outcome_learner : str | Any | None = None,
            treatment_outcome_learner : str | Any | None = None,
            propensity_learner : str | Any | None = None
        ) -> None:
        """
        Initialize the Frontdoor Adjustment estimator.

        Parameters
        ----------
        learner (optional): str | object | None
            Base estimator for all learners.
            If not provided, defaults to LinearRegression.
        control_outcome_learner (optional): str | object | None
            Estimator for control group outcome.
            Overrides `learner` if specified.
        treatment_outcome_learner (optional): str | object | None
            Estimator for control group outcome.
            Overrides `learner` if specified.
        propensity_learner (optional): str | object | None
            Estimator for treatment proability.
            If not provided, defaults to LogisticRegression.
        """

        if learner is None:
            self.control_outcome_learner = learnerFromString("LinearRegression")
            self.treatment_outcome_learner = learnerFromString("LinearRegression")
        elif isinstance(learner, str):
            self.control_outcome_learner = learnerFromString(learner)
            self.treatment_outcome_learner = learnerFromString(learner)
        else:
            self.control_outcome_learner = clone(learner)
            self.treatment_outcome_learner = learnerFromString(learner)

        if isinstance(control_outcome_learner, str):
            self.control_outcome_learner = learnerFromString(control_outcome_learner)
        else:
            self.control_outcome_learner = clone(control_outcome_learner)

        if isinstance(treatment_outcome_learner, str):
            self.treatment_outcome_learner = learnerFromString(treatment_outcome_learner)
        else:
            self.treatment_outcome_learner = clone(treatment_outcome_learner)

        if propensity_learner is None:
            self.propensity_learner = learnerFromString("LogisticRegression")
        elif isinstance(propensity_learner, str):
            self.propensity_learner = learnerFromString(propensity_learner)
        else:
            self.propensity_learner = clone(propensity_learner)

        self.treatment_probability = None


    def fit(
            self,
            M : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series,
            y : np.ndarray | pd.Series,
        ) -> None:
        """
        Fit the inference model.

        Parameters
        ----------
        M : np.matrix | np.ndarray | pd.DataFrame
            The mediator matrix.
        treatment : np.ndarray | pd.Series
            The treatment assignment vector.
        y : np.ndarray | pd.Series,
            The outcome vector.
        """

        self.control_outcome_learner.fit(
            X=M[treatment==0], y=y[treatment==0]
        )

        self.treatment_outcome_learner.fit(
            X=M[treatment==1], y=y[treatment==1]
        )

        self.propensity_learner.fit(
            X=pd.DataFrame(M),
            y=treatment
        )

        self.treatment_probability = treatment.sum() / treatment.count()


    def predict(
            self,
            M : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
        )-> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).

        Parameters
        ----------
        M: np.matrix | np.ndarray | pd.DataFrame
            The matrix of mediators.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series
            The vector of outcomes.

        Returns
        -------
        np.ndarray
            An array containing the predicted ITE.
        """

        mu0 = self.control_outcome_learner.predict(X=M)
        mu1 = self.treatment_outcome_learner.predict(X=M)
        e = self.propensity_learner.predict_proba(X=M)[:,1]
        p = self.treatment_probability
        n = len(M)

        return (e/p - (1-e)/(1-p)) * (mu1*p + mu0*(1-p))

    def estimate_ate(
            self,
            M : np.matrix | np.ndarray | pd.DataFrame,
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

        return self.predict(M, treatment, y).mean()
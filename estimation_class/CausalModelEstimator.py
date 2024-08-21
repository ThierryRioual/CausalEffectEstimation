import pyAgrum as gum
import pyAgrum.causal as csl

import numpy as np
import pandas as pd

class CausalModelEstimator:
    """
    A Causal Baysian Network estimator.
    Uses LazyPropagation from pyAgrum.causal to determine the causal effect.
    """

    def __init__(
            self,
            causal_model : csl.CausalModel,
            treatment : str,
            outcome : str,
            adjustment : str
        ) -> None:
        """
        Initialize an IPW estimator.

        Parameters
        ----------
        propensity_score_learner (optional): str | object | None
            Estimator for propensity score.
            If not provided, defaults to LogisticRegression.
        """
        if isinstance(causal_model, csl.CausalModel):
            self.causal_model = causal_model.clone()
        else:
            raise ValueError("Causal Model cannot be None. ")
        self.treatment = treatment
        self.outcome = outcome

        self.adjustment = adjustment

    def fit(
            self,
            df : pd.DataFrame,
            smoothing_prior : float = 1e-9
        ) -> None:
        """
        Fit the inference model.

        Parameters
        ----------
        df : pd.DataFrame
            The observations.
        smoothing_prior (Optional): float
            The uniform prior distribution. Default is 1e-9.
        """

        parameter_learner = gum.BNLearner(df, self.causal_model.causalBN())
        parameter_learner.useNMLCorrection()
        parameter_learner.useSmoothingPrior(smoothing_prior)

        bn = gum.BayesNet(self.causal_model.causalBN())
        parameter_learner.fitParameters(bn)

        self.causal_model = csl.CausalModel(bn)

        return self.causal_model

    def predict(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            M : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
        )-> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).

        Parameters
        ----------
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of covariates.
        X: np.matrix | np.ndarray | pd.DataFrame
            The matrix of mediators.
        treatment (optional): np.ndarray | pd.Series | None
            The vector of treatment assignments.
        y (optional): np.ndarray | pd.Series | None
            The vector of outcomes.

        Returns
        -------
        np.ndarray
            An array containing the predicted ITE.
        """

        return X.apply(self.__predictRow, axis=1).to_numpy()

    def estimate_ate(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame,
            treatment : np.ndarray | pd.Series | None = None,
            y : np.ndarray | pd.Series | None = None,
            pretrain : bool = True
        )-> np.ndarray:
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

        _, cpt0, _ = csl.causalImpact(
            self.causal_model,
            on=self.outcome,
            doing=self.treatment,
            values={self.treatment:0}
        )

        _, cpt1, _ = csl.causalImpact(
            self.causal_model,
            on=self.outcome,
            doing=self.treatment,
            values={self.treatment:1}
        )

        difference = cpt1 - cpt0
        return difference.expectedValue(
            lambda d : difference.variable(0).numerical(
                d[difference.variable(0).name()]
            )
        )

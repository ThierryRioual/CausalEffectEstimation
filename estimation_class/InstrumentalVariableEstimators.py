import pandas as pd
import numpy as np

from typing import Any

from sklearn.base import clone

from estimation_class.Learners import learnerFromString

from scipy.special import expit
from scipy.optimize import minimize

class Wald:
    """
    A basic implementation of the Wald estimator which computes the
    Local Average Causal Effect (LACE), also know as
    the Local Average Treatment Effect (LATE).
    Only Supports binary instruments.
    """

    def __init__(
            self,
            learner : str | Any | None = None
        ) -> None:
        """
        Initialize an Wald estimator.
        """


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

        if set(np.unique(w.to_numpy())) != {0,1}:
            raise ValueError(
                "Instrument must be binary with values 0 and 1 for LACE estimation."
            )

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

        w = w.iloc[:,0] if isinstance(w, pd.DataFrame) else w
        return [
            (y[w == 1].mean()  - y[w == 0].mean())/\
            (treatment[w == 1].mean()  - treatment[w == 0].mean())
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
        Predicts the Local Average Causal Effect (LACE), 
        also referred to as the Local Average Treatment Effect (LATE). 
        (The term "Average Treatment Effect" (ATE) is used in 
        the method name for compatibility purposes.)

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
            The value of the ACE.
        """

        w = w.iloc[:,0] if isinstance(w, pd.DataFrame) else w
        return (y[w == 1].mean()  - y[w == 0].mean())/\
            (treatment[w == 1].mean()  - treatment[w == 0].mean())

class WaldIPW:
    """
    A basic implementation of the Wald estimand with Inverse Propensity
    Score Weighting which computes the Local Average Causal Effect (LACE).
    (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8021190/).
    Only Supports binary instruments.
    """

    def __init__(
            self,
            iv_probability_learner : str | Any | None = None
        ) -> None:
        """
        Initialize an Wald IPW estimator for LACE.

        Parameters
        ----------
        iv_probability_learner (optional): str | object | None
            Estimator for instrumental variable probability.
            If not provided, defaults to LogisticRegression.
        """
        if iv_probability_learner is None:
            self.iv_probability_learner = learnerFromString("LogisticRegression")
        elif isinstance(iv_probability_learner, str):
            self.iv_probability_learner = learnerFromString(iv_probability_learner)
        else:
            self.iv_probability_learner = clone(iv_probability_learner)

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

        if set(np.unique(w.to_numpy())) != {0,1}:
            raise ValueError(
                "Instrument must be binary with values 0 and 1 for LACE estimation."
            )

        self.iv_probability_learner.fit(X=X, y=w)

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

        w = w.iloc[:,0] if isinstance(w, pd.DataFrame) else w

        e = self.iv_probability_learner.predict_proba(X)[:,1]

        v_func = np.vectorize(
            lambda e, w, d: (w/e - (1-w)/(1-e))*d
        )

        return v_func(e, w, y) / v_func(e, w, treatment).mean()

    def estimate_ate(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame = None,
            treatment : np.ndarray | pd.Series = None,
            y : np.ndarray | pd.Series = None,
            w : np.matrix | np.ndarray | pd.DataFrame = None,
            pretrain : bool = True
        ) -> float:
        """
        Predicts the Local Average Causal Effect (LACE), 
        also referred to as the Local Average Treatment Effect (LATE). 
        (The term "Average Treatment Effect" (ATE) is used in 
        the method name for compatibility purposes.)

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
            The value of the ACE.
        """

        return self.predict(X, treatment, y, w).mean()

class NormalizedWaldIPW:
    """
    A basic implementation of the Wald estimand with Inverse Propensity
    Score Weighting which computes the Local Average Causal Effect (LACE).
    (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8021190/).
    Only Supports binary instruments.
    """

    def __init__(
            self,
            iv_probability_learner : str | Any | None = None
        ) -> None:
        """
        Initialize an Wald IPW estimator for LACE.

        Parameters
        ----------
        iv_probability_learner (optional): str | object | None
            Estimator for instrumental probability score.
            If not provided, defaults to LogisticRegression.
        """
        if iv_probability_learner is None:
            self.iv_probability_learner = learnerFromString("LogisticRegression")
        elif isinstance(iv_probability_learner, str):
            self.iv_probability_learner = learnerFromString(iv_probability_learner)
        else:
            self.iv_probability_learner = clone(iv_probability_learner)

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

        if set(np.unique(w.to_numpy())) != {0,1}:
            raise ValueError(
                "Instrument must be binary with values 0 and 1 for LACE estimation."
            )

        self.iv_probability_learner.fit(X=X, y=w)

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

        w = w.iloc[:,0] if isinstance(w, pd.DataFrame) else w

        e = self.iv_probability_learner.predict_proba(X)[:,1]

        v_func = np.vectorize(
            lambda e, w, d: w*d/e
        )

        return (
            (v_func(e, w, y) / v_func(e, w, 1).mean()) - \
            (v_func(1-e, 1-w, y) / v_func(1-e, 1-w, 1).mean())
        ) / (
            (v_func(e, w, treatment) / v_func(e, w, 1).mean()) - \
            (v_func(1-e, 1-w, treatment) / v_func(1-e, 1-w, 1).mean())
        ).mean()

    def estimate_ate(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame = None,
            treatment : np.ndarray | pd.Series = None,
            y : np.ndarray | pd.Series = None,
            w : np.matrix | np.ndarray | pd.DataFrame = None,
            pretrain : bool = True
        ) -> float:
        """
        Predicts the Local Average Causal Effect (LACE), 
        also referred to as the Local Average Treatment Effect (LATE). 
        (The term "Average Treatment Effect" (ATE) is used in 
        the method name for compatibility purposes.)

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
            The value of the ACE.
        """

        return self.predict(X, treatment, y, w).mean()

class ICSW:
    """
    A basic implementation of the Inverse Complier Score Weighting
    (ICSW) estimator which computes the Average Causal Effect (ACE),
    under the Latent ignorability of compliance assumption.
    Based on Aronow et al. (2013)
    (see https://www.jstor.org/stable/24572676?seq=1).
    Only Supports binary instruments.
    """

    def __init__(
            self,
            learner : str | Any | None = None,
            taker_or_complier_learner : str | Any | None = None,
            conditional_taker_learner : str | Any | None = None
        ) -> None:
        """
        Initialize an Wald IPW estimator for LACE.

        Parameters
        ----------
        iv_probability_learner (optional): str | object | None
            Estimator for instrumental variable probability..
            If not provided, defaults to LogisticRegression.
        """
        if learner is None:
            self.taker_or_complier_learner = learnerFromString("LogisticRegression")
            self.conditional_taker_learner = learnerFromString("LogisticRegression")
        elif isinstance(learner, str):
            self.taker_or_complier_learner = learnerFromString(learner)
            self.conditional_taker_learner = learnerFromString(learner)
        else:
            self.taker_or_complier_learner = clone(learner)
            self.conditional_taker_learner = clone(learner)

        if taker_or_complier_learner is None:
            self.taker_or_complier_learner = learnerFromString("LogisticRegression")
        elif isinstance(taker_or_complier_learner, str):
            self.taker_or_complier_learner = learnerFromString(taker_or_complier_learner)
        else:
            self.taker_or_complier_learner = clone(taker_or_complier_learner)

        if conditional_taker_learner is None:
            self.conditional_taker_learner = learnerFromString("LogisticRegression")
        elif isinstance(conditional_taker_learner, str):
            self.conditional_taker_learner = learnerFromString(conditional_taker_learner)
        else:
            self.conditional_taker_learner = clone(conditional_taker_learner)

        self.link_func = expit
        self.theta_A = None
        self.theta_AC = None


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

        if set(np.unique(w.to_numpy())) != {0,1}:
            raise ValueError(
                "Instrument must be binary with values 0 and 1 for ACE estimation."
            )

        x = X.to_numpy() if not isinstance(X, np.ndarray) else X
        t = treatment.to_numpy() if not isinstance(treatment, np.ndarray) else treatment
        z = w.to_numpy() if not isinstance(w, np.ndarray) else w

        def Likelihood(theta):
            theta_A = theta[:X.shape[1]]
            theta_AC = theta[X.shape[1]:]

            def binomial(t, z, *args):
                x = list(args)
                p = self.link_func(x @ theta_AC) * \
                    (1-self.link_func(x @ theta_A))*z \
                    + self.link_func(x @ theta_AC) * \
                    self.link_func(x @ theta_AC)
                return -np.log(p**t * (1-p)**(1-t))
                
                ## NOT FINISHED

            return

        print(Likelihood(np.zeros(2*X.shape[1])))

        res = minimize(
            fun=Likelihood,
            x0=np.zeros(2*X.shape[1])
        )
        print(res.x)
        return res.x


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

        w = w.iloc[:,0] if isinstance(w, pd.DataFrame) else w

        e = self.iv_probability_learner.predict_proba(X)[:,1]

        v_func = np.vectorize(
            lambda e, w, d: w*d/e
        )

        return (
            (v_func(e, w, y) / v_func(e, w, 1).mean()) - \
            (v_func(1-e, 1-w, y) / v_func(1-e, 1-w, 1).mean())
        ) / (
            (v_func(e, w, treatment) / v_func(e, w, 1).mean()) - \
            (v_func(1-e, 1-w, treatment) / v_func(1-e, 1-w, 1).mean())
        ).mean()

    def estimate_ate(
            self,
            X : np.matrix | np.ndarray | pd.DataFrame = None,
            treatment : np.ndarray | pd.Series = None,
            y : np.ndarray | pd.Series = None,
            w : np.matrix | np.ndarray | pd.DataFrame = None,
            pretrain : bool = True
        ) -> float:
        """
        Predicts the Local Average Causal Effect (LACE), 
        also referred to as the Local Average Treatment Effect (LATE). 
        (The term "Average Treatment Effect" (ATE) is used in 
        the method name for compatibility purposes.)

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
            The value of the ACE.
        """

        return self.predict(X, treatment, y, w).mean()

class TSLS:
    """
    A basic implementation of the Two Stage Least-Squares Estimator.
    (see https://www.npo8.fitEstimator("NormalizedWaldIPW")
po8.estimateCausalEffect()css.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Two-Stage_Least_Squares.pdf)
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
        Predicts the Average Causal Effect (ACE),
        also referred to as the Local Average Treatment Effect (LATE).
        (The term ATE is used in the method name for compatibility purposes.)

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
            The value of the ACE.
        """

        return self.predict(
            X=X,
            treatment=treatment,
            y=y,
            w=w
        ).mean()

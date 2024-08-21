import pyAgrum as gum
import pyAgrum.causal as csl

from typing import Any

import pandas as pd
import numpy as np

from estimation_class.CausalModelEstimator import CausalModelEstimator
from estimation_class.BackdoorEstimators import SLearner, TLearner, XLearner, PStratification, IPW
from estimation_class.FrontdoorEstimators import FrontdoorSLearner, FrontdoorTLearner

class PotentialOutcomes:
    """
    Estimates causal treatment effects using observational or experimental
    data within the Rubin Causal Model framework.
    Performs causal identification through user-specified directed acyclic graphs
    or causal discovery algorithms.
    Identifies confounders using causal networks and applies do-calculus for
    d-separation of treatment assignment from outcome.
    Utilizes advanced statistical estimators and machine learning techniques
    to estimate treatment effects on the outcome.
    """

    def __init__(self, df : pd.DataFrame, causal_model : csl.CausalModel) -> None:
        """
        Initializes the causal estimator instance.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to be used for causal effect estimation.
        """

        self.df = df
        self.causal_model = causal_model
        self.M = None
        self.X = None
        self.T = None
        self.y = None
        self.estimator = None
        self.adjustment = None

    def __conditionalAssertion(self, cond_df : pd.DataFrame) -> None:
        """
        Asserts that the conditional dataframe is not empty

        Parameters
        ----------
        cond_df : pd.DataFrame
            A pandas DataFrame containing values corresponding to the conditional.

        Raises
        ------
        AssertionError
            If the input dataframe is empty, indicating that predictions cannot be made.
        """

        assert len(cond_df) > 0, \
            "No matching instances found in the data for the "\
            "provided conditions.\nPlease ensure the conditions "\
            "are correctly specified or consider using a Pandas "\
            "DataFrame with these conditions containing treatment "\
            "and control instances for estimation purposes."

    def __estimatorFromString(
            self,
            estimator_string : str,
            estimator_params : dict[str, Any] | None,
            adjustment : str
        ) -> None:
        """
        Set the estimator used for estimating the treatment effect based
        on a string identifier.

        Parameters
        ----------
        estimator_string: str
            A string identifying the type of estimator to be used.
            Supported values are "SLearner", "TLearner", "XLearner", and "RLearner".

        estimator_params : dict[str, Any] | None
            A dictionary of parameters to be passed to the estimator's constructor.
            If None, an empty dictionary will be used (default behavior).

        Raises
        ------
        ValueError
            If the provided estimator_string does not correspond to any supported
            estimator.
        """

        if estimator_params is None:
            estimator_params = dict()

        if estimator_string == "CausalModelEstimator":
            self.estimator = CausalModelEstimator(
                self.causal_model,
                self.T,
                self.y,
                **estimator_params
            )
            return

        match adjustment:
            case "frontdoor":
                match estimator_string:
                    case "FrontdoorSLearner":
                        self.estimator = FrontdoorSLearner(**estimator_params)
                    case "FrontdoorTLearner":
                        self.estimator = FrontdoorSLearner(**estimator_params)
                    case _:
                        raise ValueError(
                            "The specified estimator string does not correspond "\
                            "to any supported frontdoor estimator.\nConsider passing "\
                            "the appropriate causalML object directly as an argument."\
                            "\nThe accepted strings arguments are:"\
                            "\n- CausalModelEstimator"\
                            "\n- FrontdoorSLearner"\
                            "\n- FrontdoorTLearner"
                        )

            case "backdoor":
                match estimator_string:
                    case "SLearner":
                        self.estimator = SLearner(**estimator_params)
                    case "TLearner":
                        self.estimator = TLearner(**estimator_params)
                    case "XLearner":
                        self.estimator = XLearner(**estimator_params)
                    case "PStratification":
                        self.estimator = PStratification(**estimator_params)
                    case "IPW":
                        self.estimator = IPW(**estimator_params)
                    case _:
                        raise ValueError(
                            "The specified estimator string does not correspond "\
                            "to any supported backdoor estimator.\nConsider passing "\
                            "the appropriate causalML object directly as an argument."\
                            "\nThe accepted strings arguments are:"\
                            "\n- CausalModelEstimator"\
                            "\n- SLearner"\
                            "\n- TLearner"\
                            "\n- XLearner"\
                            "\n- PStratification"\
                            "\n- IPW"\
                        )

    def identifyAdjustmentSet(
            self,
            treatment : str,
            outcome : str
        ) -> None:
        """
        Identify the sufficent adjustment set of covariates.

        Parameters
        ----------
        treatment : str
            Treatment variable.
        outcome : str
            Outcome variable.
        """

        if set(self.df[treatment].unique()) != {0,1}:
            raise ValueError("Treatment must be binary with values 0 and 1.")

        self.T = treatment
        self.y = outcome

        backdoor = self.causal_model.backDoor(cause=treatment, effect=outcome)
        if backdoor is not None:
            self.adjustment = "backdoor"
            self.X = backdoor
            return self.adjustment

        frondoor = self.causal_model.frontDoor(cause=treatment, effect=outcome)
        if frondoor is not None:
            self.adjustment = "frontdoor"
            self.M = frondoor
            return self.adjustment

    def fitEstimator(
            self,
            estimator : str | Any,
            estimator_params : dict[str, Any] = None,
            fit_params : dict[str, Any] = None
        ) -> None:
        """
        Fits the specified causal effect estimator to the data.

        Parameters
        ----------
        estimator: str | Any
            The estimator to be used. Can be a string identifier for built-in
            estimators or a causalML object.

        estimator_params (optional): dict[str, Any]
            Parameters to initialize the estimator. Keys are parameter names,
            values are the corresponding parameter values. Default is None.

        fit_params (optional): dict[str, Any]
            Additional parameters passed to the fit method of the estimator.
            Keys are parameter names, values are the corresponding parameter
            values. Default is None.
        """

        if estimator == "CausalModelEstimator":
            self.estimator = CausalModelEstimator(self.causal_model, self.T, self.y)
            return self.estimator.fit(self.df)

        if fit_params is None:
            fit_params = dict()

        if type(estimator) is str:
            self.__estimatorFromString(estimator, estimator_params, self.adjustment)
        else:
            self.estimator = estimator

        match self.adjustment:
            case "frontdoor":
                return self.estimator.fit(M = self.df[[*self.M]],
                    treatment = self.df[self.T],
                    y = self.df[self.y],
                    **fit_params
                )
            case "backdoor":
                return self.estimator.fit(X = self.df[[*self.X]],
                    treatment = self.df[self.T],
                    y = self.df[self.y],
                    **fit_params
                )
            case _:
                return

    def estimateCausalEffect(
            self,
            conditional : pd.DataFrame | str | None = None,
            return_ci : bool = False,
            estimation_params : dict[str, Any] = None
        ) -> float | np.ndarray:
        """
        Estimates the causal or treatment effect based on the initialized data.

        Parameters
        ----------
        conditional (optional): pd.DataFrame | str | None
            Specifies conditions for estimating treatment effects.
            If DataFrame, estimates the ITE for each row.
            If string, estimates the CATE. String must be a pandas query.
            If None, estimates the ATE. Default is None.

        return_ci (optional): bool
            If True, returns the confidence interval along with the point estimate.
            Default is False.

        estimation_params (optional): dict[str, Any]
            Additional parameters for the estimation method.
            Keys are parameter names, values are the parameter values.
            Default is None.

        Returns
        -------
        float | np.ndarray
            If return_ci is False, returns the estimated treatment effect as a float
            If return_ci is True, returns a tuple containing:
                - The estimated treatment effect
                - The lower and upper bounds of the confidence interval
        """

        match self.adjustment:
            case "frontdoor":
                return self.estimateFrontdoorCausalEffect(
                    conditional,
                    return_ci,
                    estimation_params
                )
            case "backdoor":
                return self.estimateBackdoorCausalEffect(
                    conditional,
                    return_ci,
                    estimation_params
                )
            case _:
                return

    def estimateFrontdoorCausalEffect(
            self,
            conditional : pd.DataFrame | str | None = None,
            return_ci : bool = False,
            estimation_params : dict[str, Any] = None
        ) -> float | np.ndarray:
        """
        Estimates the causal or treatment effect based on the initialized data
        using frontdoor adjustment.

        Parameters
        ----------
        conditional (optional): pd.DataFrame | str | None
            Specifies conditions for estimating treatment effects.
        return_ci (optional): bool
            If True, returns the confidence interval along with the point estimate.
        estimation_params (optional): dict[str, Any]
            Additional parameters for the estimation method.

        Returns
        -------
        float | np.ndarray
            Returns the estimated treatment effect.
        """

        assert self.estimator is not None, \
            "Please fit an estimator before attempting to make an estimate."

        if estimation_params is None:
            estimation_params = dict()
        # ITE
        if isinstance(conditional, pd.DataFrame):
            conditional = pd.DataFrame(conditional)
            return self.estimator.predict(
                M = conditional[[*self.M]],
                treatment = conditional[self.T],
                y = conditional[self.y],
                **estimation_params
            )
        # CATE
        elif isinstance(conditional, str):
            cond_df = self.df.query(conditional)
            self.__conditionalAssertion(cond_df)
            predictions = self.estimator.predict(
                M = cond_df[[*self.M]],
                treatment = cond_df[self.T],
                y = cond_df[self.y],
                **estimation_params
            )
            return predictions.mean()
        # ATE
        elif conditional is None:
            return self.estimator.estimate_ate(
                M = self.df[[*self.M]],
                treatment = self.df[self.T],
                y = self.df[self.y],
                pretrain = True,
                **estimation_params
            )
        else:
            raise ValueError(
                "Invalid Conditional.\n"\
                "Please use a Pandas DataFrame, string "\
                "or Nonetype as the conditional."
            )

    def estimateBackdoorCausalEffect(
            self,
            conditional : pd.DataFrame | str | None = None,
            return_ci : bool = False,
            estimation_params : dict[str, Any] = None
        ) -> float | np.ndarray:
        """
        Estimates the causal or treatment effect based on the initialized data
        using backdoor adjustment.

        Parameters
        ----------
        conditional (optional): pd.DataFrame | str | None
            Specifies conditions for estimating treatment effects.
        return_ci (optional): bool
            If True, returns the confidence interval along with the point estimate.
        estimation_params (optional): dict[str, Any]
            Additional parameters for the estimation method.

        Returns
        -------
        float | np.ndarray
            Returns the estimated treatment effect.
        """

        assert self.estimator is not None, \
            "Please fit an estimator before attempting to make an estimate."

        if estimation_params is None:
            estimation_params = dict()
        # ITE
        if isinstance(conditional, pd.DataFrame):
            conditional = pd.DataFrame(conditional)
            return self.estimator.predict(
                X = conditional[[*self.X]],
                treatment = conditional[self.T],
                y = conditional[self.y],
                **estimation_params
            )
        # CATE
        elif isinstance(conditional, str):
            cond_df = self.df.query(conditional)
            self.__conditionalAssertion(cond_df)
            predictions = self.estimator.predict(
                X = cond_df[[*self.X]],
                treatment = cond_df[self.T],
                y = cond_df[self.y],
                **estimation_params
            )
            return predictions.mean()
        # ATE
        elif conditional is None:
            return self.estimator.estimate_ate(
                X = self.df[[*self.X]],
                treatment = self.df[self.T],
                y = self.df[self.y],
                pretrain = True,
                **estimation_params
            )
        else:
            raise ValueError(
                "Invalid Conditional.\n"\
                "Please use a Pandas DataFrame, string "\
                "or Nonetype as the conditional."
            )


    def validateCausalEstimate(self):
        """
        """
        pass


import pyAgrum.causal as csl

from typing import Any

import pandas as pd
import numpy as np

from estimation_class.utilsPotentialOutcomes import (
    EmptyConditionException,
    BackdoorException,
    FrontdoorException,
    IVException
)

from estimation_class.CausalModelEstimator import CausalModelEstimator

from estimation_class.BackdoorEstimators import (
    SLearner,
    TLearner,
    XLearner,
    PStratification,
    IPW
)
from estimation_class.FrontdoorEstimators import (
    PlugIn,
    GeneralizedPlugIn
)
from estimation_class.InstrumentalVariableEstimators import (
    Wald,
    WaldIPW,
    NormalizedWaldIPW,
    ICSW,
    TSLS
)

class PotentialOutcomes:
    """
    Estimates causal effects using  data within the
    Neyman-Rubin Potential Outcomes framework.


    This class conducts causal identification through user-specified directed
    acyclic graphs (DAGs). It identifies confounders via causal networks and
    employs do-calculus to achieve d-separation between intervention
    (or treatment assignment) and outcome. The class integrates advanced
    statistical estimators and machine learning techniques to estimate causal
    effects on the outcome.

    Attributes
    ----------
    df : pd.DataFrame
        The dataset used for causal effect estimation.
    causal_model : csl.CausalModel
        The causal model for causal effect identification.
    estimator : Any
        An estimator instance for causal effect estimation, initialized to None.
    adjustment : str or None
        Adjustment strategy or method for causal effect estimation,
        initialized to None.
    w : str or set[str] or None
        Instrumental Variables used in the analysis, initialized to None.
    M : str or set[str] or None
        Mediators used in the analysis, initialized to None.
    X : str or set[str] or None
        Covariates or features used in the analysis, initialized to None.
    T : str or None
        Cause or `Treatment` variable in the dataset, initialized to None.
    y : str or None
        Outcome variable in the dataset, initialized to None.

    Methods
    -------
    identifyAdjustmentSet(
        self,
        treatment : str,
        outcome : str
    ) -> None:
        Identify the sufficent adjustment set of covariates.
    useAdjustment(self, adjustment : str) -> None:
        Select the adjustment used for estimation.
    fitEstimator(
            self,
            estimator : str | Any,
            estimator_params : dict[str, Any] = None,
            fit_params : dict[str, Any] = None
    ) -> None:
        Fits the specified causal effect estimator to the data.
    estimateCausalEffect(
        self,
        conditional : pd.DataFrame | str | None = None,
        return_ci : bool = False,
        estimation_params : dict[str, Any] = None
    ) -> float | np.ndarray:
        Estimate the causal or treatment effect based on the initialized data.

    Raises
    ------
    AssertionError
        If the input dataframe is empty, indicating that predictions 
        cannot be made.
    ValueError
            If the provided estimator_string does not correspond to any
            supported estimator.
    """

    BACKDOOR = "backdoor"
    FRONTDOOR = "generalized frontdoor"
    IV = "conditional instrumental variable"

# Util methods

    def __init__(self, df : pd.DataFrame, causal_model : csl.CausalModel) -> None:
        """
        Initializes the causal estimator instance.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset for causal effect estimation.
        causal_model : csl.CausalModel
            The causal model for causal effect identification.
        """

        self.df = df
        self.causal_model = causal_model
        self.estimator = None
        self.adjustment = None
        self.w = None
        self.M = None
        self.X = None
        self.T = None
        self.y = None

    def __generalizedFrontDoor(
            self,
            intervention : str,
            outcome : str
        ) -> tuple[set[str]] | None:
        """
        Identifies the generalised frontdoor adjustment set and covariates.

        Parameters
        ----------
        intervention : str
            Intervention (treatment) variable.
        outcome : str
            Outcome variable.

        Returns
        -------
        tuple[set[str]] or None
            Set with the names of the mediators,
            set with the names of covariates, or None if not applicable.
        """

        obn = self.causal_model.observationalBN()
        cbn = self.causal_model.causalBN()

        mediators = csl._doorCriteria.nodes_on_dipath(
            obn,
            obn.idFromName(intervention),
            obn.idFromName(outcome)
        )

        mediators = {obn.variable(m).name() for m in mediators}

        backdoor_T_Y = csl.CausalModel(obn).backDoor(intervention, outcome)
        confounders = set() if backdoor_T_Y is None else backdoor_T_Y

        for m in mediators:
            backdoor_T_M =  self.causal_model.backDoor(intervention, m)
            backdoor_M_Y =  self.causal_model.backDoor(m, outcome)
            backdoor_T_M = set() if backdoor_T_M is None else backdoor_T_M
            backdoor_M_Y = set() if backdoor_M_Y is None else backdoor_M_Y
            confounders |=  backdoor_T_M | backdoor_M_Y

        confounders = confounders - {intervention}

        # Did not find a way to clone with latent variables
        mutilated_causal_model = self.causal_model.clone()

        for id in self.causal_model.latentVariablesIds():
            childrens = cbn.children(id)
            childrens = {cbn.variable(c).name() for c in childrens}
            mutilated_causal_model.addLatentVariable(
                cbn.variable(id).name(), tuple(childrens)
            )

        for x in confounders:
            if mutilated_causal_model.existsArc(x, intervention):
                mutilated_causal_model.eraseCausalArc(x, intervention)
            if mutilated_causal_model.existsArc(x, outcome):
                mutilated_causal_model.eraseCausalArc(x, outcome)
            for m in mediators:
                if mutilated_causal_model.existsArc(x, m):
                    mutilated_causal_model.eraseCausalArc(x, m)

        frontdoor = mutilated_causal_model.frontDoor(
            cause=intervention,
            effect=outcome
        )

        #print("ok", frontdoor)
        #frontdoor bug

        return (None, None) if frontdoor is None or len(mediators) == 0 \
            else (frontdoor, confounders)

    def __instrumentalVariable(
            self,
            intervention : str,
            outcome : str
        ) -> tuple[set[str]] | None:
        """
        Identifies the instrumental variables and covariates.
        (see https://ftp.cs.ucla.edu/pub/stat_ser/r303-reprint.pdf)

        Parameters
        ----------
        intervention : str
            Intervention (treatment) variable.
        outcome : str
            Outcome variable.

        Returns
        -------
        tuple[set[str]] or None
            Set with the names of the instrumental variables,
            set with the names of covariates, or None if not applicable.
        """

        # TO BE IMPLEMENTED: https://ftp.cs.ucla.edu/pub/stat_ser/r303-reprint.pdf

        return None, None

# Causal identification

    def useBackdoorAdjustment(
            self,
            intervention : str,
            outcome : str,
            confounders : set[str] | None = None
        ) -> None:
        """
        Specify the Backdoor Adjustment.

        Parameters
        ----------
        intervention : str
            Intervention (or treatment) variable.
        outcome : str
            Outcome variable.
        confounders : set[str] or None, optional
            Set of confounder variables (or covariates).
        """

        self.adjustment = self.BACKDOOR
        self.T = intervention
        self.y = outcome
        self.X = set() if confounders is None else confounders

    def useFrontdoorAdjustment(
            self,
            intervention : str,
            outcome : str,
            mediators : set[str],
            confounders : set[str] | None = None
        ) -> None:
        """
        Specify the (General) Frontdoor Adjustment.

        Parameters
        ----------
        intervention : str
            Intervention (or treatment) variable.
        outcome : str
            Outcome variable.
        mediators : set[str]
            Mediator variables.
        confounders : set[str] or None, optional
            Set of confounder variables (or covariates).
        """

        self.adjustment = self.FRONTDOOR
        self.T = intervention
        self.y = outcome
        self.M = mediators
        self.X = set() if confounders is None else confounders

    def useIVAdjustment(
            self,
            intervention : str,
            outcome : str,
            instrument : str,
            confounders : set[str] | None = None
        ) -> None:
        """
        Specify the (Conditional) Instrumental Variable Adjustment.

        Parameters
        ----------
        intervention : str
            Intervention (or treatment) variable.
        outcome : str
            Outcome variable.
        instruments : str
            Instrumental variable.
        confounders : set[str] or None, optional
            Set of confounder variables (or covariates).
        """

        self.adjustment = self.IV
        self.T = intervention
        self.y = outcome
        self.w = instrument
        self.X = set() if confounders is None else confounders

    def identifyAdjustmentSet(
            self,
            intervention : str,
            outcome : str
        ) -> None:
        """
        Identify the sufficent adjustment set of covariates.

        Parameters
        ----------
        intervention : str
            Intervention (treatment) variable.
        outcome : str
            Outcome variable.

        Raises
        ------
        ValueError
            The tratment isn't binary or no adjustment set was found.
        """

        if set(self.df[intervention].unique()) != {0,1}:
            raise ValueError("Treatment must be binary with values 0 and 1.")

        backdoor = self.causal_model.backDoor(
            intervention, outcome)
        frontdoor, fd_covariates = self.__generalizedFrontDoor(
            intervention, outcome)
        instrumental_variable, iv_covariates = self.__instrumentalVariable(
            intervention, outcome)

        if backdoor is not None:
            self.useBackdoorAdjustment(
                intervention,
                outcome,
                backdoor
            )
        elif frontdoor is not None:
            self.useFrontdoorAdjustment(
                intervention,
                outcome,
                frontdoor,
                fd_covariates
            )
        elif instrumental_variable is not None:
            self.useIVAdjustment(
                intervention,
                outcome,
                instrumental_variable,
                iv_covariates
            )
        else:
            raise ValueError("No adjustment set found.")

        return self.adjustment

# Model fitting estimation

    def _fitEstimator(self, **fit_params) -> None:
        """
        Fits the specified causal effect estimator to the data.

        Parameters
        ----------
        estimator : str or Any
            The estimator to be used. Can be a string identifier for built-in
            estimators or a causalML object.

        estimator_params : dict[str, Any], optional
            Parameters to initialize the estimator. Keys are parameter names,
            values are the corresponding parameter values. Default is None.

        fit_params : dict[str, Any], optional
            Additional parameters passed to the fit method of the estimator.
            Keys are parameter names, values are the corresponding parameter
            values. Default is None.

        Raises
        ------
        ValueError
            No adjustment have been selected before fitting an estimator.
        """

        match self.adjustment:

            case self.IV:
                try:
                    return self.estimator.fit(
                        X = self.df[[*self.X]],
                        treatment = self.df[self.T],
                        y = self.df[self.y],
                        w = self.df[[*self.w]],
                        **fit_params
                    )
                except TypeError:
                    return self.estimator.fit(
                        X = self.df[[*self.X]],
                        treatment = self.df[self.T],
                        y = self.df[self.y],
                        assignment = self.df[[*self.w]],
                        **fit_params
                    )

            case self.FRONTDOOR:
                return self.estimator.fit(
                    X = self.df[[*self.X]],
                    treatment = self.df[self.T],
                    y = self.df[self.y],
                    M = self.df[[*self.M]],
                    **fit_params
                )

            case self.BACKDOOR:
                return self.estimator.fit(
                    X = self.df[[*self.X]],
                    treatment = self.df[self.T],
                    y = self.df[self.y],
                    **fit_params
                )

            case _:
                raise AssertionError(
                    "Please select an adjustment before fitting an estimator."
                )

    def fitCausalModelEstimator(self, **estimator_params) -> Any:
        """
        Fit the Causal Model Estimator.

        Parameters
        ----------
        **estimator_params : Any
        The parameters of the estimator

        Returns
        -------
        The estimator object.
        """

        self.estimator = CausalModelEstimator(
            self.causal_model,
            self.T,
            self.y,
        )

        self.estimator.fit(self.df)

    def fitCustomEstimator(self, estimator : Any) -> Any:
        """
        Fits the specified `estimator` object, which must implement
        `.fit()`, `.predict()`, and `.estimate_ate()` methods consistent 
        with CausalML estimators. 

        Note: Compatibility with the current adjustment is not verified.

        Parameters
        ----------
        estimator : Any
            The estimator object to be fitted, adhering to the CausalML 
            method declarations.
        """

        self.estimator = estimator
        self._fitEstimator()

# Backdoor

    def fitSLearner(self, **estimator_params) -> Any:
        """
        Fit the S-Learner Estimator.

        Parameters
        ----------
        estimator_params : Any
            The parameters of the estimator.
        """

        if self.adjustment is self.FRONTDOOR:
            raise FrontdoorException("SLearner")
        if self.adjustment is self.IV:
            raise IVException("SLearner")

        self.estimator = SLearner(**estimator_params)
        self._fitEstimator()

    def fitTLearner(self, **estimator_params) -> Any:
        """
        Fit the T-Learner Estimator.

        Parameters
        ----------
        estimator_params : Any
            The parameters of the estimator.
        """

        if self.adjustment is self.FRONTDOOR:
            raise FrontdoorException("TLearner")
        if self.adjustment is self.IV:
            raise IVException("TLearner")

        self.estimator = TLearner(**estimator_params)
        self._fitEstimator()

    def fitXLearner(self, **estimator_params) -> Any:
        """
        Fit the X-Learner Estimator.

        Parameters
        ----------
        estimator_params : Any
            The parameters of the estimator.
        """

        if self.adjustment is self.FRONTDOOR:
            raise FrontdoorException("XLearner")
        if self.adjustment is self.IV:
            raise IVException("XLearner")

        self.estimator = XLearner(**estimator_params)
        self._fitEstimator()

    def fitPStratification(self, **estimator_params) -> Any:
        """
        Fit the Propensity score Stratification Estimator.

        Parameters
        ----------
        estimator_params : Any
            The parameters of the estimator.
        """

        if self.adjustment is self.FRONTDOOR:
            raise FrontdoorException("PStratification")
        if self.adjustment is self.IV:
            raise IVException("PStratification")

        self.estimator = PStratification(**estimator_params)
        self._fitEstimator()

    def fitIPW(self, **estimator_params) -> Any:
        """
        Fit the Inverse Propensity score Weighting Estimator.

        Parameters
        ----------
        estimator_params : Any
            The parameters of the estimator.
        """

        if self.adjustment is self.FRONTDOOR:
            raise FrontdoorException("IPW")
        if self.adjustment is self.IV:
            raise IVException("IPW")

        self.estimator = IPW(**estimator_params)
        self._fitEstimator()

# Frontdoor

    def fitPlugIn(self, **estimator_params) -> Any:
        """
        Fit the Plug-in Estimator.

        Parameters
        ----------
        estimator_params : Any
            The parameters of the estimator.
        """

        if self.adjustment is self.BACKDOOR:
            raise BackdoorException("PlugIn")
        if self.adjustment is self.IV:
            raise IVException("PlugIn")

        self.estimator = PlugIn(**estimator_params)
        self._fitEstimator()

    def fitGeneralizedPlugIn(self, **estimator_params) -> Any:
        """
        Fit the Generalized plug-in Estimator.

        Parameters
        ----------
        estimator_params : Any
            The parameters of the estimator.
        """

        if self.adjustment is self.BACKDOOR:
            raise BackdoorException("GeneralizedPlugIn")
        if self.adjustment is self.IV:
            raise IVException("GeneralizedPlugIn")

        self.estimator = GeneralizedPlugIn(**estimator_params)
        self._fitEstimator()

# Instrumental Variable

    def fitWald(self, **estimator_params) -> Any:
        """
        Fit the Wald Estimator.

        Parameters
        ----------
        estimator_params : Any
            The parameters of the estimator.
        """

        if self.adjustment is self.BACKDOOR:
            raise BackdoorException("Wald")
        if self.adjustment is self.FRONTDOOR:
            raise FrontdoorException("Wald")

        self.estimator = Wald(**estimator_params)
        self._fitEstimator()

    def fitWaldIPW(self, **estimator_params) -> Any:
        """
        Fit the Wald Inverse Probability Weighting Estimator.

        Parameters
        ----------
        estimator_params : Any
            The parameters of the estimator.
        """

        if self.adjustment is self.BACKDOOR:
            raise BackdoorException("WaldIPW")
        if self.adjustment is self.FRONTDOOR:
            raise FrontdoorException("WaldIPW")

        self.estimator = WaldIPW(**estimator_params)
        self._fitEstimator()

    def fitNormalizedWaldIPW(self, **estimator_params) -> Any:
        """
        Fit the Normalized Wald Inverse Probability Weighting Estimator.

        Parameters
        ----------
        estimator_params : Any
            The parameters of the estimator.
        """

        if self.adjustment is self.BACKDOOR:
            raise BackdoorException("NormalizedWaldIPW")
        if self.adjustment is self.FRONTDOOR:
            raise FrontdoorException("NormalizedWaldIPW")

        self.estimator = NormalizedWaldIPW(**estimator_params)
        self._fitEstimator()

    def fitTSLS(self, **estimator_params) -> Any:
        """
        Fit the Two Stage Least Squares Estimator.

        Parameters
        ----------
        estimator_params : Any
            The parameters of the estimator
        """

        if self.adjustment is self.BACKDOOR:
            raise BackdoorException("TSLS")
        if self.adjustment is self.FRONTDOOR:
            raise FrontdoorException("TSLS")

        self.estimator = TSLS(**estimator_params)
        self._fitEstimator()

# Causal estimation

    def estimateCausalEffect(
            self,
            conditional : pd.DataFrame | str | None = None,
            return_ci : bool = False,
            estimation_params : dict[str, Any] = None
        ) -> float | np.ndarray:
        """
        Estimate the causal or treatment effect based on the initialized data.

        Parameters
        ----------
        conditional : pd.DataFrame, str, or None, optional
            Specifies conditions for estimating causal effects.
            - If `pd.DataFrame`, estimates the Individual Causal Effect (ICE)
                for each row.
            - If `str`, estimates the Conditional Average Causal Effect (CACE).
                The string must be a valid pandas query.
            - If `None`, estimates the Average Causal Effect (ACE). 
            Default is `None`.
        return_ci : bool, optional
            If `True`, returns the confidence interval along with the point
            estimate. Default is `False`.
        estimation_params : dict of str to Any, optional
            Additional parameters for the estimation method. 
            Keys are parameter names, and values are the corresponding parameter 
            values. Default is `None`.

        Returns
        -------
        float or np.ndarray
            - If `return_ci` is `False`, returns the estimated causal effect
                as a float.
            - If `return_ci` is `True`, returns a tuple containing:
            - The estimated causal effect (float)
            - The lower and upper bounds of the confidence interval
                (tuple of floats)

        Raises
        ------
        ValueError
            No adjustment have been selected before making the estimate.
        """

        assert self.estimator is not None, \
            "Please fit an estimator before attempting to make an estimate."

        match self.adjustment:
            case self.IV:
                return self._estimateIVCausalEffect(
                    conditional,
                    return_ci,
                    estimation_params
                )
            case self.FRONTDOOR:
                return self._estimateFrontdoorCausalEffect(
                    conditional,
                    return_ci,
                    estimation_params
                )
            case self.BACKDOOR:
                return self._estimateBackdoorCausalEffect(
                    conditional,
                    return_ci,
                    estimation_params
                )
            case _:
                raise ValueError(
                    "Please select an adjustment before making an estimate."
                )

    def _estimateIVCausalEffect(
            self,
            conditional : pd.DataFrame | str | None = None,
            return_ci : bool = False,
            estimation_params : dict[str, Any] = None
        ) -> float | np.ndarray:
        """
        Estimate the causal or treatment effect using instrumental
        variable adjustment.

        Parameters
        ----------
        conditional : pd.DataFrame, str, or None, optional
            Conditions for estimating causal effects.
        return_ci : bool, optional
            If True, returns the confidence interval with the estimate.
        estimation_params : dict[str, Any], optional
            Additional parameters for the estimation method.

        Returns
        -------
        float or np.ndarray
            The estimated causal effect.

        Raises
        ------
        ValueError
            The inputed conditional is invalid.
        """

        assert self.estimator is not None, \
            "Please fit an estimator before attempting to make an estimate."

        if estimation_params is None:
            estimation_params = dict()
        # ICE
        if isinstance(conditional, pd.DataFrame):
            conditional = pd.DataFrame(conditional)
            return self.estimator.predict(
                X = conditional[[*self.X]],
                w = conditional[[*self.w]],
                treatment = conditional[self.T],
                y = conditional[self.y],
                **estimation_params
            )
        # CACE
        elif isinstance(conditional, str):
            cond_df = self.df.query(conditional)
            if len(cond_df) == 0:
                raise EmptyConditionException()
            predictions = self.estimator.predict(
                X = cond_df[[*self.X]],
                w = cond_df[[*self.w]],
                treatment = cond_df[self.T],
                y = cond_df[self.y],
                **estimation_params
            )
            return predictions.mean()
        # ACE
        elif conditional is None:
            return self.estimator.estimate_ate(
                X = self.df[[*self.X]],
                w = self.df[[*self.w]],
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

    def _estimateFrontdoorCausalEffect(
            self,
            conditional : pd.DataFrame | str | None = None,
            return_ci : bool = False,
            estimation_params : dict[str, Any] = None
        ) -> float | np.ndarray:
        """
        Estimate the causal or treatment effect using generalized
        frontdoor adjustment.

        Parameters
        ----------
        conditional : pd.DataFrame, str, or None, optional
           Conditions for estimating treatment effects.
        return_ci : bool, optional
            If True, returns the confidence interval with the estimate.
        estimation_params : dict[str, Any], optional
            Additional parameters for the estimation method.

        Returns
        -------
        float or np.ndarray
            The estimated treatment effect.

        Raises
        ------
        ValueError
            The inputed conditional is invalid.
        """

        assert self.estimator is not None, \
            "Please fit an estimator before attempting to make an estimate."

        if estimation_params is None:
            estimation_params = dict()
        # ICE
        if isinstance(conditional, pd.DataFrame):
            conditional = pd.DataFrame(conditional)
            return self.estimator.predict(
                X = conditional[[*self.X]],
                treatment = conditional[self.T],
                y = conditional[self.y],
                M = conditional[[*self.M]],
                **estimation_params
            )
        # CACE
        elif isinstance(conditional, str):
            cond_df = self.df.query(conditional)
            if len(cond_df) == 0:
                raise EmptyConditionException()
            predictions = self.estimator.predict(
                X = cond_df[[*self.X]],
                treatment = cond_df[self.T],
                y = cond_df[self.y],
                M = cond_df[[*self.M]],
                **estimation_params
            )
            return predictions.mean()
        # ACE
        elif conditional is None:
            return self.estimator.estimate_ate(
                X = self.df[[*self.X]],
                treatment = self.df[self.T],
                y = self.df[self.y],
                M = self.df[[*self.M]],
                pretrain = True,
                **estimation_params
            )
        else:
            raise ValueError(
                "Invalid Conditional.\n"\
                "Please use a Pandas DataFrame, string "\
                "or Nonetype as the conditional."
            )

    def _estimateBackdoorCausalEffect(
            self,
            conditional : pd.DataFrame | str | None = None,
            return_ci : bool = False,
            estimation_params : dict[str, Any] = None
        ) -> float | np.ndarray:
        """
        Estimate the causal or treatment effect using backdoor adjustment.

        Parameters
        ----------
        conditional : pd.DataFrame, str, or None, optional
            Specifies conditions for estimating treatment effects.
        return_ci : bool, optional
            If True, returns the confidence interval along with the point
            estimate.
        estimation_params : dict[str, Any], optional
            Additional parameters for the estimation method.

        Returns
        -------
        float or np.ndarray
            The estimated treatment effect.

        Raises
        ------
        ValueError
            The inputed conditional is invalid.
        """

        assert self.estimator is not None, \
            "Please fit an estimator before attempting to make an estimate."

        if estimation_params is None:
            estimation_params = dict()
        # ICE
        if isinstance(conditional, pd.DataFrame):
            conditional = pd.DataFrame(conditional)
            return self.estimator.predict(
                X = conditional[[*self.X]],
                treatment = conditional[self.T],
                y = conditional[self.y],
                **estimation_params
            )
        # CACE
        elif isinstance(conditional, str):
            cond_df = self.df.query(conditional)
            if len(cond_df) == 0:
                raise EmptyConditionException()
            predictions = self.estimator.predict(
                X = cond_df[[*self.X]],
                treatment = cond_df[self.T],
                y = cond_df[self.y],
                **estimation_params
            )
            return predictions.mean()
        # ACE
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


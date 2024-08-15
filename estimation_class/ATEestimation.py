import pandas as pd

import causalml

import causalml.inference.meta.slearner as slearner
import causalml.inference.meta.tlearner as tlearner
import causalml.inference.meta.xlearner as xlearner
import causalml.inference.meta.rlearner as rlearner

import pyAgrum as gum

from typing import Any

class ATEestimation:
    """
    """

    def __init__(self, df : pd.DataFrame) -> None:
        """
        """

        self.df = df
        self.X = None
        self.T = None
        self.y = None
        self.__DAG = None
        self.estimator = None
        pass

    def __conditionalAssertion(self, cond_df):
        assert len(cond_df) > 0, \
            "No matching instances found in the data for the "\
            "provided conditions. Please ensure the conditions "\
            "are correctly specified or consider using a Pandas "\
            "DataFrame with these conditions containing treatment "\
            "and control instances for estimation purposes."

    def identifyCausalStructure(self, treatment : str, outcome : str):
        """Structure Learning (Non paramtric identification)
        """
        self.T = treatment
        self.y = outcome

    def useCausalStructure(self, DAG, treatment : str, outcome : str):
        """Use DAG
        """
        self.__DAG = DAG
        self.T = treatment
        self.y = outcome

    def fitEstimator(
            self,
            estimator : str | Any,
            estimator_params : dict[str, Any] = None,
            fit_params : dict[str, Any] = None
        ) -> None:
        """causalml
        """

        if estimator_params is None:
            estimator_params = dict()
        if fit_params is None:
            fit_params = dict()

        if type(estimator) is str:
            match estimator:
                case "SLearner":
                    self.estimator = slearner.BaseSLearner(**estimator_params)
                case "TLearner":
                    self.estimator = tlearner.BaseTLearner(**estimator_params)
                case "XLearner":
                    self.estimator = xlearner.BaseXLearner(**estimator_params)
                case "RLearner":
                    self.estimator = rlearner.BaseRLearner(**estimator_params)
        else:
            self.estimator = estimator

        self.estimator.fit(X = self.df[[*self.X]],
                           treatment = self.df[self.T],
                           y = self.df[self.y],
                           **fit_params
                           )

    def estimateCausalEffect(
            self,
            conditional : pd.DataFrame | str | None = None,
            return_ci : bool = False,
            estimation_params : dict[str, Any] = None
        ):
        """causalml
        """
        assert self.estimator is not None, \
            "Please fit an estimator before attempting to make an estimate."
        if estimation_params is None:
            estimation_params = dict()
        # ITE
        if isinstance(conditional, (pd.DataFrame, pd.Series)):
            conditional = pd.DataFrame(conditional)
            return self.estimator.predict(
                X = conditional[[*self.X]].to_numpy(),
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
            raise AssertionError("Invalid Conditional")


    def validateCausalEstimate(self):
        """
        """
        pass

    
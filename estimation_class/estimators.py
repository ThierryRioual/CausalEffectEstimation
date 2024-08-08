import pyAgrum as gum
import pyAgrum.causal as csl

import copy

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression

class estimator:
    """
    """
    def __init__(self, df : pd.DataFrame,
                       cslbn : csl.CausalModel,
                       treatment : str,
                       outcome : str,
                       covariates : set[str]) -> None:
        """
        """
        self.df = df
        self.cslbn = cslbn

        self.T = treatment
        self.Y = outcome
        self.X = covariates

        self.e = None

    def propensityScoreFunc(self):
        """
        """
        e = LogisticRegression()
        e.fit(X=self.df[[*self.X]], y=self.df[self.T])
        self.e = e

        return e

    def Slearner(self, learner : any):
        """
        """
        mu = copy.deepcopy(learner)

        mu.fit(X=self.df[[*self.X]+[self.T]], y=self.df[self.Y])

        mu0_pred = mu.predict(self.df[[*self.X]+[self.T]].assign(**{self.T: 0}))
        mu1_pred = mu.predict(self.df[[*self.X]+[self.T]].assign(**{self.T: 1}))

        return mu1_pred.mean() - mu0_pred.mean()

    def Tlearner(self, learner : any):
        """
        """
        mu0 = copy.deepcopy(learner)
        mu1 = copy.deepcopy(learner)

        df0 = self.df[self.df[self.T] == 0]
        df1 = self.df[self.df[self.T] == 1]

        mu0.fit(X=df0[[*self.X]], y=df0[self.Y])
        mu1.fit(X=df1[[*self.X]], y=df1[self.Y])

        mu0_pred = mu0.predict(self.df[[*self.X]])
        mu1_pred = mu1.predict(self.df[[*self.X]])

        return mu1_pred.mean() - mu0_pred.mean()

    def Xlearner(self, learner : any):
        """
        """
        df0 = self.df[self.df[self.T] == 0]
        df1 = self.df[self.df[self.T] == 1]

        mu0 = copy.deepcopy(learner)
        mu1 = copy.deepcopy(learner)
        mu0.fit(X=df0[[*self.X]], y=df0[self.Y])
        mu1.fit(X=df1[[*self.X]], y=df1[self.Y])

        Delta0 = mu1.predict(df0[[*self.X]]) - df0[self.Y]
        Delta1 = df1[self.Y] - mu0.predict(df1[[*self.X]])

        tau0 = copy.deepcopy(learner)
        tau1 = copy.deepcopy(learner)
        tau0.fit(X=df0[[*self.X]], y=Delta0)
        tau1.fit(X=df1[[*self.X]], y=Delta1)

        tau0_pred = tau0.predict(self.df[[*self.X]])
        tau1_pred = tau1.predict(self.df[[*self.X]])

        e = self.e if self.e is not None else self.propensityScoreFunc()

        propensity_score = e.predict_proba(self.df[[*self.X]])

        v_func = np.vectorize(lambda e0, e1, t0, t1: e0*t0 + e1*t1)

        tau_list = v_func(propensity_score[:,0], propensity_score[:,1], tau0_pred, tau1_pred)

        return tau_list.mean()

    def Pstrat(self, num_strata : int):
        """
        """
        e = self.e if self.e is not None else self.propensityScoreFunc()
        propensity_score = e.predict_proba(self.df[[*self.X]])[:,1]
        propensity_score = pd.DataFrame({"e":propensity_score}).sort_values("e")

        df = self.df.reindex(propensity_score.index)
        strata = np.array_split(df, num_strata)
        tau_list = [len(Y)*(Y[Y[self.T] == 1][self.Y].mean() - Y[Y[self.T] == 0][self.Y].mean()) for Y in strata]

        return sum(tau_list)/len(df)

    def IPW(self):
        """
        """
        e = self.e if self.e is not None else self.propensityScoreFunc()
        propensity_score = e.predict_proba(self.df[[*self.X]])[:,1]

        v_func = np.vectorize(lambda e, t, y: (t/e - (1-t)/(1-e))*y)
        tau_list = v_func(propensity_score, self.df[self.T], self.df[self.Y])

        return tau_list.mean()

    def AIPW(self, learner : any):
        """
        """
        e = self.e if self.e is not None else self.propensityScoreFunc()
        propensity_score = e.predict_proba(self.df[[*self.X]])[:,1]

        mu0 = copy.deepcopy(learner)
        mu1 = copy.deepcopy(learner)

        df0 = self.df[self.df[self.T] == 0]
        df1 = self.df[self.df[self.T] == 1]

        mu0.fit(X=df0[[*self.X]], y=df0[self.Y])
        mu1.fit(X=df1[[*self.X]], y=df1[self.Y])

        mu0_pred = mu0.predict(self.df[[*self.X]])
        mu1_pred = mu1.predict(self.df[[*self.X]])

        v_func = np.vectorize(lambda e, t, y, mu0, mu1: (t*y - (t-e)*mu1)/e - ((1-t)*y - (t-e)*mu0)/(1-e))
        tau_list = v_func(propensity_score, self.df[self.T], self.df[self.Y], mu0_pred, mu1_pred)

        return tau_list.mean()

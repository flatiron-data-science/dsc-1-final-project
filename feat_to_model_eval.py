import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
from matplotlib import pyplot as plt

class feat_model_eval:
    def eval(self, target, df):
        '''Quickly evaluate a multi-var linreg model for a target and features in a dataframe'''
        
        predictors = df.drop(target, axis=1)
        f = "+".join(predictors.columns)
        f = target + "~" + f
        f
        model = ols(formula=f, data=df).fit()
        display(model.summary())

        y = df[target]
        X = df.drop(target, axis=1)
        a = np.mean(cross_val_score(linreg, X, y, cv=5, scoring='neg_mean_squared_error'))
        b = np.mean(cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error'))
        c = np.mean(cross_val_score(linreg, X, y, cv=20, scoring='neg_mean_squared_error'))
        avg = (a + b + c) / 3
        print(
            f'Average Negative Mean Squared Errors:\n'
            f' 5 Fold: {a}\n'
            f'10 Fold: {b}\n'
            f'20 Fold: {c}\n'
            f'Average of 5, 10, 20 Avg\'s: {avg}')



import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


def feat_to_model_kfold_eval(target: str, df, kvals: list, show_summary=False, price_logged=False, MAE=False):
            
    predictors = df.drop(target, axis=1)
    f = "+".join(predictors.columns)
    f = target + "~" + f
    model = ols(formula=f, data=df).fit()
    if show_summary:
        display(model.summary())
    
    y = df[target]
    X = df.drop(target, axis=1)
    
    total_errs = np.array([])
    k_vals = [5,10, 20]
    for k in k_vals:
        for train_index, test_index in KFold(n_splits=k).split(X):

                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                linreg.fit(X_train, y_train)
                y_hat = linreg.predict(X_test)
                errs = y_test - y_hat
                if price_logged:
                    errs = np.exp(y_test / y_hat)

                total_errs = np.concatenate((total_errs, errs))
    line = np.array(model.params)
    rmse = np.sqrt(np.mean(total_errs**2))
    if MAE:
        mae = abs(total_errs).mean()
        print(f'Line: {line}\n'
             f'RMSE: {rmse}\n'
             f'MAE: {mae}\n'
             f'THIS IS A STRINGGG')
        return line, rmse, mae
    else:
        print(f'Line: {line}\n'
             f'RMSE: {rmse}\n'
             f'THIS IS A ANOTHER STRINGGG')
        return line, rmse
            
def predict_from_line(line_vector, predict_vector):
    '''Predict vector element unit 1 to multiply the intercept'''
    print('This is probbbbbaaaabbbbllllyy the price for those variables')
    return sum(line_vector*predict_vector)
    

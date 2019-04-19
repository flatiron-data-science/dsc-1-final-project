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


# def feat_to_model_kfold_eval(target: str, predictors:list, df, kvals: list, show_summary=False, price_logged=False, MAE=False):
#     '''Returns:
#             -line vector of type array
#             -list of residuals of type array
#             -root mean squared error
#             -mean absolute error if activated'''
    
#     df = df[[target] + predictors]     # removing any features not included in model
    
#     predictors_df = df.drop(target, axis=1)    
    
#     f = "+".join(predictors_df.columns)
#     f = target + "~" + f
#     model = ols(formula=f, data=df).fit()
#     if show_summary:
#         display(model.summary())
    
#     y = df[target]
#     X = df.drop(target, axis=1)
    
#     total_errs = np.array([])
#     k_vals = [5,10, 20]
#     for k in k_vals:
#         for train_index, test_index in KFold(n_splits=k).split(X):

#                 X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#                 y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#                 linreg.fit(X_train, y_train)
#                 y_hat = linreg.predict(X_test)
#                 errs = y_test - y_hat
#                 if price_logged:
#                     errs = np.exp(y_test) - np.exp(y_hat)

#                 total_errs = np.concatenate((total_errs, errs))
#     line = np.array(model.params)
#     rmse = np.sqrt(np.mean(total_errs**2))

#     if MAE:
#         mae = abs(total_errs).mean()
#         print(f'Line: {line}\n'
#              f'RMSE: {rmse}\n'
#              f'MAE: {mae}\n'
#              f'MAE DISPLAYED')
#         return line, total_errs, rmse, mae
#     else:
#         print(f'Line: {line}\n'
#              f'RMSE: {rmse}\n'
#              f'Please pass in MAE=True if you\'d like to see MAE')
#         return line, total_errs, rmse
            
# def predict_from_line(line_vector, predict_vector):
#     '''Predict vector element unit 1 to multiply the intercept'''
#     print('This is probbbbbaaaabbbbllllyy the price for those variables')
#     return sum(line_vector*predict_vector)
    

def feat_to_model_kfold_eval(target: str, predictors:list, df, kvals: list, show_summary=False, price_logged=False, MAE=False, return_items=True):
    '''Returns:
            -line vector of type array
            -list of residuals of type array
            -root mean squared error
            -mean absolute error if activated
            -dict of pvalues'''
    
    df = df[[target] + predictors]     # removing any features not included in model
    
    predictors_df = df.drop(target, axis=1)    
    
    f = "+".join(predictors_df.columns)    # This could be simplified to predictors instead predictors_df.columns but its the end of the day skip for now
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
                    errs = np.exp(y_test) - np.exp(y_hat)

                total_errs = np.concatenate((total_errs, errs))
                
                
    linear_coefs = np.array(model.params)
    
    line = {'intercept': linear_coefs[0]}
    for col, coef in zip(predictors, linear_coefs[1:]):
        line[col] = coef
    
    pvals = model.pvalues.to_dict()
    
    rmse = np.sqrt(np.mean(total_errs**2))
    
    if MAE:
        mae = abs(total_errs).mean()
        print(
             f'RMSE: {rmse}\n'
             f'MAE: {mae}\n')
        if not return_items:
            return
        return line, total_errs, rmse, mae, pvals
    else:
        print(
             f'RMSE: {rmse}\n')
        if not return_items:
            return
        return line, total_errs, rmse, pvals
            
def predict_from_line(line_vector, predict_vector):
    '''Predict vector element unit 1 to multiply the intercept'''
    print('This is probbbbbaaaabbbbllllyy the price for those variables')
    return sum(line_vector*predict_vector)
    
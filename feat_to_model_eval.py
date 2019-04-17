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

# class feat_model_eval:
#     def eval(self, target, df):
#         '''Quickly evaluate a multi-var linreg model for a target and features in a dataframe'''
        
#         predictors = df.drop(target, axis=1)
#         f = "+".join(predictors.columns)
#         f = target + "~" + f
#         model = ols(formula=f, data=df).fit()
#         display(model.summary())

#         y = df[target]
#         X = df.drop(target, axis=1)
        
#         cv5_errs  = cross_val_score(linreg, X, y, cv=5,  scoring='neg_mean_squared_error')
#         cv10_errs = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')
#         cv20_errs = cross_val_score(linreg, X, y, cv=20, scoring='neg_mean_squared_error')
        
        
#         cv5_avg  = np.mean(cv5_errs)
#         cv10_avg = np.mean(cv10_errs)
#         cv20_avg = np.mean(cv20_errs)
        
#         avg_errs = (cv5_avg + cv10_avg + cv20_avg) / 3
        
        
        
#         print(
#             f'Average Negative Mean Squared Errors:\n'
#             f' 5 Fold: {cv5_avg}\n'
#             f'10 Fold: {cv10_avg}\n'
#             f'20 Fold: {cv20_avg}\n'
#             f'Average of 5, 10, 20 Avg\'s: {avg_errs}'
# #             f'Average Dollars Risked')


# def dummy_fxn():
#     return 5

# def model_eval(target, dataframe, price_log=False, metric='RMSE'):
#     if price_log:
#         print('Not implemented yet')
#     else:
#         predictors = df.drop(target, axis=1)
#         f = "+".join(predictors.columns)
#         f = target + "~" + f
#         model = ols(formula=f, data=df).fit()
#         display(model.summary())

#         y = df[target]
#         X = df.drop(target, axis=1)
        
#         if metric == 'RMSE':
#             metric = 'neg_mean_squared_error'
#             cv5_errs  = cross_val_score(linreg, X, y, cv=5,  scoring= metric)
#             cv10_errs = cross_val_score(linreg, X, y, cv=10, scoring= metric)
#             cv20_errs = cross_val_score(linreg, X, y, cv=20, scoring= metric)
            
#         else: metric == 'MAE':
#             print('Not implemented yet')
#             metric = 'neg_median_absolute_error'
#             cv5_errs  = cross_val_score(linreg, X, y, cv=5,  scoring= metric)
#             cv10_errs = cross_val_score(linreg, X, y, cv=10, scoring= metric)
#             cv20_errs = cross_val_score(linreg, X, y, cv=20, scoring= metric)
   
        
        
#         cv5_avg  = np.mean(cv5_errs)
#         cv10_avg = np.mean(cv10_errs)
#         cv20_avg = np.mean(cv20_errs)
        
#         avg_errs = (cv5_avg + cv10_avg + cv20_avg) / 3
        
        
        
#         print(
#             f'Average Negative Mean Squared Errors:\n'
#             f' 5 Fold: {cv5_avg}\n'
#             f'10 Fold: {cv10_avg}\n'
#             f'20 Fold: {cv20_avg}\n'
#             f'Average of 5, 10, 20 Avg\'s: {avg_errs}\n'
#             f'Average Dollars Risked: {np.sqrt(abs(avg_errs))}')

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
            
def predict(line_vector, predict_vector):
    '''Predict vector element unit 1 to multiply the intercept'''
    print('This is probbbbbaaaabbbbllllyy the price for those variables')
    return sum(line_vector*predict_vector)
    
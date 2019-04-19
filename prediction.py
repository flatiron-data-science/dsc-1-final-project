import pandas as pd
import numpy as np

def make_a_house(**kwargs):
    '''Takes in a dataframe as a skeleton for generating logs'''
    mock_home_df = pd.DataFrame([])
    for col, value in kwargs.items():
        mock_home_df.loc['test_home', str(col)] = value
        if str(col) in ['price', 'sqft_living', 'sqft_living15', 'grade']:
            mock_home_df.loc['test_home', f'log_{col}'] = np.log(value)
    display(mock_home_df)
    return mock_home_df

def predict_home_price(linear_coefs: dict, **kwargs):
    sum_price = 0
    house = make_a_house(**kwargs)
    for col, coef in linear_coefs.items():
        if str(col) == 'intercept':
            sum_price += coef
        elif str(col) not in house:
            continue
        else:
            sum_price += coef * house.loc['test_home', str(col)]
    return sum_price
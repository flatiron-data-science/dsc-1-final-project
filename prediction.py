import pandas as pd
import numpy as np

def make_a_house(**kwargs):
    mock_home_df = pd.DataFrame([])
    for col, value in kwargs.items():
        mock_home_df.loc['test_home', str(col)] = value
        if str(col) in ['price', 'sqft_living', 'sqft_living15', 'grade']:
            mock_home_df.loc['test_home', f'log_{col}'] = np.log(value)
    display(mock_home_df)
    return mock_home_df

def predict_home_price(linear_coefs: dict, price_logged=False, **kwargs):
    sum_price = 0
    house = make_a_house(**kwargs)
    for col, coef in linear_coefs.items():
        if str(col) == 'intercept':
            sum_price += coef
        elif str(col) not in house:
            continue
        else:
            sum_price += coef * house.loc['test_home', str(col)]
    if price_logged:
        return np.exp(sum_price)
    return sum_price
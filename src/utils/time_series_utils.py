from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import numpy as np


'''The most commonly used is the Augmented Dickey Fuller (ADF) test, where the null hypothesis is the time series 
possesses a unit root and is non-stationary. So, if the P-Value in ADH test is less than the significance level (0.05), 
you reject the null hypothesis; hence, the data is stationary'''
def adf_test(data: np.ndarray):
    result = adfuller(data, autolag='AIC')
    d = { 'ADF Statistic': result[0], 'p-value': "%.2f" % result[1], 'crit_values': result[4].items()}
    df = pd.DataFrame(d)
    return df


'''The KPSS test, on the other hand, is used to test for trend stationarity. The null hypothesis and the P-Value 
interpretation is just the opposite of ADH test.'''
def kpss_test(data: np.ndarray):
    result = kpss(data, regression='c')
    d = {'KPSS Statistic': result[0], 'p-value': "%.2f" % result[1], 'crit_values': result[3].items()}
    df = pd.DataFrame(d)
    return df


def calc_delta(data: pd.DataFrame, col: str):
    df = data[f'{col}']
    n = 0
    temp = 0
    for idx, row in data.iterrows():
        n = idx



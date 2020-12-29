from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
df = pd.read_csv(r'data\sales_sample.csv', header=0, parse_dates=['date'])

diff_df = df.diff().fillna(0.0)
# diff_df = diff_df.fillna(0.0)
print('done')


def calc_delta(data: pd.DataFrame, col: str):
    df = data[f'{col}']
    n = 0
    temp = 0
    for idx, row in data.iterrows():
        n = idx
        temp = idx - 1
        row['delta'] = row['s']

calc_delta(df, 'sales')



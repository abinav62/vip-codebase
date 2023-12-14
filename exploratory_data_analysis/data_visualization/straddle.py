"""
This file contains data visualizations of short straddle data.
"""


import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

straddles = pd.DataFrame(columns=['date', 'vanilla', 'stoploss_10', 'stoploss_25', 'stoploss_30', 'stoploss_40', 'stoploss_50', 'stoploss_75', 'stoploss_100'])

for straddle in ['vanilla', 'stoploss_10', 'stoploss_25', 'stoploss_30', 'stoploss_40', 'stoploss_50', 'stoploss_75', 'stoploss_100']:
    df = pd.read_csv(f'../../data/{straddle}_straddle_cleaned.csv')
    df['date'] = pd.to_datetime(df['date'])
    daily_pl = df.groupby(df['date'].dt.date).last()
    daily_pl = daily_pl.set_index('date').resample('1D').last()
    daily_pl['pl_growth'] = daily_pl['pl_dollars'].cumsum()
    daily_pl.drop(['straddle_entry', 'straddle_exit', 'pl_points', 'pl_dollars'], axis=1, inplace=True)
    # Forwards fill the so that all the dates are present
    daily_pl['pl_growth'].fillna(method='ffill', inplace=True)
    # Rename the column to the straddle name
    daily_pl.rename(columns={'pl_growth': straddle}, inplace=True)
    # Merge straddles and daily_pl
    straddles = straddles.merge(daily_pl, how='outer', left_index=True, right_index=True, suffixes=("_x", None))
    straddles.drop([f'{straddle}_x'], axis=1, inplace=True)
    straddles[straddle].fillna(method='ffill', inplace=True)
straddles.drop(['date'], axis=1, inplace=True)
straddles.plot()
plt.title(f'Short straddle variations')
plt.show()
# print(straddles)

import pandas as pd

# for straddle in ['vanilla', 'stoploss_10', 'stoploss_25', 'stoploss_30', 'stoploss_40', 'stoploss_50', 'stoploss_75', 'stoploss_100']:
for straddle in ['vanilla']:
    df = pd.read_csv(f'../data/straddle/{straddle}_straddle_updated.csv')
    null_dates = df['date'][df['pl_dollars'].isna()].tolist()
    for i, date in enumerate(null_dates):
        null_dates[i] = pd.to_datetime(date).date()
    df['date'] = pd.to_datetime(df['date'])
    # Get rows for which the date is not in null_dates
    df = df[~df['date'].dt.date.isin(null_dates)]
    df.to_csv(f'../data/straddle/{straddle}_straddle_updated_cleaned.csv', index=False, header=True)

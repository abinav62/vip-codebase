import pandas as pd

df = pd.read_csv('../data/straddle/vanilla_straddle_updated_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])
for index in ['SPX', 'NDAQ']:
    index_df = pd.read_csv(f'../data/indices/{index}_updated_cleaned.csv')
    index_df['date'] = pd.to_datetime(index_df['date'])
    df = df.merge(index_df, how='left', suffixes=(None, f"_{index.lower()}"), left_on='date', right_on='date')
for stock in ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'META', 'NVDA']:
    stock_df = pd.read_csv(f'../data/stocks/{stock}_updated_cleaned.csv')
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    df = df.merge(stock_df, how='left', suffixes=(None, f"_{stock.lower()}"), left_on='date', right_on='date')

# Create a column exit that says yes if the next row of pl_points is less than the current row
df['exit'] = df['pl_points'].shift(-1) < df['pl_points']

df.to_csv('../data/master_with_15_updated.csv', index=False, header=True)
df = df[df['date'].dt.hour < 15]
df.to_csv('../data/master_without_15_updated.csv', index=False, header=True)

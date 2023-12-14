"""
This file contains data visualizations of DJI data and its indicators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
from io import StringIO

import pandas_ta as ta
from datetime import timedelta

ACCESS_KEY = "AKIASX4WXL2YVHGAP44K"
SECRET_ACCESS_KEY = '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'


def list_s3_objects(bucket_name):
    """
    Get a list of objects in an S3 bucket.
    """
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY, region_name='us-east-2')
    files = []

    # List objects in the specified bucket and prefix (folder)
    # objects = s3.list_objects_v2(Bucket='vip-mlfs-option-data')
    objects = s3.list_objects_v2(Bucket=bucket_name)

    if 'Contents' in objects:
        for obj in objects['Contents']:
            files.append(obj['Key'])
    print(files)


def get_s3_file(bucket_name, file_name):
    """
    Get a single S3 file and return as dataframe
    """
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY, region_name='us-east-2')
    s3_bucket = 'vip-mlfs-index-data'
    s3_key = 'SPX.csv'

    # Download CSV file from S3
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    csv_bytes = response['Body'].read().decode('utf-8')

    # Convert CSV data to a DataFrame
    df = pd.read_csv(StringIO(csv_bytes))
    return df


def clean_data(df, options=False):
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Sort the dataframe by the datetime column
    df.sort_values('datetime', inplace=True)
    df.set_index('datetime', inplace=True, drop=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df


def plot_data(df, title='SPX', xlabel='Date', ylabel='Price'):
    """
    Plot SPX data, with optional title and labels.
    """
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def plot_selected(df, columns, start_index, end_index):
    """
    Plot the desired columns over index values in the given range.
    """
    plot_data(df.loc[start_index:end_index, columns])


# Convert the dataframe with timestamps into hourly data keeping the first value in open column, last value in close column, max value in high column and min value in low column
def convert_to_hourly(df):
    df['date'] = df.index
    df = df[(df['date'].dt.hour >= 10) & (df['date'].dt.hour < 14)]
    df['date'] = df['date'].dt.round('H') + timedelta(hours=1)
    df = df.groupby('date').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.reset_index(inplace=True)
    return df

def get_indicators(df):
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['rsi_7'] = ta.rsi(df['close'], length=7)
    df['sma_5'] = ta.sma(df['close'], length=5)
    df['sma_10'] = ta.sma(df['close'], length=10)
    df['ema_5'] = ta.ema(df['close'], length=5)
    df['ema_10'] = ta.ema(df['close'], length=10)
    df['wma_5'] = ta.wma(df['close'], length=5)
    df['wma_10'] = ta.wma(df['close'], length=10)
    df['supertrend'] = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3)['SUPERT_7_3.0']
    return df


# Write a code to complete entire exploratory data analysis on stock data
def main():
    # list_s3_objects('vip-mlfs-index-data')
    df = get_s3_file('vip-mlfs-stock-data', 'AAPL.csv')
    df = clean_data(df)
    cleaned_df = get_indicators(convert_to_hourly(df))
    cleaned_df.to_csv('../../data/stocks/AAPL_cleaned.csv', index=False, header=True)



    # plot_data(df)
    # plot_selected(df, ['close'], '2021-10-01', '2021-10-31')


main()

"""
This module contains functions for backtesting the intraday straddle strategy. The logic
for this strategy is a vanilla short straddle entering at 9:45am and exiting at 3:15pm.
There is a stoploss but no trailing stoploss set for this strategy. The strategy runs on
the SPX weekly options and runs only on the expiration date of the options, as it follows
the European options model. The strategy is backtested on the SPX from 2021 to 2023.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
from io import StringIO
import pytz

from datetime import datetime, timedelta, time, date

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


def clean_data(df, options=False, set_index=False):
    if options:

        df['t'] = pd.to_datetime(df['t'], unit='ms')

        # Sort the dataframe by the datetime column
        df.sort_values('t', inplace=True)
        us_eastern = pytz.timezone('US/Eastern')
        df['t'].dt.tz_localize(pytz.utc).dt.tz_convert(us_eastern)
        if set_index:
            df.set_index('t', inplace=True, drop=True)
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Sort the dataframe by the datetime column
        df.sort_values('datetime', inplace=True)
        if set_index:
            df.set_index('datetime', inplace=True, drop=True)

    return df


def adjust_timezone(df):
    if df['t'][0].hour == 13:
        df['t'] = df['t'] - timedelta(hours=4)
    elif df['t'][0].hour == 14:
        df['t'] = df['t'] - timedelta(hours=5)
    return df


def simulate_intraday_straddle(ce, pe, pl_data):
    """
    Simulate the intraday straddle strategy on the given options data. The strategy
    is a vanilla short straddle entering at 9:45am and exiting at 3:15pm. There is no
    stoploss or trailing stoploss set for this strategy. The strategy runs on the SPX
    weekly options and runs only on the expiration date of the options, as it follows
    the European options model. The strategy is backtested on the SPX from 2021 to 2023.
    """
    try:
        # Get the price of the options at 10am
        ce_entry = ce['c'][ce['t'].dt.time == time(10)].iloc[0]
        pe_entry = pe['c'][pe['t'].dt.time == time(10)].iloc[0]

        ce_sl = 1.40 * ce_entry
        pe_sl = 1.40 * pe_entry

        for i in range(11, 16):
            if max(ce['h'][(ce['t'].dt.time >= time(10)) & (ce['t'].dt.time <= time(i))]) >= ce_sl:
                ce_close = ce_sl
            else:
                ce_close = ce['c'][ce['t'].dt.time == time(i)].iloc[0]

            if max(pe['h'][(pe['t'].dt.time >= time(10)) & (pe['t'].dt.time <= time(i))]) >= pe_sl:
                pe_close = pe_sl
            else:
                pe_close = pe['c'][pe['t'].dt.time == time(i)].iloc[0]

            pl_data = pd.concat([pl_data, pd.DataFrame([{
                'date': ce['t'][ce['t'].dt.time == time(i)].iloc[0],
                'straddle_entry': ce_entry + pe_entry,
                'straddle_exit': ce_close + pe_close,
                'pl_points': round(ce_entry - ce_close + pe_entry - pe_close, 2),
                'pl_dollars': round((ce_entry - ce_close + pe_entry - pe_close) * 100, 2),
            }])], ignore_index=True)

        return pl_data

    except Exception as e:
        return pd.concat([pl_data, pd.DataFrame([{
            'date': ce['t'][0],
            'straddle_entry': 0,
            'straddle_exit': 0,
            'pl_points': 0
        }])], ignore_index=True)


def main():
    df = clean_data(get_s3_file('vip-mlfs-index-data', 'SPX.csv'))
    df = df[(df['datetime'].dt.time == time(10)) & (df['datetime'].dt.date > date(2021, 10, 22))]
    pl_data = pd.DataFrame(columns=['date', 'straddle_entry', 'straddle_exit', 'pl_points'])
    for i, row in df.iterrows():
        # Remove this condition if running for a timeframe on or after October 2022
        if row['datetime'].weekday() in [1, 3] and row['datetime'] < datetime(2022, 10, 1, 0, 0, 0):
            continue

        expiry_date = row['datetime'].strftime("%Y-%m-%d")
        expiry = row['datetime'].strftime("%y%m%d")
        close = int(int(row['close']) / 5) * 5
        try:
            ce = adjust_timezone(
                clean_data(
                    get_s3_file(
                        'vip-mlfs-option-data',
                        f'expiry={expiry}/strike={close}/SPXW{expiry}C0{close}000.csv'
                    ),
                    options=True
                )
            )
            pe = adjust_timezone(
                clean_data(
                    get_s3_file(
                        'vip-mlfs-option-data',
                        f'expiry={expiry}/strike={close}/SPXW{expiry}P0{close}000.csv'
                    ),
                    options=True
                )
            )
            pl_data = simulate_intraday_straddle(ce, pe, pl_data)
        except Exception as e:
            print(e)
            continue

    # Export pl_data to csv
    pl_data.to_csv('../data/stoploss_40_straddle.csv', index=False, header=True)

main()
import os
import time
import datetime
import pandas as pd
import pytrademl.utilities.dataframe_utilities as dataframe_utilities
import pytrademl.utilities.ticker_utilities as ticker_utilities
from pytrademl.utilities.key_utilities import load_key
from alpha_vantage.timeseries import TimeSeries
from pathlib import Path

def get_data_from_alphaVantage(key, reload=False, ticker_file="ETFTickers", symbol_market="TSX", output_size="full"):

    root_dir = Path().resolve().parent

    if reload:
        tickers = ticker_utilities.obtain_tickers(ticker_file)
    else:
        tickers = ticker_utilities.import_object((root_dir / ticker_file).with_suffix('.pickle'))

    folder_name = root_dir / 'dataframes' / ticker_file.split("Tickers")[0]
    folder_name.mkdir(parents=True, exist_ok=True)

    # if ticker_file == "./tickers/sp500tickers.pickle":
    #     folder_name = 'dataframes/sp500'
    # elif ticker_file == "./tickers/ETFTickers.pickle":
    #     folder_name = 'dataframes/ETF'
    # elif ticker_file == "./tickers/TSXTickers.pickle":
    #     folder_name = "dataframes/TSX"
    # else:
    #     return 1
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)

    ts = TimeSeries(key=key, output_format='pandas')

    # If today is a weekend, go back to the friday
    today = datetime.datetime.today()
    weekday = today.weekday()
    if weekday == 5:
        today = datetime.datetime.today() - datetime.timedelta(days=1)
    elif weekday == 6:
        today = datetime.datetime.today() - datetime.timedelta(days=2)
    else:
        pass
    today = today.strftime("%Y-%m-%d")
    totalAPICalls = 0
    maxAPICalls = 500 # Max of 500 API requests per day on free license
    api_calls_per_minute = 5
    sleep_delay = int(60 / api_calls_per_minute) + 2
    for ticker in tickers:
        # Check if a dataframe for the ticker already exists
        ticker_file = (folder_name / ticker[1]).with_suffix('.csv')
        if Path(ticker_file).is_file():
            df = pd.read_csv(ticker_file)
            df.set_index('date', inplace=True)
            if today in df.index: # Check if ticker information is up to date
                print(ticker[1], "is up to date.")
                continue
        if totalAPICalls < maxAPICalls:
            if symbol_market == "TSX":
                tickername = 'TSX:'+ticker[1]
                try:
                    data, _ = ts.get_daily_adjusted(tickername, outputsize=output_size)  # full or compact
                    totalAPICalls = totalAPICalls + 1
                    print('Retrieved data for:',
                          tickername, '(', ticker[0], ')')
                    got_flag = True
                except Exception as e:
                    print("Error retrieving data for", tickername+":", e)
                    if output_size == "full":
                        try:
                            tickername = 'TSX:'+ticker[1]
                            data, _ = ts.get_daily_adjusted(
                                tickername, outputsize='compact')
                            totalAPICalls = totalAPICalls + 1
                            print('Retrieved compact data for:',
                                  tickername, '(', ticker[0], ')')
                            got_flag = True
                        except Exception as e:
                            print("Error retrieving data for",
                                  tickername+":", e)
                            got_flag = False
                            time.sleep(sleep_delay)
                    else:
                        time.sleep(sleep_delay)
                        got_flag = False
            else:
                try:
                    data, _ = ts.get_daily_adjusted(
                        ticker[1], outputsize=output_size)
                    totalAPICalls = totalAPICalls + 1
                    print('Retrieved data for:',
                          ticker[1], '(', ticker[0], ')')
                    got_flag = True
                except Exception as e:
                    print("Error retrieving data for", ticker[1]+":", e)
                    time.sleep(sleep_delay)
                    got_flag = False

            if got_flag:
                if Path(ticker_file).is_file():
                # if os.path.exists('{}/{}.csv'.format(folder_name, ticker[1])):
                    data.to_csv('temp.csv')
                    df_new = pd.read_csv('temp.csv', parse_dates=True, index_col=0)
                    df_old = pd.read_csv(ticker_file, parse_dates=True, index_col=0)
                    # Merge and drop exact duplicates
                    df = pd.concat([df_new, df_old]).drop_duplicates()
                    # Drops duplicates with updated values, keeping the most recent data
                    df = df.loc[~df.index.duplicated(keep='first')]
                else:
                    data.to_csv(ticker_file)
                    df = pd.read_csv(ticker_file, parse_dates=True, index_col=0)
                df.sort_index(axis=0, inplace=True)
                df = cleanup_zeros(df)
                df.to_csv(ticker_file)
                time.sleep(sleep_delay)
        else:
            print("Already used 500 API requests. Have to wait 24 hours now.")

def cleanup_zeros(df):
    previous_row = {}
    for _, row in df.iterrows():
        if row['1. open'] == 0:
            try:
                row['1. open'] = previous_row['1. open']
                row['2. high'] = previous_row['2. high']
                row['3. low'] = previous_row['3. low']
                row['4. close'] = previous_row['4. close']
                row['5. adjusted close'] = previous_row['5. adjusted close']
                row['6. volume'] = previous_row['6. volume']
            except Exception as err:
                print(err)
        previous_row = row
    return df


if __name__ == "__main__":
    key = load_key()
    print(key)
    get_data_from_alphaVantage(key=key, reload=True)

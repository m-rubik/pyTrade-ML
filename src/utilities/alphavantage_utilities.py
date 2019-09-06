
import src.utilities.dataframe_utilities as dataframe_utilities
import src.utilities.ticker_utilities as ticker_utilities
import os
import time
import datetime
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

def get_data_from_alphaVantage(reload=False,tickerFile="./tickers/ETFTickers.pickle", symbol_market="TSX", output_size="full"):
    
    if reload:
        tickers = ticker_utilities.obtain_tickers(tickerFile)
    else:
        tickers = ticker_utilities.import_tickers(tickerFile)

    if tickerFile == "./tickers/sp500tickers.pickle":
        folderName = 'sp500_dfs'
    elif tickerFile == "./tickers/ETFTickers.pickle":
        folderName = 'ETF_dfs'
    else:
        return 1

    if not os.path.exists(folderName):
        os.makedirs(folderName)

    ts = TimeSeries(key='7M23W012GUSSMR90',output_format='pandas',retries=0)

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    totalAPICalls = 0
    for ticker in tickers:
        # Check if a dataframe for the ticker already exists
        if os.path.exists('{}/{}.csv'.format(folderName,ticker[1])):
            df = pd.read_csv('{}/{}.csv'.format(folderName,ticker[1]))
            df.set_index('date', inplace=True)
            # Check if ticker information is up to date
            if today in df.index:
                print(ticker[1],"is up to date.")
                continue
        if totalAPICalls < 500: # Max of 500 API requests per day on free license 
            if symbol_market == "TSX":
                try:
                    tickername = 'TSX:'+ticker[1]
                    data, meta_data = ts.get_daily_adjusted(tickername,outputsize=output_size) # full or compact
                    totalAPICalls = totalAPICalls + 1
                    print('Retrieved data for:',tickername,'(',ticker[0],')')
                    got_flag = True
                except Exception as e:
                    print("Error retrieving data for",tickername+":",e)
                    if output_size == "full":
                        try:
                            tickername = 'TSX:'+ticker[1]
                            data, meta_data = ts.get_daily_adjusted(tickername,outputsize='compact')
                            totalAPICalls = totalAPICalls + 1
                            print('Retrieved compact data for:',tickername,'(',ticker[0],')')
                            got_flag = True
                        except Exception as e:
                            print("Error retrieving data for",tickername+":",e)
                            got_flag = False
                            time.sleep(13) # Max 5 API requests per minute.
                    else:
                        time.sleep(13) # Max 5 API requests per minute.
                        got_flag = False
            else:
                try:
                    data, meta_data = ts.get_daily_adjusted(ticker[1],outputsize=output_size)
                    totalAPICalls = totalAPICalls + 1
                    print('Retrieved data for:',ticker[1],'(',ticker[0],')')
                    got_flag = True
                except Exception as e:
                    print("Error retrieving data for",ticker[1]+":",e)
                    time.sleep(13) # Max 5 API requests per minute.
                    got_flag = False

            if got_flag:
                if os.path.exists('{}/{}.csv'.format(folderName,ticker[1])):
                    data.to_csv('temp.csv')
                    df_new = pd.read_csv('temp.csv', parse_dates=True, index_col=0)
                    df_old = pd.read_csv('{}/{}.csv'.format(folderName,ticker[1]), parse_dates=True, index_col=0)
                    df = pd.concat([df_new,df_old]).drop_duplicates() # Merge and drop exact duplicates
                    # df = df.loc[~df.index.duplicated(keep='first')] # Drops duplicates with updated values, keeping the most recent data
                else:
                    data.to_csv('{}/{}.csv'.format(folderName,ticker[1]))
                    df = pd.read_csv('{}/{}.csv'.format(folderName,ticker[1]), parse_dates=True, index_col=0)
                df.sort_index(axis = 0, inplace=True)
                df = cleanup_zeros(df)
                df.to_csv('{}/{}.csv'.format(folderName,ticker[1]))
                time.sleep(13) # Max 5 API requests per minute.
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
    get_data_from_alphaVantage()
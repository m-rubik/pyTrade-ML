import os
import ta
import pandas as pd
import numpy as np
import src.utilities.ticker_utilities as ticker_utilities


def import_dataframe(ticker, ticker_file="./tickers/sp500tickers.pickle", starting_date=None):
    if ticker_file == "./tickers/sp500tickers.pickle":
        datafolder = './sp500_dfs/'
    elif ticker_file == "./tickers/ETFTickers.pickle":
        datafolder = './ETF_dfs/'
    else:
        print("Unrecognized ticker file:", ticker_file)
    tickerData = datafolder+ticker+'.csv'
    try:
        df = pd.read_csv(tickerData, parse_dates=True, index_col=0)
        df = df.truncate(before=starting_date)
    except Exception as df:
        print(df)
    finally:
        return df

def export_dataframe(df, name):
    pass

def generate_adjclose_df(ticker_file="./tickers/sp500tickers.pickle"):
    """
    Generate a dataframe whose columns are:
    date, symbol_1, symbol_2, symbol_3, etc...
    The values in the symbol columns are the adjusted close values of that date.
    """

    print("Generating the master adjusted close dataframe...")
    tickers = ticker_utilities.import_tickers(ticker_file)
    main_df = pd.DataFrame()
    for _, ticker in enumerate(tickers):
        if ticker_file == "./tickers/sp500tickers.pickle":
            datafile = './sp500_dfs/{}.csv'.format(ticker[1])
        elif ticker_file == "./tickers/ETFTickers.pickle":
            datafile = './ETF_dfs/{}.csv'.format(ticker[1])
        else:
            print("Unrecognized ticker file:", ticker_file)

        if os.path.isfile(datafile):
            print("Processing", ticker[1])
            df = pd.read_csv(datafile, engine='python')
            df.set_index('date', inplace=True)

            df.rename(columns={'5. adjusted close': ticker[1]}, inplace=True)
            df.drop(['1. open', '2. high', '3. low', '4. close', '6. volume', '7. dividend amount', '8. split coefficient'], 1, inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

    main_df = main_df.reindex(sorted(main_df.columns), axis=1)
    print("Master adjusted close dataframe generated.")

    if ticker_file == "./tickers/sp500tickers.pickle":
        main_df.to_csv('sp500_bysymbol_joined_closes.csv')
    elif ticker_file == "./tickers/ETFTickers.pickle":
        main_df.to_csv('ETF_bysymbol_joined_closes.csv')
    else:
        print("Unrecognized ticker file:", ticker_file)

    return main_df

def add_indicators(df):
    
    print("Adding technical indicators")
    # df['100ma'] = df['5. adjusted close'].rolling(window=100).mean()
    # df['9ema'] = df['5. adjusted close'].ewm(span=9, adjust=False).mean()
    # df['12ema'] = df['5. adjusted close'].ewm(span=12, adjust=False).mean()
    # df['26ema'] = df['5. adjusted close'].ewm(span=26, adjust=False).mean()
    # df['macd'] = df['12ema'] - df['26ema']
    # previousRow = 0
    # for index, row in df.iterrows():
    #     df.loc[index, 'macd_relChange'] = abs(row.macd-previousRow)*100
    #     previousRow = row.macd

    # # Add bollinger band high indicator filling Nans values
    # df['bb_high_indicator'] = ta.bollinger_hband_indicator(df["close"], n=20, ndev=2, fillna=True)

    # # Add bollinger band low indicator filling Nans values
    # df['bb_low_indicator'] = ta.bollinger_lband_indicator(df["close"], n=20, ndev=2, fillna=True)

    # # Add bollinger band high
    # df['bb_high'] = ta.bollinger_hband(df["close"], n=20, ndev=2, fillna=True)

    # # Add bolling band low
    # df['bb_low'] = ta.bollinger_lband(df["close"], n=20, ndev=2, fillna=True)

    df = ta.add_all_ta_features(df, "1. open", "2. high", "3. low", "4. close", "6. volume", fillna=True)

    df.to_csv('test.csv')

    
    # # Get rid of infinite changes
    # df = df.replace([np.inf, -np.inf], np.nan)

    # # Replace NaN with 0
    # df.fillna(0, inplace=True)

    return df


if __name__ == "__main__":
    # df = generate_adjclose_df("./tickers/ETFTickers.pickle")
    df = import_dataframe("XIC","./tickers/ETFTickers.pickle")

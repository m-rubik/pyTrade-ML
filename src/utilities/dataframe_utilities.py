import pandas as pd

def import_dataframe(ticker, tickerFile="./tickers/sp500tickers.pickle", startingDate=None):
    if tickerFile == "./tickers/sp500tickers.pickle":
        datafolder = './sp500_dfs/'
    elif tickerFile == "./tickers/ETFTickers.pickle":
        datafolder = './ETF_dfs/'
    else:
        print("Unrecognized ticker file:",tickerFile)
    tickerData = datafolder+ticker+'.csv'
    try:
        df = pd.read_csv(tickerData, parse_dates=True, index_col=0)
        df = df.truncate(before=startingDate)
    except Exception as df:
        print(df)
    finally:
        return df

def export_dataframe(df, name):
    pass


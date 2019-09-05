import bs4 as bs
import datetime as dt
import os
import csv
import pandas as pd
import pandas_datareader.data as web
from pprint import pprint
from alpha_vantage.timeseries import TimeSeries
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates

from pandas.plotting import register_matplotlib_converters

import time
import pickle
import requests

import src.utilities.dataframe_utilities as dataframe_utilities
import src.utilities.ticker_utilities as ticker_utilities

from datetime import datetime

import accounts

register_matplotlib_converters()

def get_data_from_alphaVantage(reload=False,tickerFile="./tickers/sp500tickers.pickle"):
    
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

    totalAPICalls = 0
    for ticker in tickers:
        if not os.path.exists('{}/{}.csv'.format(folderName,ticker[1])):
            if totalAPICalls < 500:
                # Get dataframe object
                try:
                    # data, meta_data = ts.get_daily_adjusted(ticker[1],outputsize='full')
                    data, meta_data = ts.get_daily_adjusted(ticker[1],outputsize='compact')
                    totalAPICalls = totalAPICalls + 1
                    print('Retrieved data for:',ticker[1],'(',ticker[0],')')
                    # pprint(data.head(1))
                    data.to_csv('{}/{}.csv'.format(folderName,ticker[1]))
                except Exception as e:
                    print("Error retrieving data for",ticker[1]+":",e)
                    time.sleep(13) # Max 5 API requests per minute.
                    try:
                        tickername = 'TSX:'+ticker[1]
                        # data, meta_data = ts.get_daily_adjusted(tickername,outputsize='full')
                        data, meta_data = ts.get_daily_adjusted(tickername,outputsize='compact')
                        totalAPICalls = totalAPICalls + 1
                        print('Retrieved data for:',tickername,'(',ticker[0],')')
                        # pprint(data.head(1))
                        data.to_csv('{}/{}.csv'.format(folderName,ticker[1]))
                    except Exception as e:
                        print("Error retrieving data for",tickername+":",e)

                # Max of 500 API requests per day
                time.sleep(13) # Max 5 API requests per minute.
            else:
                print("Already used 500 API requests. Have to wait 24Hours now.")
        else:
            print('Already have {}'.format(ticker[1]))
    
def compile_data(tickerFile="./tickers/sp500tickers.pickle"):
    with open(tickerFile, "rb") as f:
        tickers = pickle.load(f)
    
    main_df = pd.DataFrame()

    for count,ticker in enumerate(tickers):
        if tickerFile == "./tickers/sp500tickers.pickle":
            datafile = './sp500_dfs/{}.csv'.format(ticker[1])
        elif tickerFile == "./tickers/ETFTickers.pickle":
            datafile = './ETF_dfs/{}.csv'.format(ticker[1])
        else:
            print("What pickle file?",tickerFile)

        if os.path.isfile(datafile):
            df = pd.read_csv(datafile,engine='python')
            df.set_index('date', inplace=True)

            df.rename(columns = {'5. adjusted close': ticker[1]}, inplace=True)
            df.drop(['1. open','2. high','3. low','4. close','6. volume','7. dividend amount','8. split coefficient'],1,inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            if count % 10 == 0:
                print(count)

    main_df = main_df.reindex(sorted(main_df.columns), axis=1)

    # print(main_df.head())
    if tickerFile == "./tickers/sp500tickers.pickle":
        main_df.to_csv('sp500_bysymbol_joined_closes.csv')
    elif tickerFile == "./tickers/ETFTickers.pickle":
        main_df.to_csv('ETF_bysymbol_joined_closes.csv')
    else:
        print("What pickle file?",tickerFile)

def visualize_data():
    df = pd.read_csv('./sp500_bysymbol_joined_closes.csv',engine='python')
    # df['AAPL'].plot()
    # plt.show()

    df.set_index('date', inplace=True) 

    df_corr = df.truncate(before="2018-10-01").pct_change().corr() # Find the correlation of each stock's returns
    
    # df_corr = df.corr() ## Correlation table of the data frame

    data = df_corr.values
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,1,1) # 1 by 1, plot number 1

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0] + 0.5), minor=False)
    ax.set_yticks(np.arange(data.shape[1] + 0.5), minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_lables = df_corr.index

    ax.set_xticklabels(column_labels,fontsize=7)
    ax.set_yticklabels(row_lables,fontsize=7)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()

    fig.savefig('correlation.png', dpi=100)

    # plt.show()

def process_data_for_labels(ticker,tickerFile,hm_days):

    if tickerFile == "./tickers/sp500tickers.pickle":
        datafile = './sp500_bysymbol_joined_closes.csv'
    elif tickerFile == "./tickers/ETFTickers.pickle":
        datafile = 'ETF_bysymbol_joined_closes.csv'
    else:
        print("What pickle file?",tickerFile)

    df = pd.read_csv(datafile,engine='python',index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker] * 100 # Calculate the "percent" change (a.k.a error)

    ## Get rid of infinite changes
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)

    return tickers,df
    
def buy_sell_hold(*args):
    cols = [c for c in args]
    buy_threshold = 4.5
    sell_threshold = 4.5
    dominantCol = 0
    # index = 0
    # print(df_global[ticker_global])
    for col in cols:
        if col > buy_threshold:
            if col > abs(dominantCol):
                dominantCol = col
        if col < -sell_threshold:
            if abs(col) > abs(dominantCol):
                dominantCol = col
    if dominantCol > 0:
        return 1 # Buy
    if dominantCol < 0:
        return -1 # Sell
    else:
        return 0 # Hold

def extract_featuresets(ticker,tickerFile,hm_days=7):
    tickers, df = process_data_for_labels(ticker, tickerFile, hm_days)
    # global df_global
    # df_global = df
    # global ticker_global
    # ticker_global = ticker

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, *[df['{}_{}d'.format(ticker, i)]for i in range(1, hm_days+1)],))

    # print(df['{}_target'.format(ticker)].tail())

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    c = Counter(str_vals)
    print('Buy Percentage:',round(c['1']/len(str_vals)*100,2))
    print('Sell Percentage:',round(c['-1']/len(str_vals)*100,2))
    print('Hold Percentage:',round(c['0']/len(str_vals)*100,2))
    print('Data spread:', c)

    df.fillna(0, inplace=True) ## Get rid of N/A

    ## Get rid of infinite changes
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()*100 ## Percent change from the day before
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # print(df_vals)
    # print(df['{}_target'.format(ticker)].values)
    df_vals.to_csv('./pctChange_dfs/pctChange.csv')

    ## X is feature sets, y is labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df

def do_ml(ticker,tickerFile="./tickers/sp500tickers.pickle",hm_days=7,testFraction=0.25,initialQuantity=100):
    X, y, df = extract_featuresets(ticker,tickerFile,hm_days)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testFraction,shuffle=False,stratify = None)

    # clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc',svm.LinearSVC()),('knn',neighbors.KNeighborsClassifier()),('rfor',RandomForestClassifier())])

    clf.fit(X_train,y_train)

    ## We want confidence to be > 33% because at random, there's a 1/3 chance of getting it right
    confidence = clf.score(X_test, y_test) ## This can be pickled! The clf that is...
    predictions = clf.predict(X_test)
    testIndex = len(X)-len(predictions)
    startingDate = df.index[testIndex]
    df_test = df.truncate(before=startingDate)
    df_test['predictions'] = predictions
    df_test.to_csv('./test_dfs/{}.csv'.format(ticker))

    c = Counter(predictions)
    print('Predicted Buy Percentage:',round(c[1]/len(predictions)*100,2))
    print('Predicted Sell Percentage:',round(c[-1]/len(predictions)*100,2))
    print('Predicted Hold Percentage:',round(c[0]/len(predictions)*100,2))
    print('Predicted Data spread:', c)

    myDict = {ticker:initialQuantity}
    initialbalance_in_securities = df_test.loc[startingDate,ticker] * initialQuantity
    myAccount = accounts.Account(-1*initialbalance_in_securities,initialbalance_in_securities,myDict,4.95)

    for index, row in df_test.iterrows():
        if row.predictions == 1: ## BUY
            print(index)
            buyQuantity = 10
            transaction_info = {'security':ticker,'quantity':buyQuantity,'value':row[ticker].astype('float64').item()}
            myAccount.buySecurity(transaction_info)
        if row.predictions == -1: ## SELL
            print(index)
            sellQuantity = 10
            transaction_info = {'security':ticker,'quantity':sellQuantity,'value':row[ticker].astype('float64').item()}
            myAccount.sellSecurity(transaction_info)
        else: ## HOLD
            transaction_info = {'security':ticker,'value':row[ticker].astype('float64').item()}
            myAccount.holdUpdate(transaction_info)


    timeFormat = '%Y-%m-%d'
    elapsedTime = str(datetime.strptime(index, timeFormat) - datetime.strptime(startingDate, timeFormat))

    print("=======================================================================")
    print("Summary for trading",ticker)
    print('\n')

    print("Broker fee per trade:",str(myAccount.broker_fee),'$')
    print('\n')

    print('Benchmark future vision range:',str(hm_days),'days')
    print('Training Set Percentage:',str((1-testFraction)*100),'%')
    print('Classifier Accuracy to Benchmark:',round(confidence*100,2),'%')
    print('\n')

    print('Initial Shares Held:',str(initialQuantity))
    print('Initial Price per Share:',str(round(df.loc[startingDate, ticker],2)),'$')
    print('Initial Buy-In Cost on',startingDate,':',round(initialbalance_in_securities,2),'$')

    print('Final Shares Held:',str(myAccount.securities[ticker]))
    print('Final Price per Share',str(df.loc[index, ticker]),'$')
    print('Final balance_in_securities on',index,':',round(myAccount.balance_in_securities,2),'$')
    print('Final cash:',str(round(myAccount.balance_in_cash,2)),'$')
    profit = round(myAccount.balance_in_securities - abs(myAccount.balance_in_cash),2)
    # round(myAccount.balance_in_securities,2)-round(initialbalance_in_securities,2)
    print('Profit:',round(profit,2),'$')
    percentGrowth = profit / round(initialbalance_in_securities,2) * 100
    print('Percent Growth:',round(percentGrowth,2),'%')
    print('Elapsed Time:',elapsedTime)
    days = (datetime.strptime(index, timeFormat) - datetime.strptime(startingDate, timeFormat)).days
    years = days/365.25
    average_yearly_growth = round(percentGrowth / years, 2)
    months = years * 12
    average_monthly_growth = round(percentGrowth / months, 2)
    print("Average Yearly Growth:", str(average_yearly_growth)+"%")
    print("Average Monthly Growth:", str(average_monthly_growth)+"%")
    print('\n')

    buy_and_hold_profit = round(round(df.loc[index, ticker]*initialQuantity,2)-round(initialbalance_in_securities,2),2)
    print('Buy-and-Hold of',str(initialQuantity),'shares profit:',buy_and_hold_profit,'$')
    buy_and_hold_percentGrowth = buy_and_hold_profit / round(initialbalance_in_securities,2) * 100
    print('Buy-and-Hold of',str(initialQuantity),'shares percent growth:',round(buy_and_hold_percentGrowth,2),'%')
    print("=======================================================================")

    # plotTicker(ticker,tickerFile,startingDate)

def plotTicker(ticker,tickerFile="./tickers/sp500tickers.pickle",startingDate=None):

    # register_matplotlib_converters()
    
    if tickerFile == "./tickers/sp500tickers.pickle":
        datafolder = './sp500_dfs/'
    elif tickerFile == "./tickers/ETFTickers.pickle":
        datafolder = './ETF_dfs/'
    else:
        print("What pickle file?",tickerFile)

    tickerData = datafolder+ticker+'.csv'
    df = pd.read_csv(tickerData, parse_dates=True, index_col=0)

    if startingDate != None:
        df = df.truncate(before=startingDate)

    # df['100ma'] = df['5. adjusted close'].rolling(window=100).mean()

    df['9ema'] = df['5. adjusted close'].ewm(span=9, adjust=False).mean()
    df['12ema'] = df['5. adjusted close'].ewm(span=12, adjust=False).mean()
    df['26ema'] = df['5. adjusted close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['12ema'] - df['26ema']
    # df['macd_pct'] = df['macd'].pct_change()

    previousRow = 0
    for index, row in df.iterrows():
        df.loc[index, 'macd_relChange'] = abs(row.macd-previousRow)*100
        previousRow = row.macd
    # print(df['macd_relChange'])


    ## Get rid of infinite changes
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True) ## Get rid of N/A

    ### ohlc: open high, low close
    df_ohlc = df['5. adjusted close'].resample('10D').ohlc()
    df_volume = df['6. volume'].resample('10D').sum()

    df_ohlc.reset_index(inplace=True)

    df_ohlc.dropna(inplace=True)
    df_volume.dropna(inplace=True)

    # print(df_ohlc.head())
    # print(df_volume.head())

    df_ohlc['date'] = df_ohlc['date'].map(mdates.date2num)
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(19.2,10.8), dpi = 96, sharex=True,gridspec_kw = {'height_ratios':[3, 1, 1]})

    # ax1 = plt.subplot(2, 1, 1)
    # ax2 = plt.subplot(2, 1, 2,sharex=ax1)

    # ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    # ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()

    candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price per Share (USD)')
    titleAX1 = 'Historic Share Price of ' + ticker
    ax1.set_title(titleAX1,horizontalalignment='center', verticalalignment='top')
    ax1.legend()

    # ax2.plot(df.index, df['100ma'])
    # ax2.plot(df.index, df['9ema'],color='blue',linewidth=0.5)
    ax2.plot(df.index, df['macd'],color='black',linewidth=0.5)
    # ax2.plot(df.index, df['macd_relChange'],color='grey',alpha=0.3,linewidth=1, markersize=1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    titleAX2 = 'Technical Indicators of ' + ticker
    ax2.set_title(titleAX2,horizontalalignment='center', verticalalignment='top')
    ax2.legend()

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Number of Shares')
    titleAX3 = 'Market Share Volume of ' + ticker
    ax3.set_title(titleAX3,horizontalalignment='center', verticalalignment='top')
    ax3.fill_between(df_volume.index.map(mdates.date2num), df_volume.values,0,color="blue")

    plt.tight_layout()

    my_dpi=96
    if startingDate == None:
        fig.savefig('./figures/fullhistory/ETF/{}.png'.format(ticker), dpi=my_dpi*10,bbox_inches='tight')

    plt.show()

def add_indicator(dataframe):
    pass

def load_dataframe(ticker, tickerFile="./tickers/sp500tickers.pickle", startingDate=None):
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


def trackTicker(ticker,initialQuantity,startingDate,tickerFile="./tickers/sp500tickers.pickle"):

    # Load the dataframe object
    df = load_dataframe(ticker, tickerFile, startingDate)

    myDict = {ticker:initialQuantity}
    initialbalance_in_securities = df.loc[startingDate,'5. adjusted close'] * initialQuantity
    # myAccount = accounts.Account(-1*initialbalance_in_securities,initialbalance_in_securities,myDict,4.95)
    myAccount = accounts.Account(0,initialbalance_in_securities,myDict,0)

    df = df.truncate(before=startingDate)

    df['9ema'] = df['5. adjusted close'].ewm(span=9, adjust=False).mean()
    df['12ema'] = df['5. adjusted close'].ewm(span=12, adjust=False).mean()
    df['26ema'] = df['5. adjusted close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['12ema'] - df['26ema']

    previousRow = 0
    for index, row in df.iterrows():
        df.loc[index, 'macd_relChange'] = abs(row.macd-previousRow)*100
        previousRow = row.macd

    df.rename(columns={'4. close': 'close'}, inplace=True)
    ## Get rid of infinite changes
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df.close != 0]

    import ta
    # Add bollinger band high indicator filling Nans values
    df['bb_high_indicator'] = ta.bollinger_hband_indicator(df["close"], n=20, ndev=2, fillna=True)

    # Add bollinger band low indicator filling Nans values
    df['bb_low_indicator'] = ta.bollinger_lband_indicator(df["close"], n=20, ndev=2, fillna=True)

    df['bb_high'] = ta.bollinger_hband(df["close"], n=20, ndev=2, fillna=True)
    df['bb_low'] = ta.bollinger_lband(df["close"], n=20, ndev=2, fillna=True)
    # df = ta.add_all_ta_features(df, "1. open", "2. high", "3. low", "4. close", "6. volume", fillna=True)

    df.to_csv('test.csv')

    df.fillna(0, inplace=True) ## Get rid of N/A in main dataframe

    ### ohlc: open high, low close
    df_ohlc = df['5. adjusted close'].resample('10D').ohlc()
    df_volume = df['6. volume'].resample('10D').sum()

    df_ohlc.reset_index(inplace=True)

    df_ohlc.dropna(inplace=True)
    df_volume.dropna(inplace=True)

    # print(df_ohlc.head())
    # print(df_volume.head())

    df_ohlc['date'] = df_ohlc['date'].map(mdates.date2num)

    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()

    # ax1.plot(df.index, df['100ma'])
    ax1.plot(df.index, df['9ema'],color='brown')
    ax1.plot(df.index, df['macd'],color='blue')
    ax1.plot(df.index, df['macd_relChange'],color='grey')
    
    candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values,0) 

    # plt.show()

    df.rename(columns={'4. close': 'close'}, inplace=True)

    previousMACD = 0
    for index, row in df.iterrows():
        if row.close.astype("float64").item() != 0:
            if row.macd < 0 and previousMACD > 0: ## SELL
                print('Date:',index)
                if abs(row.macd_relChange) > 3:
                    if abs(row.macd_relChange) > 6:
                        sellQuantity = 4
                    elif abs(row.macd_relChange) > 5 and abs(row.macd_relChange) < 6:
                        sellQuantity = 3
                    elif abs(row.macd_relChange) > 4 and abs(row.macd_relChange) < 5:
                        sellQuantity = 2
                    else:
                        sellQuantity = 1
                    transaction_info = {'security':ticker,'quantity':sellQuantity,'value':row.close.astype('float64').item()}
                    myAccount.sellSecurity(transaction_info)
                else:
                    sellQuantity = 0
                    print("Not selling. Relative MACD change is only",abs(row.macd_relChange))
            if row.macd > 0 and previousMACD < 0: ## BUY
                print('Date:',index)
                if abs(row.macd_relChange) > 2:
                    if abs(row.macd_relChange) > 5:
                        buyQuantity = 4
                    elif abs(row.macd_relChange) > 4 and abs(row.macd_relChange) < 5:
                        buyQuantity = 3
                    elif abs(row.macd_relChange) > 3 and abs(row.macd_relChange) < 4:
                        buyQuantity = 2
                    else:
                        buyQuantity = 1
                    transaction_info = {'security':ticker,'quantity':buyQuantity,'value':row.close.astype('float64').item()}
                    myAccount.buySecurity(transaction_info)
                else:
                    buyQuantity = 0
                    print("Not buying. Relative MACD change is only",abs(row.macd_relChange))
            previousMACD = row.macd

    timeFormat = '%Y-%m-%d'
    elapsedTime = str(datetime.strptime(index.strftime("%Y-%m-%d"), timeFormat) - datetime.strptime(startingDate, timeFormat))

    print("=======================================================================")
    print("Summary for trading",ticker)
    print('\n')

    print("Broker fee per trade:",str(myAccount.broker_fee),'$')
    print('Initial Shares Held:',str(initialQuantity))
    print('Initial Price per Share:',str(round(df.loc[startingDate,'5. adjusted close'],2)),'$')
    print('Initial Buy-In Cost on',startingDate,':',round(initialbalance_in_securities,2),'$')

    # print(myAccount.securities)
    # print('Initial Buy-In on',startingDate,':',round(initialbalance_in_securities,2),'$')

    print('Final balance_in_securities on',index.strftime("%Y-%m-%d"),':',round(myAccount.balance_in_securities,2),'$')
    print('Profit:',round(round(myAccount.balance_in_securities,2)-round(initialbalance_in_securities,2),2),'$')
    percentGrowth = (round(myAccount.balance_in_securities,2) - round(initialbalance_in_securities,2)) / round(initialbalance_in_securities,2) * 100
    print('Percent Growth:',round(percentGrowth,2),'%')
    print('Time Elapsed:',elapsedTime)
    days = (datetime.strptime(index.strftime("%Y-%m-%d"), timeFormat) - datetime.strptime(startingDate, timeFormat)).days
    years = days/365.25
    average_yearly_growth = round(percentGrowth / years, 2)
    months = years * 12
    average_monthly_growth = round(percentGrowth / months, 2)
    print("Average Yearly Growth:", str(average_yearly_growth)+"%")
    print("Average Monthly Growth:", str(average_monthly_growth)+"%")

    buy_and_hold_profit = round(round(df.loc[index,'5. adjusted close']*initialQuantity,2)-round(initialbalance_in_securities,2),2)
    print('Buy-and-Hold of',str(initialQuantity),'shares profit:',buy_and_hold_profit,'$')
    buy_and_hold_percentGrowth = buy_and_hold_profit / round(initialbalance_in_securities,2) * 100
    print('Buy-and-Hold of',str(initialQuantity),'shares percent growth:',round(buy_and_hold_percentGrowth,2),'%')
    print("=======================================================================")
    
if __name__ == "__main__":

    ticker = 'XIC'

    style.use('ggplot')

    # get_data_from_alphaVantage(False,"./tickers/ETFTickers.pickle")

    # compile_data("./tickers/ETFTickers.pickle")

    # visualize_data()

    # do_ml(ticker,"./tickers/ETFTickers.pickle",hm_days=3,testFraction=0.2,initialQuantity=1000)
    # do_ml(ticker,hm_days=7,testFraction=0.1,initialQuantity=1000)

    ## GENERATE FULL HISTORY PLOTS
    # tickers = list()
    # with open("./tickers/ETFTickers.pickle", "rb") as f:
    #     tickers_full = pickle.load(f)
    # for count,ticker in enumerate(tickers_full):
    #     tickers.append(ticker[1])
    # for ticker in tickers:
    #     fileName = './figures/fullhistory/ETF/{}.png'.format(ticker)
    #     if not os.path.exists(fileName): 
    #         plotTicker(ticker,tickerFile="./tickers/ETFTickers.pickle",startingDate=None)

    trackTicker(ticker,100,'2004-02-02',tickerFile="./tickers/ETFTickers.pickle")

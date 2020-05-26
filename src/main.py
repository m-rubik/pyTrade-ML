"""
NOTE: This script is deprecated.
To be updated...
"""


import bs4 as bs
import datetime as dt
import os
import csv
import pandas as pd
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
from src.utilities.key_utilities import load_key

from datetime import datetime

import accounts

register_matplotlib_converters()


def process_data_for_labels(ticker, tickerFile, hm_days):

    if tickerFile == "./tickers/sp500tickers.pickle":
        datafile = './sp500_bysymbol_joined_closes.csv'
    elif tickerFile == "./tickers/ETFTickers.pickle":
        datafile = 'ETF_bysymbol_joined_closes.csv'
    else:
        print("What pickle file?", tickerFile)

    df = pd.read_csv(datafile, engine='python', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        # Calculate the "percent" change (a.k.a error)
        df['{}_{}d'.format(ticker, i)] = (
            df[ticker].shift(-i) - df[ticker]) / df[ticker] * 100

    # Get rid of infinite changes
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)

    return tickers, df


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
        return 1  # Buy
    if dominantCol < 0:
        return -1  # Sell
    else:
        return 0  # Hold


def extract_featuresets(ticker, tickerFile, hm_days=7):
    tickers, df = process_data_for_labels(ticker, tickerFile, hm_days)
    # global df_global
    # df_global = df
    # global ticker_global
    # ticker_global = ticker

    df['{}_target'.format(ticker)] = list(map(
        buy_sell_hold, *[df['{}_{}d'.format(ticker, i)]for i in range(1, hm_days+1)],))

    # print(df['{}_target'.format(ticker)].tail())

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    c = Counter(str_vals)
    print('Buy Percentage:', round(c['1']/len(str_vals)*100, 2))
    print('Sell Percentage:', round(c['-1']/len(str_vals)*100, 2))
    print('Hold Percentage:', round(c['0']/len(str_vals)*100, 2))
    print('Data spread:', c)

    df.fillna(0, inplace=True)  # Get rid of N/A

    # Get rid of infinite changes
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # Percent change from the day before
    df_vals = df[[ticker for ticker in tickers]].pct_change()*100
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # print(df_vals)
    # print(df['{}_target'.format(ticker)].values)
    df_vals.to_csv('./pctChange_dfs/pctChange.csv')

    # X is feature sets, y is labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def do_ml(ticker, tickerFile="./tickers/sp500tickers.pickle", hm_days=7, testFraction=0.25, initialQuantity=100):
    X, y, df = extract_featuresets(ticker, tickerFile, hm_days)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testFraction, shuffle=False, stratify=None)

    # clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc', svm.LinearSVC(
    )), ('knn', neighbors.KNeighborsClassifier()), ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)

    # We want confidence to be > 33% because at random, there's a 1/3 chance of getting it right
    # This can be pickled! The clf that is...
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)
    testIndex = len(X)-len(predictions)
    starting_date = df.index[testIndex]
    df_test = df.truncate(before=starting_date)
    df_test['predictions'] = predictions
    df_test.to_csv('./test_dfs/{}.csv'.format(ticker))

    c = Counter(predictions)
    print('Predicted Buy Percentage:', round(c[1]/len(predictions)*100, 2))
    print('Predicted Sell Percentage:', round(c[-1]/len(predictions)*100, 2))
    print('Predicted Hold Percentage:', round(c[0]/len(predictions)*100, 2))
    print('Predicted Data spread:', c)

    myDict = {ticker: initialQuantity}
    initialbalance_in_securities = df_test.loc[starting_date,
                                               ticker] * initialQuantity
    myAccount = accounts.Account(-1*initialbalance_in_securities,
                                 initialbalance_in_securities, myDict, 4.95)

    for index, row in df_test.iterrows():
        if row.predictions == 1:  # BUY
            print(index)
            buyQuantity = 10
            transaction_info = {'security': ticker, 'quantity': buyQuantity,
                                'value': row[ticker].astype('float64').item()}
            myAccount.buySecurity(transaction_info)
        if row.predictions == -1:  # SELL
            print(index)
            sellQuantity = 10
            transaction_info = {'security': ticker, 'quantity': sellQuantity,
                                'value': row[ticker].astype('float64').item()}
            myAccount.sellSecurity(transaction_info)
        else:  # HOLD
            transaction_info = {'security': ticker,
                                'value': row[ticker].astype('float64').item()}
            myAccount.holdUpdate(transaction_info)

    timeFormat = '%Y-%m-%d'
    elapsedTime = str(datetime.strptime(index, timeFormat) -
                      datetime.strptime(starting_date, timeFormat))

    print("=======================================================================")
    print("Summary for trading", ticker)
    print('\n')

    print("Broker fee per trade:", str(myAccount.broker_fee), '$')
    print('\n')

    print('Benchmark future vision range:', str(hm_days), 'days')
    print('Training Set Percentage:', str((1-testFraction)*100), '%')
    print('Classifier Accuracy to Benchmark:', round(confidence*100, 2), '%')
    print('\n')

    print('Initial Shares Held:', str(initialQuantity))
    print('Initial Price per Share:', str(
        round(df.loc[starting_date, ticker], 2)), '$')
    print('Initial Buy-In Cost on', starting_date, ':',
          round(initialbalance_in_securities, 2), '$')

    print('Final Shares Held:', str(myAccount.securities[ticker]))
    print('Final Price per Share', str(df.loc[index, ticker]), '$')
    print('Final balance_in_securities on', index, ':',
          round(myAccount.balance_in_securities, 2), '$')
    print('Final cash:', str(round(myAccount.balance_in_cash, 2)), '$')
    profit = round(myAccount.balance_in_securities -
                   abs(myAccount.balance_in_cash), 2)
    # round(myAccount.balance_in_securities,2)-round(initialbalance_in_securities,2)
    print('Profit:', round(profit, 2), '$')
    percentGrowth = profit / round(initialbalance_in_securities, 2) * 100
    print('Percent Growth:', round(percentGrowth, 2), '%')
    print('Elapsed Time:', elapsedTime)
    days = (datetime.strptime(index, timeFormat) -
            datetime.strptime(starting_date, timeFormat)).days
    years = days/365.25
    average_yearly_growth = round(percentGrowth / years, 2)
    months = years * 12
    average_monthly_growth = round(percentGrowth / months, 2)
    print("Average Yearly Growth:", str(average_yearly_growth)+"%")
    print("Average Monthly Growth:", str(average_monthly_growth)+"%")
    print('\n')

    buy_and_hold_profit = round(round(
        df.loc[index, ticker]*initialQuantity, 2)-round(initialbalance_in_securities, 2), 2)
    print('Buy-and-Hold of', str(initialQuantity),
          'shares profit:', buy_and_hold_profit, '$')
    buy_and_hold_percentGrowth = buy_and_hold_profit / \
        round(initialbalance_in_securities, 2) * 100
    print('Buy-and-Hold of', str(initialQuantity),
          'shares percent growth:', round(buy_and_hold_percentGrowth, 2), '%')
    print("=======================================================================")

    # plot_dataframe(ticker,tickerFile,starting_date)


def add_indicator(dataframe):
    pass


def load_dataframe(ticker, tickerFile="./tickers/sp500tickers.pickle", starting_date=None):
    if tickerFile == "./tickers/sp500tickers.pickle":
        datafolder = './sp500_dfs/'
    elif tickerFile == "./tickers/ETFTickers.pickle":
        datafolder = './ETF_dfs/'
    else:
        print("Unrecognized ticker file:", tickerFile)
    tickerData = datafolder+ticker+'.csv'
    try:
        df = pd.read_csv(tickerData, parse_dates=True, index_col=0)
        df = df.truncate(before=starting_date)
    except Exception as df:
        print(df)
    finally:
        return df


def trackTicker(ticker, initialQuantity, starting_date, tickerFile="./tickers/sp500tickers.pickle"):

    # Load the dataframe object
    df = load_dataframe(ticker, tickerFile, starting_date)

    myDict = {ticker: initialQuantity}
    initialbalance_in_securities = df.loc[starting_date,
                                          '5. adjusted close'] * initialQuantity
    # myAccount = accounts.Account(-1*initialbalance_in_securities,initialbalance_in_securities,myDict,4.95)
    myAccount = accounts.Account(0, initialbalance_in_securities, myDict, 0)

    df = df.truncate(before=starting_date)

    df['9ema'] = df['5. adjusted close'].ewm(span=9, adjust=False).mean()
    df['12ema'] = df['5. adjusted close'].ewm(span=12, adjust=False).mean()
    df['26ema'] = df['5. adjusted close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['12ema'] - df['26ema']

    previousRow = 0
    for index, row in df.iterrows():
        df.loc[index, 'macd_relChange'] = abs(row.macd-previousRow)*100
        previousRow = row.macd

    df.rename(columns={'4. close': 'close'}, inplace=True)
    # Get rid of infinite changes
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df.close != 0]

    import ta
    # Add bollinger band high indicator filling Nans values
    df['bb_high_indicator'] = ta.bollinger_hband_indicator(
        df["close"], n=20, ndev=2, fillna=True)

    # Add bollinger band low indicator filling Nans values
    df['bb_low_indicator'] = ta.bollinger_lband_indicator(
        df["close"], n=20, ndev=2, fillna=True)

    df['bb_high'] = ta.bollinger_hband(df["close"], n=20, ndev=2, fillna=True)
    df['bb_low'] = ta.bollinger_lband(df["close"], n=20, ndev=2, fillna=True)
    # df = ta.add_all_ta_features(df, "1. open", "2. high", "3. low", "4. close", "6. volume", fillna=True)

    df.to_csv('test.csv')

    df.fillna(0, inplace=True)  # Get rid of N/A in main dataframe

    # ohlc: open high, low close
    df_ohlc = df['5. adjusted close'].resample('10D').ohlc()
    df_volume = df['6. volume'].resample('10D').sum()

    df_ohlc.reset_index(inplace=True)

    df_ohlc.dropna(inplace=True)
    df_volume.dropna(inplace=True)

    # print(df_ohlc.head())
    # print(df_volume.head())

    df_ohlc['date'] = df_ohlc['date'].map(mdates.date2num)

    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()

    # ax1.plot(df.index, df['100ma'])
    ax1.plot(df.index, df['9ema'], color='brown')
    ax1.plot(df.index, df['macd'], color='blue')
    ax1.plot(df.index, df['macd_relChange'], color='grey')

    candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

    # plt.show()

    df.rename(columns={'4. close': 'close'}, inplace=True)

    previousMACD = 0
    for index, row in df.iterrows():
        if row.close.astype("float64").item() != 0:
            if row.macd < 0 and previousMACD > 0:  # SELL
                print('Date:', index)
                if abs(row.macd_relChange) > 3:
                    if abs(row.macd_relChange) > 6:
                        sellQuantity = 4
                    elif abs(row.macd_relChange) > 5 and abs(row.macd_relChange) < 6:
                        sellQuantity = 3
                    elif abs(row.macd_relChange) > 4 and abs(row.macd_relChange) < 5:
                        sellQuantity = 2
                    else:
                        sellQuantity = 1
                    transaction_info = {
                        'security': ticker, 'quantity': sellQuantity, 'value': row.close.astype('float64').item()}
                    myAccount.sellSecurity(transaction_info)
                else:
                    sellQuantity = 0
                    print("Not selling. Relative MACD change is only",
                          abs(row.macd_relChange))
            if row.macd > 0 and previousMACD < 0:  # BUY
                print('Date:', index)
                if abs(row.macd_relChange) > 2:
                    if abs(row.macd_relChange) > 5:
                        buyQuantity = 4
                    elif abs(row.macd_relChange) > 4 and abs(row.macd_relChange) < 5:
                        buyQuantity = 3
                    elif abs(row.macd_relChange) > 3 and abs(row.macd_relChange) < 4:
                        buyQuantity = 2
                    else:
                        buyQuantity = 1
                    transaction_info = {
                        'security': ticker, 'quantity': buyQuantity, 'value': row.close.astype('float64').item()}
                    myAccount.buySecurity(transaction_info)
                else:
                    buyQuantity = 0
                    print("Not buying. Relative MACD change is only",
                          abs(row.macd_relChange))
            previousMACD = row.macd

    timeFormat = '%Y-%m-%d'
    elapsedTime = str(datetime.strptime(index.strftime(
        "%Y-%m-%d"), timeFormat) - datetime.strptime(starting_date, timeFormat))

    print("=======================================================================")
    print("Summary for trading", ticker)
    print('\n')

    print("Broker fee per trade:", str(myAccount.broker_fee), '$')
    print('Initial Shares Held:', str(initialQuantity))
    print('Initial Price per Share:', str(
        round(df.loc[starting_date, '5. adjusted close'], 2)), '$')
    print('Initial Buy-In Cost on', starting_date, ':',
          round(initialbalance_in_securities, 2), '$')

    # print(myAccount.securities)
    # print('Initial Buy-In on',starting_date,':',round(initialbalance_in_securities,2),'$')

    print('Final balance_in_securities on', index.strftime("%Y-%m-%d"),
          ':', round(myAccount.balance_in_securities, 2), '$')
    print('Profit:', round(round(myAccount.balance_in_securities, 2) -
                           round(initialbalance_in_securities, 2), 2), '$')
    percentGrowth = (round(myAccount.balance_in_securities, 2) - round(
        initialbalance_in_securities, 2)) / round(initialbalance_in_securities, 2) * 100
    print('Percent Growth:', round(percentGrowth, 2), '%')
    print('Time Elapsed:', elapsedTime)
    days = (datetime.strptime(index.strftime("%Y-%m-%d"), timeFormat) -
            datetime.strptime(starting_date, timeFormat)).days
    years = days/365.25
    average_yearly_growth = round(percentGrowth / years, 2)
    months = years * 12
    average_monthly_growth = round(percentGrowth / months, 2)
    print("Average Yearly Growth:", str(average_yearly_growth)+"%")
    print("Average Monthly Growth:", str(average_monthly_growth)+"%")

    buy_and_hold_profit = round(round(
        df.loc[index, '5. adjusted close']*initialQuantity, 2)-round(initialbalance_in_securities, 2), 2)
    print('Buy-and-Hold of', str(initialQuantity),
          'shares profit:', buy_and_hold_profit, '$')
    buy_and_hold_percentGrowth = buy_and_hold_profit / \
        round(initialbalance_in_securities, 2) * 100
    print('Buy-and-Hold of', str(initialQuantity),
          'shares percent growth:', round(buy_and_hold_percentGrowth, 2), '%')
    print("=======================================================================")


if __name__ == "__main__":

    ticker = 'XIC'

    style.use('ggplot')

    key = load_key()

    # get_data_from_alphaVantage(key, False,"./tickers/ETFTickers.pickle")

    # compile_data("./tickers/ETFTickers.pickle")

    # do_ml(ticker,"./tickers/ETFTickers.pickle",hm_days=3,testFraction=0.2,initialQuantity=1000)
    # do_ml(ticker,hm_days=7,testFraction=0.1,initialQuantity=1000)

    # GENERATE FULL HISTORY PLOTS
    # tickers = list()
    # with open("./tickers/ETFTickers.pickle", "rb") as f:
    #     tickers_full = pickle.load(f)
    # for count,ticker in enumerate(tickers_full):
    #     tickers.append(ticker[1])
    # for ticker in tickers:
    #     fileName = './figures/fullhistory/ETF/{}.png'.format(ticker)
    #     if not os.path.exists(fileName):
    #         plot_dataframe(ticker,tickerFile="./tickers/ETFTickers.pickle",starting_date=None)

    trackTicker(ticker, 100, '2004-02-02',
                tickerFile="./tickers/ETFTickers.pickle")

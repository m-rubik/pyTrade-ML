import matplotlib.pyplot as plt
import numpy as np
import pytrademl.utilities.dataframe_utilities as dataframe_utilities
import matplotlib.dates as mdates
from matplotlib import style
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
style.use('ggplot')


def generate_correlation_plot(df, name="correlation", starting_date=None, show=False, save=True):
    """
    Generates a correlation plot between all columns in a dataframe.
    Note that in order to get a meaningful correlation, we take the percent change of each column.
    """

    if starting_date is not None:
        df_corr = df.truncate(before=starting_date).pct_change().corr()
    else:
        df_corr = df.pct_change().corr()
    data = df_corr.values

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)  # 1 by 1, plot number 1
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0] + 0.5), minor=False)
    ax.set_yticks(np.arange(data.shape[1] + 0.5), minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    column_labels = df_corr.columns
    row_lables = df_corr.index
    ax.set_xticklabels(column_labels, fontsize=20)
    ax.set_yticklabels(row_lables, fontsize=20)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()

    if save:
        fig.savefig('./figures/'+name+'.png', dpi=100)
    if show:
        plt.show()


def plot_dataframe(df, ticker, name=None, plot=True, starting_date=None):

    from mplfinance import candlestick_ohlc

    df = dataframe_utilities.add_indicators(df)

    if starting_date is not None:
        df = df.truncate(before=starting_date)

    # ohlc: open high, low close
    # df_ohlc = df['5. adjusted close'].resample('10D').ohlc()
    # df_ohlc = df['4. close'].resample('10D').ohlc()
    # df_volume = df['6. volume'].resample('10D').sum()

    df_ohlc = df.resample("D").agg(
        {'1. open': 'first', '2. high': 'max', '2. low': 'min', '4. close': 'last'})
    # df_ohlc = df_resampled.ohlc()
    df_volume = df['6. volume'].resample("D").sum()
    df_ohlc.reset_index(inplace=True)
    df_ohlc.dropna(inplace=True)
    df_volume.dropna(inplace=True)

    df_ohlc['date'] = df_ohlc['date'].map(mdates.date2num)

    df_ohlc.head()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(
        19.2, 10.8), dpi=96, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 0.5]})

    ax1.xaxis_date()

    candlestick_ohlc(ax1, df_ohlc.values, width=1, colorup='#77d879', colordown='#db3f3f')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price per Share (USD)')
    titleAX1 = 'Historic Share Price of ' + ticker
    ax1.set_title(titleAX1, horizontalalignment='center',
                  verticalalignment='top')
    ax1.plot(df.index, df['volatility_bbh'], color="yellow",
             alpha=0.5, linewidth=1, label="Bollinger High")
    ax1.plot(df.index, df['volatility_bbl'], color="red",
             alpha=0.5, linewidth=1, label="Bollinger Low")
    ax1.fill_between(df.index, y1=df['volatility_bbh'],
                     y2=df['volatility_bbl'], color='orange', alpha='0.3')
    ax1.legend(loc="lower left")

    ax2.plot(df.index, df['trend_macd'], color='purple',
             linewidth=0.5, label="MACD")
    ax2.plot(df.index, df['trend_macd_signal'], color='orange',
             alpha=0.5, linewidth=1, markersize=1, label="MACD Signal")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.set_ylim([-1, 1])
    titleAX2 = 'MACD of ' + ticker
    ax2.set_title(titleAX2, horizontalalignment='center',
                  verticalalignment='top')
    ax2.legend(loc="lower left")

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    titleAX3 = 'Other Indicators of ' + ticker
    ax3.set_title(titleAX3, horizontalalignment='center',
                  verticalalignment='top')
    ax3.plot(df.index, df['momentum_rsi'], color='purple',
             alpha=1, linewidth=1, markersize=1, label="RSI")
    ax3.plot(df.index, df['momentum_stoch'], color='black',
             alpha=1, linewidth=1, markersize=1, label="Stochastic %K")
    ax3.plot(df.index, df['momentum_stoch_signal'], color='red',
             alpha=1, linewidth=1, markersize=1, label="Stochastic %D")

    horiz_line_data = np.array([30 for i in range(len(df.index))])
    ax3.plot(df.index, horiz_line_data, '--',
             alpha=0.5, linewidth=0.5, label="RSI Low")
    horiz_line_data = np.array([70 for i in range(len(df.index))])
    ax3.plot(df.index, horiz_line_data, '--',
             alpha=0.5, linewidth=0.5, label="RSI High")
    horiz_line_data = np.array([20 for i in range(len(df.index))])
    ax3.plot(df.index, horiz_line_data, '--', alpha=0.5,
             linewidth=0.5, label="Stochastic Low")
    horiz_line_data = np.array([80 for i in range(len(df.index))])
    ax3.plot(df.index, horiz_line_data, '--', alpha=0.5,
             linewidth=0.5, label="Stochastic High")
    ax3.legend(loc="lower left")

    ax4.set_xlabel('Time')
    ax4.set_ylabel('Number of Shares')
    titleAX4 = 'Market Share Volume of ' + ticker
    ax4.set_title(titleAX4, horizontalalignment='center',
                  verticalalignment='top')
    ax4.fill_between(df_volume.index.map(mdates.date2num),
                     df_volume.values, 0, color="blue")

    plt.tight_layout()

    my_dpi = 96

    if name is None:
        name = ticker
    fig.savefig('./figures/{}.png'.format(name),
                dpi=my_dpi*10, bbox_inches='tight')

    # if starting_date == None:
    #     fig.savefig('./figures/fullhistory/ETF/{}.png'.format(ticker), dpi=my_dpi*10,bbox_inches='tight')

    if plot:
        plt.show()


def plot_account_pie(account):
    import pandas as pd
    import datetime
    data = {}
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
    for security, quantity in account.securities.items():
        if quantity != 0:
            datafolder = './ETF_dfs/'
            tickerData = datafolder+security+'.csv'
            df = pd.read_csv(tickerData, parse_dates=True, index_col=0)
            data[security] = df.loc[today, "5. adjusted close"] * quantity

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(19.2, 10.8), dpi=96)
    ax1.pie(data.values(), labels=data.keys(),
            autopct='%1.1f%%', startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')
    titleAX1 = 'Holdings by Share Values'
    ax1.set_title(titleAX1, horizontalalignment='center',
                  verticalalignment='top')
    ax2.pie([float(v) for v in account.securities.values()],
            labels=account.securities.keys(), autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    titleAX2 = 'Holdings by Share Volume'
    ax2.set_title(titleAX2, horizontalalignment='center',
                  verticalalignment='top')
    plt.show()


def plot_account(account):
    import pandas as pd
    import datetime
    dfs = {}

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

    for security, quantity in account.securities.items():
        if quantity != 0:
            datafolder = './TSX_dfs/'
            tickerData = datafolder+security+'.csv'
            dfs[security] = pd.read_csv(
                tickerData, parse_dates=True, index_col=0)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(19.2, 10.8), dpi=96)
    ax1.xaxis_date()
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price per Share (CAD)')
    titleAX1 = 'Market History'
    ax1.set_title(titleAX1, horizontalalignment='center',
                  verticalalignment='top')

    for ticker, df in dfs.items():
        ax1.plot(df.index, df['5. adjusted close'], linewidth=1, label=ticker)
    ax1.legend(loc="lower left")

    import pandas as pd
    import datetime
    data = {}
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
    for security, quantity in account.securities.items():
        if quantity != 0:
            datafolder = './TSX_dfs/'
            tickerData = datafolder+security+'.csv'
            df = pd.read_csv(tickerData, parse_dates=True, index_col=0)
            data[security] = round(
                df.loc[today, "5. adjusted close"] * quantity, 2)

    ax2.pie(data.values(), labels=data.keys(),
            autopct='%1.1f%%', startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.axis('equal')
    titleAX2 = 'Holdings by Share Values'
    ax2.set_title(titleAX2, horizontalalignment='center',
                  verticalalignment='top')

    plt.show()


def plot_account_history(account):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(19.2, 10.8), dpi=96)
    ax1.xaxis_date()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Balance (CAD)')
    titleAX1 = 'Account Balance History'
    ax1.set_title(titleAX1, horizontalalignment='center',
                  verticalalignment='top')

    for k, v in dict(account.balance_history).items():
        if k is None:
            del account.balance_history[k]
    # sorted by key, return a list of tuples
    lists = sorted(account.balance_history.items())
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    ax1.plot(x, y)
    plt.show()


def plot_confusion_matrix(confusion_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(confusion_matrix, cmap=plt.get_cmap('Blues'))
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    if confusion_matrix.shape[0] == 3:
        labels = ['Sell', 'Hold', 'Buy']
    else:
        labels = ['Sell', 'Buy']
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="black")
    fig.tight_layout()
    plt.show()


def plot_predictions(predictions, y_test):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(range(0, len(predictions)), predictions)
    ax.scatter(range(0, len(y_test)), y_test)
    plt.show()


if __name__ == "__main__":
    import pytrademl.utilities.dataframe_utilities as dataframe_utilities

    # df = dataframe_utilities.generate_adjclose_df("./tickers/ETFTickers.pickle")
    # generate_correlation_plot(df=df, name="correlation_2019-04-11", starting_date="2019-04-11", show=True, save=True)

    df = dataframe_utilities.import_dataframe("XIC")
    plot_dataframe(df, "XIC", "XIC_test", True, "2019-09-03")

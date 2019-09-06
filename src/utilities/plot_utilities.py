import matplotlib.pyplot as plt
import numpy as np
import src.utilities.dataframe_utilities as dataframe_utilities
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib import style

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
    ax.set_xticklabels(column_labels,fontsize=20)
    ax.set_yticklabels(row_lables,fontsize=20)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()

    if save:
        fig.savefig('./figures/'+name+'.png', dpi=100)
    if show:
        plt.show()

def plot_dataframe(df, ticker, name=None, plot=True):

    df = dataframe_utilities.add_indicators(df)

    ### ohlc: open high, low close
    # df_ohlc = df['5. adjusted close'].resample('10D').ohlc()
    df_ohlc = df['4. close'].resample('10D').ohlc()
    df_volume = df['6. volume'].resample('10D').sum()

    df_ohlc.reset_index(inplace=True)

    df_ohlc.dropna(inplace=True)
    df_volume.dropna(inplace=True)


    df_ohlc['date'] = df_ohlc['date'].map(mdates.date2num)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(19.2,10.8), dpi = 96, sharex=True,gridspec_kw = {'height_ratios':[2, 1, 1, 0.5]})

    ax1.xaxis_date()

    candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price per Share (USD)')
    titleAX1 = 'Historic Share Price of ' + ticker
    ax1.set_title(titleAX1,horizontalalignment='center', verticalalignment='top')
    ax1.plot(df.index, df['volatility_bbh'], color="yellow", alpha=0.5, linewidth=1, label="Bollinger High")
    ax1.plot(df.index, df['volatility_bbl'], color="red", alpha=0.5, linewidth=1, label="Bollinger Low")
    ax1.fill_between(df.index, y1=df['volatility_bbh'], y2=df['volatility_bbl'], color='orange', alpha='0.3')
    ax1.legend(loc="lower left")

    ax2.plot(df.index, df['trend_macd'],color='purple',linewidth=0.5, label="MACD")
    ax2.plot(df.index, df['trend_macd_signal'],color='orange', alpha=0.5, linewidth=1, markersize=1, label="MACD Signal")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.set_ylim([-1,1])
    titleAX2 = 'MACD of ' + ticker
    ax2.set_title(titleAX2,horizontalalignment='center', verticalalignment='top')
    ax2.legend(loc="lower left")

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    titleAX3 = 'Other Indicators of ' + ticker
    ax3.set_title(titleAX3,horizontalalignment='center', verticalalignment='top')
    ax3.plot(df.index, df['momentum_rsi'],color='purple',alpha=1,linewidth=1, markersize=1, label="RSI")
    ax3.plot(df.index, df['momentum_stoch'],color='black',alpha=1,linewidth=1, markersize=1, label="Stochastic %K")
    ax3.plot(df.index, df['momentum_stoch_signal'],color='red',alpha=1,linewidth=1, markersize=1, label="Stochastic %D")

    horiz_line_data = np.array([30 for i in range(len(df.index))])
    ax3.plot(df.index, horiz_line_data, '--', alpha=0.5, linewidth=0.5, label="RSI Low") 
    horiz_line_data = np.array([70 for i in range(len(df.index))])
    ax3.plot(df.index, horiz_line_data, '--', alpha=0.5, linewidth=0.5, label="RSI High") 
    horiz_line_data = np.array([20 for i in range(len(df.index))])
    ax3.plot(df.index, horiz_line_data, '--', alpha=0.5, linewidth=0.5, label="Stochastic Low") 
    horiz_line_data = np.array([80 for i in range(len(df.index))])
    ax3.plot(df.index, horiz_line_data, '--', alpha=0.5, linewidth=0.5, label="Stochastic High")
    ax3.legend(loc="lower left")


    ax4.set_xlabel('Time')
    ax4.set_ylabel('Number of Shares')
    titleAX4 = 'Market Share Volume of ' + ticker
    ax4.set_title(titleAX4,horizontalalignment='center', verticalalignment='top')
    ax4.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0, color="blue")

    plt.tight_layout()

    my_dpi=96

    if name is None:
        name = ticker
    fig.savefig('./figures/{}.png'.format(name), dpi=my_dpi*10,bbox_inches='tight')    

    # if starting_date == None:
    #     fig.savefig('./figures/fullhistory/ETF/{}.png'.format(ticker), dpi=my_dpi*10,bbox_inches='tight')

    if plot:
        plt.show()


if __name__ == "__main__":
    import src.utilities.dataframe_utilities as dataframe_utilities

    # df = dataframe_utilities.generate_adjclose_df("./tickers/ETFTickers.pickle")
    # generate_correlation_plot(df=df, name="correlation_2019-04-11", starting_date="2019-04-11", show=True, save=True)

    df = dataframe_utilities.import_dataframe("XIC","./tickers/ETFTickers.pickle")
    plot_dataframe(df,"XIC","XIC_test", True)
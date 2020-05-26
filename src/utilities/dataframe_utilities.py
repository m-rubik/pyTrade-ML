import os
import ta
import pandas as pd
import numpy as np
import datetime
import src.utilities.ticker_utilities as ticker_utilities


def import_dataframe(ticker, starting_date=None, enhanced=False):
    """
    Import the dataframe containing historical intraday data for a ticker
    """

    if enhanced:
        df, path = find_dataframe(ticker+'_enhanced.csv')
        if df is not None:
            if starting_date is not None:
                try:
                    df = df.truncate(before=starting_date)
                except Exception as err:
                    print(err)
                finally:
                    return df
            else:
                return df
    df, path = find_dataframe(ticker+'.csv')
    if df is not None:
        if enhanced:
            df = add_indicators(df)
            df = add_historic_indicators(df)
            df = add_pct_change(df)
            df.to_csv(path.split(ticker+'.csv')[0]+ticker+"_enhanced.csv")
        if starting_date is not None:
            try:
                df = df.truncate(before=starting_date)
            except Exception as err:
                print(err)
            finally:
                return df
        return df
    else:
        return df


def find_dataframe(ticker_name):
    found = False
    for ticker_file in os.listdir('./tickers/'):
        try:
            name = ticker_file.split("tickers.pickle")[0]
            ticker_dataframe_path = './dataframes/'+name+"/"+ticker_name
            if os.path.exists(ticker_dataframe_path):
                df = pd.read_csv(ticker_dataframe_path,
                                 parse_dates=True, index_col=0)
                found = True
                break
        except Exception as err:
            print(err)
            break
    if not found:
        return None, None
    else:
        return df, ticker_dataframe_path


def export_dataframe(df, name):
    pass


def generate_adjclose_df(ticker_file="./tickers/ETFTickers.pickle"):
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
            df.drop(['1. open', '2. high', '3. low', '4. close', '6. volume',
                     '7. dividend amount', '8. split coefficient'], 1, inplace=True)

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

    df = ta.add_all_ta_features(
        df, "1. open", "2. high", "3. low", "4. close", "6. volume", fillna=True)

    # # Get rid of infinite changes
    # df = df.replace([np.inf, -np.inf], np.nan)

    # # Replace NaN with 0
    # df.fillna(0, inplace=True)

    return df


def analyse_volume_indicators(df, date):

    df_day = df.loc[date]

    # TODO: Learn about A/D, OBV, FI, EM, VPT, NVI
    if df_day["volume_cmf"] > 0:
        buy_pressure = round(df_day["volume_cmf"] * 100, 2)
        color_print("CMF BUY pressure at " + str(buy_pressure) + "%")
    elif df_day["volume_cmf"] < 0:
        sell_pressure = round(df_day["volume_cmf"] * 100, 2)
        color_print("CMF SELL pressure at " + str(sell_pressure) + "%")
    else:
        color_print("CMF NEUTRAL")


def analyse_volatility_indicators(df, date):

    df_day = df.loc[date]

    # TODO: Learn about ATR
    if df_day["volatility_bbhi"] == 1:
        color_print("Bollinger Band High Breakout: SELL")
    if df_day["volatility_bbli"] == 1:
        color_print("Bollinger Band Low Breakout: BUY")
    else:
        color_print("Bollinger Band NEUTRAL")

    if df_day["volatility_kchi"] == 1:
        color_print("Keltner Channel High Breakout: BUY")
    if df_day["volatility_kcli"] == 1:
        color_print("Keltner Channel Low Breakout: SELL")
    else:
        color_print("Keltner Channel NEUTRAL")

    if df_day["volatility_dchi"] == 1:
        color_print("Donchian Channel High Breakout: BUY")
    if df_day["volatility_dcli"] == 1:
        color_print("Donchian Channel Low Breakout: SELL")
    else:
        color_print("Donchian Channel NEUTRAL")


def analyse_trend_indicators(df, date):

    df_day = df.loc[date]
    yesterday = (datetime.datetime.strptime(date, "%Y-%m-%d") -
                 datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    df_yesterday = df.loc[yesterday]

    if df_day["trend_macd_signal"] > df_day["trend_macd"] and df_yesterday["trend_macd_signal"] <= df_yesterday["trend_macd"]:
        color_print("MACD signal SELL")
    elif df_day["trend_macd_signal"] < df_day["trend_macd"] and df_yesterday["trend_macd_signal"] >= df_yesterday["trend_macd"]:
        color_print("MACD signal BUY")
    else:
        color_print("MACD signal NEUTRAL")

    if df_day["trend_vortex_ind_pos"] > df_day["trend_vortex_ind_neg"] and df_yesterday["trend_vortex_ind_pos"] <= df_yesterday["trend_vortex_ind_neg"]:
        color_print("Vortex signal BUY")
    elif df_day["trend_vortex_ind_pos"] < df_day["trend_vortex_ind_neg"] and df_yesterday["trend_vortex_ind_pos"] >= df_yesterday["trend_vortex_ind_neg"]:
        color_print("Vortex signal SELL")
    else:
        color_print("Vortex signal NEUTRAL")

    if df_day["trend_trix"] > 0:
        strength = round(abs(df_day["trend_trix"]) * 100, 2)
        color_print("TRIX signal BUY strength " + str(strength))
    elif df_day["trend_trix"] < 0:
        strength = round(abs(df_day["trend_trix"]) * 100, 2)
        color_print("TRIX signal SELL strength " + str(strength))
    else:
        color_print("TRIX signal NEUTRAL")

    # TODO: Figure out how to use mass_index

    if df_day["trend_cci"] > 100:
        color_print("CCI signal BUY")
    elif df_day["trend_cci"] < -100:
        color_print("CCI signal SELL")
    else:
        color_print("CCI signal NEUTRAL")

    # TODO: NOT SURE ABOUT THIS
    # if df_day["trend_kst_sig"] > df_day["trend_kst"] and df_yesterday["trend_kst_sig"] <= df_yesterday["trend_kst"]:
    #     color_print("KST signal SELL")
    # elif df_day["trend_kst_sig"] < df_day["trend_kst"] and df_yesterday["trend_kst_sig"] >= df_yesterday["trend_kst"]:
    #     color_print("KST signal BUY")
    # else:
    #     color_print("KST signal neutral")

    if df_day["trend_ichimoku_a"] > df_yesterday["trend_ichimoku_a"] and df_day["trend_ichimoku_a"] > df_day["trend_ichimoku_b"]:
        color_print("Ichimoku signal BUY")
    elif df_day["trend_ichimoku_a"] < df_yesterday["trend_ichimoku_a"] and df_day["trend_ichimoku_a"] < df_day["trend_ichimoku_b"]:
        color_print("Ichimoku signal BUY")
    else:
        color_print("Ichimoku signal NEUTRAL")

    if df_day["trend_aroon_ind"] > 0:
        color_print("AROON signal BUY")
    if df_day["trend_aroon_ind"] < 0:
        color_print("AROON signal SELL")
    else:
        color_print("AROON signal NEUTRAL")


def analyse_momentum_indicators(df, date):
    df_day = df.loc[date]
    yesterday = (datetime.datetime.strptime(date, "%Y-%m-%d") -
                 datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    df_yesterday = df.loc[yesterday]

    if df_day["momentum_rsi"] > 80:
        color_print("RSI signal STRONG OVERBOUGHT")
    elif df_day["momentum_rsi"] > 70:
        color_print("RSI signal OVERBOUGHT")
    elif df_day["momentum_rsi"] < 20:
        color_print("RSI signal STRONG OVERSELL")
    elif df_day["momentum_rsi"] < 30:
        color_print("RSI signal OVERSELL")
    else:
        color_print("RSI signal NEUTRAL")

    if df_day["momentum_mfi"] > 80:
        color_print("MFI signal STRONG OVERBOUGHT")
    elif df_day["momentum_mfi"] > 70:
        color_print("MFI signal OVERBOUGHT")
    elif df_day["momentum_mfi"] < 20:
        color_print("MFI signal STRONG OVERSOLD")
    elif df_day["momentum_mfi"] < 30:
        color_print("MFI signal OVERSOLD")
    else:
        color_print("MFI signal NEUTRAL")

    # TODO: Figure out how to use TSI, UO

    if df_day["momentum_stoch"] > 80:
        color_print("STOCHASTIC indicating OVERBOUGHT")
    elif df_day["momentum_stoch"] < 20:
        color_print("STOCHASTIC indicating OVERSOLD")
    else:
        color_print("STOCHASTIC indicating NEUTRAL")

    if df_day["momentum_stoch"] > df_day["momentum_stoch_signal"] and df_yesterday["momentum_stoch"] <= df_yesterday["momentum_stoch_signal"]:
        color_print("STOCHASTIC signal BUY")
    elif df_day["momentum_stoch"] < df_day["momentum_stoch_signal"] and df_yesterday["momentum_stoch"] >= df_yesterday["momentum_stoch_signal"]:
        color_print("STOCHASTIC signal SELL")
    else:
        color_print("STOCHASTIC signal NEUTRAL")

    if df_day["momentum_wr"] < 0 and df_day["momentum_wr"] > -20:
        color_print("Williams indicating OVERBOUGHT")
    elif df_day["momentum_wr"] < -80 and df_day["momentum_wr"] > -100:
        color_print("Williams indicating OVERSOLD")
    else:
        color_print("Williams indicating NEUTRAL")

    # TODO: Add ao


def analyse_df(df, date):
    # Analyse volume indicators
    color_print("==== VOLUME ANALYSIS ====")
    analyse_volume_indicators(df, date)

    # Analyse volatility indicators
    color_print("==== VOLTATILITY ANALYSIS ====")
    analyse_volatility_indicators(df, date)

    # Analyse trend indicators
    color_print("==== TREND ANALYSIS ====")
    analyse_trend_indicators(df, date)

    # Analyse momentum indicators
    color_print("==== MOMENTUM ANALYSIS ====")
    analyse_momentum_indicators(df, date)


def color_print(text):
    # text = text.replace("BUY", "\033[0;32m" + " BUY " + "\033[0m")
    # text = text.replace("SELL", "\033[0;31m" + " SELL " + "\033[0m")
    # text = text.replace("NEUTRAL", "\033[0;37m" + " NEUTRAL " + "\033[0m")
    print(text)


def add_future_vision(df, buy_threshold, sell_threshold):

    if 'correct_decision' not in df.keys():
        print("Adding future vision...")
        df['correct_decision'] = 0
        for i in range(0, df.shape[0]-1):
            tomorrow_close = df.iloc[i+1]['5. adjusted close']
            today_close = df.iloc[i]['5. adjusted close']
            tomorrow_low = df.iloc[i+1]['3. low']

            if tomorrow_close > today_close:
                pct_change = (today_close / tomorrow_close) * 100
            # elif tomorrow_close < today_close:
            #     pct_change = -1 * (tomorrow_close / today_close) * 100
            elif tomorrow_low < today_close:
                pct_change = -1 * (tomorrow_low / today_close) * 100
            else:
                pct_change = 0

            if pct_change > buy_threshold:
                df.iloc[i, df.columns.get_loc('correct_decision')] = 1
            elif pct_change < sell_threshold:
                df.iloc[i, df.columns.get_loc('correct_decision')] = -1  # SELL
            else:
                df.iloc[i, df.columns.get_loc('correct_decision')] = 0  # HOLD
    return df


def add_historic_indicators(df):
    """
    This is for the ML stuff. Since finance data is "contunious" (so-to-speak) it is important
    that each row contains not only the indicators for the day, but also previous days' information.
    """
    df_yesterday = None
    counter = 0
    total = len(df)
    for index, row in df.iterrows():
        if df_yesterday is None:
            df_yesterday = row
            counter += 1
            continue
        for column_and_val in row.iteritems():
            column = column_and_val[0]
            df.loc[index, 'yesterday_'+column] = df_yesterday[column]
        counter += 1
        print("Adding history information... Completed:", counter, "of", total)
        df_yesterday = row
    # Percent change from the day before
    df['pct_change_macd_diff'] = df['trend_macd_diff'].pct_change()*100
    # Percent change from the day before
    df['pct_change_momentum_rsi'] = df['momentum_rsi'].pct_change()*100
    df = df.replace([np.inf, -np.inf], 0)
    df.fillna(0, inplace=True)
    print('Finished adding historic information.')
    # df.to_csv('./TSX_dfs/messed.csv')
    return df


def add_pct_change(df):
    print("Adding percent change in adjusted close prices...")
    # Percent change from the day before
    df['pct_change'] = df['5. adjusted close'].pct_change()*100
    df = df.replace([np.inf, -np.inf], 0)
    df.fillna(0, inplace=True)
    return df


def generate_featuresets(df, train_size=0.9, random_state=None, shuffle=False, pca_components=5, today=False):
    from sklearn import preprocessing, decomposition
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    print("Generating feature set...")
    y = df['correct_decision'].values
    df = df.loc[:, df.columns != 'correct_decision']
    X = df[df.columns[0:]].values
    if today:
        # use the latest value
        x_test = X[-1:]
        x_train = X[:-1]

        y_test = None
        y_train = None
    else:
        # feature_names = ['pct_change','trend_macd_diff', 'pct_change_macd_diff', 'momentum_rsi', 'pct_change_momentum_rsi']
        # X = df[feature_names].values
        if train_size == 0:
            x_train = None
            y_train = None
            x_test = X
            y_test = y
            print("Normalizing feature set...")
            scaler = StandardScaler()
            x_test = scaler.fit_transform(x_test)
        else:
            # Ignore the most recent value, since we don't know what tomorrow will bring
            X = X[:-1]
            y = y[:-1]
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, random_state=random_state, shuffle=shuffle)
            print("Normalizing feature set...")
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

        # print("Generating PCA fit to reduce feature set to", n_components, "dimensions...")
        # pca = PCA(n_components=n_components)

        # print("Transforming with PCA...")
        # pca.fit(x_train)
        # x_train = pca.transform(x_train)
        # x_test = pca.transform(x_test)

        # principal_df = pd.DataFrame(pca.components_,columns=df[df.columns[0:]].columns)
        # principal_df.to_csv('./TSX_dfs/principals.csv')

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    # df = generate_adjclose_df("./tickers/ETFTickers.pickle")
    df = import_dataframe("temp3", "./tickers/TSXTickers.pickle")
    # df = add_indicators(df)
    # df = add_future_vision(df, 1, -1)
    # df = add_pct_change(df)
    # df = add_historic_indicators(df)
    # analyse_df(df, "2019-11-13")

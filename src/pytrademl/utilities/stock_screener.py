from pytrademl.utilities.alphavantage_utilities import get_data_from_alphaVantage
from pytrademl.utilities.ticker_utilities import obtain_tickers
from pytrademl.utilities.key_utilities import load_key

tickers = obtain_tickers("./tickers/TSXTickers.pickle")
key = load_key()
get_data_from_alphaVantage(
    key, reload=False, tickerFile="./tickers/TSXTickers.pickle")

from src.utilities.alphavantage_utilities import get_data_from_alphaVantage
from src.utilities.ticker_utilities import obtain_tickers

tickers = obtain_tickers("./tickers/TSXTickers.pickle")
key = "adsdasad"
get_data_from_alphaVantage(key, reload=False,tickerFile="./tickers/TSXTickers.pickle")
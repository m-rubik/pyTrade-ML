from src.utilities.alphavantage_utilities import get_data_from_alphaVantage
from src.utilities.ticker_utilities import obtain_tickers
from src.utilities.key_utilities import load_key

tickers = obtain_tickers("./tickers/TSXTickers.pickle")
key = load_key()
get_data_from_alphaVantage(key, reload=False,tickerFile="./tickers/TSXTickers.pickle")
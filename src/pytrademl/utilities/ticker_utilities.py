"""
TODO: For ETFs, it would be cool to add the sector weighting.
For example: VCN would have something like
# {"Financials": 0.362,
"Oil & Gas": 0.172,
etc
to show the percent holdings of each sector
"""

import pickle
import requests
import bs4 as bs
from pytrademl.utilities.object_utilities import export_object
from pathlib import Path

def obtain_all_tickers():
    ticker_file_list = ['ETFTickers', 'sp500Tickers', 'TSXTickers']
    for ticker_file in ticker_file_list:
        obtain_tickers(ticker_file)

def obtain_tickers(ticker_file="ETFTickers"):
    """
    TODO: Improve the method for obtaining ETFs
    """

    root_dir = Path(__file__).resolve().parent.parent / "tickers"
    root_dir.mkdir(parents=True, exist_ok=True)

    if 'ETF' in ticker_file:
        tickers = []
        # for i in range(1,59):
        #     print(i)
        #     resp = requests.get('https://etfdb.com/etfs/region/north-america/#etfs&sort_name=assets_under_management&sort_order=desc&page={}'.format(i))
        #     soup = bs.BeautifulSoup(resp.text, 'lxml')
        #     table = soup.find('tbody')
        #     for row in table.findAll('tr')[1:]:
        #         security = row.findAll('td')[0].text
        #         symbol = row.findAll('td')[1].text
        #         category = row.findAll('td')[3].text
        #         pair = [security,symbol,category]
        #         tickers.append(pair)
        tickers.append(['BMO AGGREGATE BOND INDEX ETF', 'ZAG', ''])
        tickers.append(['ISHARES CORE S&P U.S. TOTAL MARKEY INDEX ETF', 'XUU', ''])
        tickers.append(['ISHARES CORE MSCI EMG MKTS IMI ETF', 'XEC', ''])
        tickers.append([' ISHARES CORE SP TSX CAPD COM INX ETF', 'XIC', ''])
        tickers.append(['ISHARES CORE MSCI EAFE IMI INDEX ETF', 'XEF', ''])
        tickers.append(['VANGUARD ALL CAP INDEX ETF UNITS', 'VCN', ''])
        tickers.append(['VANGUARD CANADIAN AGGREGATE BOND INDEX ETF', 'VAB', ''])
        export_object(root_dir / "ETFTickers.pickle" , tickers)
        # export_tickers(tickers, ticker_file)
    elif 'sp500' in ticker_file:
        resp = requests.get(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            security = row.findAll('td')[0].text
            symbol = row.findAll('td')[1].text
            category = row.findAll('td')[3].text
            pair = [security, symbol, category]
            tickers.append(pair)
        export_object(root_dir / "sp500Tickers.pickle" , tickers)
    elif 'TSX' in ticker_file:
        tickers = []
        tickers.append(['GREEN ORGANIC DUTCHMAN HOLDINGS INC', 'TGOD', ''])
        tickers.append(['AURORA CANABIS INC', 'ACB', ''])
        tickers.append(['', 'HEXO', ''])
        tickers.append(['', 'WEED', ''])
        tickers.append(['', 'APHA', ''])
        tickers.append(['', 'OGI', ''])
        # export_tickers(tickers, ticker_file)
        export_object(root_dir / "TSXTickers.pickle" , tickers)
    else:
        print("Unrecognized ticker file:", ticker_file)
        tickers = []
    return tickers
 
if __name__ == "__main__":
    tickers = obtain_all_tickers()

import pickle
import requests
import bs4 as bs

def obtain_tickers(ticker_file="./tickers/ETFTickers.pickle"):
    if ticker_file == "./tickers/sp500tickers.pickle":
        resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class':'wikitable sortable'})
        tickers=[]
        for row in table.findAll('tr')[1:]:
            security = row.findAll('td')[0].text
            symbol = row.findAll('td')[1].text
            category = row.findAll('td')[3].text
            pair = [security,symbol,category]
            tickers.append(pair)
            
        export_tickers(tickers, ticker_file)
    elif ticker_file == "./tickers/ETFTickers.pickle":
    
        tickers=[]

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

        tickers.append(['BMO AGGREGATE BOND INDEX ETF','ZAG',''])
        tickers.append(['ISHARES CORE S&P U.S. TOTAL MARKEY INDEX ETF','XUU',''])
        tickers.append(['ISHARES CORE MSCI EMG MKTS IMI ETF','XEC',''])
        tickers.append([' ISHARES CORE SP TSX CAPD COM INX ETF','XIC',''])
        tickers.append(['ISHARES CORE MSCI EAFE IMI INDEX ETF','XEF',''])
        tickers.append(['VANGUARD ALL CAP INDEX ETF UNITS','VCN',''])

        export_tickers(tickers, ticker_file)
    else:
        print("Unrecognized ticker file:", ticker_file)
        tickers = []
    return tickers

def export_tickers(tickers, ticker_file="./tickers/ETFTickers.pickle"):
    with open(ticker_file,"wb") as f:
        pickle.dump(tickers,f)
    return 0

def import_tickers(ticker_file="./tickers/ETFTickers.pickle"):
    with open(ticker_file,"rb") as f:
        tickers = pickle.load(f)
    return tickers


if __name__ == "__main__":
    tickers = obtain_tickers()
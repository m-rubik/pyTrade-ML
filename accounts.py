"""
balance_in_cash is a little tricky.
It means the amount you've spent to buy the securities.
Take for example a new account. If you buy 1 share of something at 500$:
balance_in_cash = -500 because you've SPENT 500
balance_in_securities = 500 because you're holding that 500 in securities.. not in cash
So when you take your total balance, you have to do balance_in_cash + balance_in_securities.
Your earnings are then balance_in_securities - abs(balance_in_cash)
"""

import datetime


class Account():

	balance_in_cash: int = None
	balance_in_securities: int = None
	broker_fee: int = None
	securities: dict = {}

	def __init__(self, balance_in_cash=0,balance_in_securities=0, securities={}, broker_fee = 0):
		self.balance_in_securities = balance_in_securities
		self.securities = securities
		self.broker_fee = broker_fee
		self.balance_in_cash = balance_in_cash
	
	def sellSecurity(self,transaction_info):
		print(transaction_info)
		security = transaction_info['security']
		sellPrice = round(transaction_info['quantity'] * transaction_info['value'],2)
		if self.securities[security] >= transaction_info['quantity']: 
			self.securities[security] = self.securities[security] - transaction_info['quantity']
			self.balance_in_securities = round(self.securities[security] * transaction_info['value'],2)
			self.balance_in_cash = round(self.balance_in_cash + sellPrice,2) - self.broker_fee
			print('Selling',transaction_info['quantity'],'shares of',security,'for',str(sellPrice)+'$')
			print('balance_in_securities',self.balance_in_securities,'$')
			print('balance_in_cash',self.balance_in_cash,'$')
			return self  
		else:
			print('Unable to sell',security+'. Requested sell:',str(transaction_info['quantity'])+'. Held amount:',str(self.securities[security]))
			return self

	def buySecurity(self,transaction_info):
		security = transaction_info['security']
		buyCost = round(transaction_info['quantity'] * transaction_info['value'],2)
		if self.balance_in_cash >= buyCost:
			self.securities[security] = self.securities.get(security, 0) + transaction_info['quantity']
			self.balance_in_securities = round(self.securities[security] * transaction_info['value'],2)
			self.balance_in_cash = round(self.balance_in_cash - buyCost,2) - self.broker_fee
			print('Buying',transaction_info['quantity'],'shares of',security,'for',str(buyCost)+'$')
			print('balance_in_securities',self.balance_in_securities,'$')
			print('balance_in_cash',self.balance_in_cash,'$')
			return self  
		else:
			print('Unable to buy',transaction_info['quantity'], 'holdings of', security+'. Requested buy:',str(buyCost)+'$. Account balance_in_cash:',str(round(self.balance_in_cash,2))+'$')
			transaction_info['quantity'] = int(self.balance_in_cash / transaction_info['value']) # This automatically rounds down to nearest int
			self.buySecurity(transaction_info)
			return self

	def deposit(self,amount):
		self.balance_in_cash += amount
		
	def withdraw(self,amount):
		if self.balance_in_cash >= amount:
			self.balance_in_cash = self.balance_in_cash - amount
		else:
			print("Unable to withdraw", amount+"$. Current cash balance is", self.balance_in_cash+"$.")
	
	def holdUpdate(self,transaction_info):
		security = transaction_info['security']
		self.balance_in_securities = round(self.securities[security] * transaction_info['value'],2)

if __name__ == "__main__":

	import pandas as pd

	myAccount = Account(0,0,{},0)
	myAccount.deposit(500)
	
	today = datetime.datetime.now().strftime("%Y-%m-%d")
	print(today)

	ticker = "XIC"
	datafolder = './ETF_dfs/'
	tickerData = datafolder+ticker+'.csv'
	df = pd.read_csv(tickerData, parse_dates=True, index_col=0)

	print(df.loc[today, :])
	print(df.loc[today, '5. adjusted close'])

	transaction_info = {'security':ticker,'quantity':1,'value':df.loc[today, '5. adjusted close'].astype('float64').item()}
	myAccount.buySecurity(transaction_info)

	# print(df.head())

	df2  = pd.read_csv("./ETF_dfs/XICStupid.csv", parse_dates=True, index_col=0)

	df_merge = pd.concat([df,df2]).drop_duplicates()

	df_merge.to_csv('test.csv')

	# print(df_merge)


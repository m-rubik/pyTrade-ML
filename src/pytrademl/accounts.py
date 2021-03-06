"""
Contains all account-related functionality.

balance_in_cash is a little tricky.
It means the amount you've spent to buy the securities.
Take for example a new account. If you buy 1 share of something at 500$:
balance_in_cash = -500 because you've SPENT 500
balance_in_securities = 500 because you're holding that 500 in securities.. not in cash
So when you take your total balance, you have to do balance_in_cash + balance_in_securities.
Your earnings are then balance_in_securities - abs(balance_in_cash)
"""

import datetime
import pickle
import logging
import os
import shutil
import pandas as pd
from pytrademl.utilities.dataframe_utilities import import_dataframe


class Account():
    """
    Class that acts as a banking/investment portfolio/account.
    """

    balance_in_cash: int = 0
    balance_in_securities: int = 0
    broker_fee: int = 0
    securities: dict = {}
    balance: int = 0
    logger = None
    dataframes: dict = {}

    def __init__(self, balance_in_cash=0, balance_in_securities=0, securities=dict(), broker_fee=0, name="Default", max_cash_swing_pct=0.9):

        self.name = name
        self.securities = securities
        self.broker_fee = broker_fee
        self.max_cash_swing_pct = max_cash_swing_pct
        self.balance_in_securities = balance_in_securities
        self.balance_in_cash = balance_in_cash
        self.balance = self.balance_in_cash + self.balance_in_securities
        self.balance_history = {}
        self.balance_in_securities_history = {}
        self.balance_in_cash_history = {}
        self.securities_history = {}

        # create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        # create file handler
        transaction_records_folder = './accounts/'+self.name+"/"+'transaction_records/'
        if not os.path.exists(transaction_records_folder):
            os.makedirs(transaction_records_folder)
        fh = logging.FileHandler(transaction_records_folder+name+'.log')
        fh.setLevel(logging.DEBUG)
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s--%(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info("Opening new account.")

    def sellSecurity(self, transaction_info):
        """
        Method for attempting to sell holdings of a security.
        """

        security = transaction_info['security']
        if security not in self.securities:
            print("Cannot sell", security+". No holdings in this security.")
            # Update current balance
            self.update_balance()
            return self
        if transaction_info['quantity'] == "MAX":
            # Complete sell-off
            transaction_info['quantity'] = self.securities[security]
        sellPrice = round(
            transaction_info['quantity'] * transaction_info['value'], 2)
        if self.securities[security] >= transaction_info['quantity']:
            self.logger.info(transaction_info['date']+": Selling "+str(transaction_info['quantity'])+' shares of ' +
                             security+' for '+str(sellPrice)+'$. Price per share = '+str(transaction_info['value'])+"$")
            # Update current securities information
            self.securities[security] = self.securities.get(
                security, 0) - transaction_info['quantity']
            self.securities_history[transaction_info['date']
                                    ] = self.securities.copy()
            # Update current cash balance
            self.balance_in_cash = round(
                self.balance_in_cash + sellPrice - self.broker_fee, 2)
            # Update historic cash balance
            self.balance_in_cash_history[transaction_info['date']
                                         ] = self.balance_in_cash
            # Update current balance in securities
            self.update_balance_in_securities(transaction_info['date'])
            self.balance_in_securities_history[transaction_info['date']
                                               ] = self.balance_in_securities
            # Update current balance
            self.update_balance()
            self.balance_history[transaction_info['date']] = self.balance
            print('Balance in Securities', self.balance_in_securities, '$')
            print('Balance in Cash', self.balance_in_cash, '$')
        else:
            print('Unable to sell', security+'. Requested sell:',
                  str(transaction_info['quantity'])+'. Held amount:', str(self.securities[security]))
        save_account(self)
        return self

    def buySecurity(self, transaction_info):
        """
        Method for attempting to buy holdings of a security.
        """

        security = transaction_info['security']
        if transaction_info['quantity'] == "MAX":
            max_cash_buy = self.max_cash_swing_pct * self.balance_in_cash
            transaction_info['quantity'] = int(max_cash_buy/transaction_info['value'])

            # if self.balance_in_cash >= 10000:  # Never spend more than 10000 dollars
            #     # This automatically rounds down to nearest int
            #     transaction_info['quantity'] = int(
            #         10000/transaction_info['value'])
            # else:
            #     # This automatically rounds down to nearest int
            #     transaction_info['quantity'] = int(
            #         self.balance_in_cash / transaction_info['value'])
        
        buyCost = round(transaction_info['quantity']
                        * transaction_info['value'], 2)
        if self.balance_in_cash >= buyCost:
            self.logger.info(transaction_info['date']+": Buying "+str(transaction_info['quantity'])+' shares of ' +
                             security+' for '+str(buyCost)+'$. Price per share = '+str(transaction_info['value'])+"$")
            # Update current securities information
            self.securities[security] = self.securities.get(
                security, 0) + transaction_info['quantity']
            self.securities_history[transaction_info['date']
                                    ] = self.securities.copy()
            # Update current cash balance
            self.balance_in_cash = round(
                self.balance_in_cash - buyCost - self.broker_fee, 2)
            # Update historic cash balance
            self.balance_in_cash_history[transaction_info['date']
                                         ] = self.balance_in_cash
            # Update current balance in securities
            self.update_balance_in_securities(transaction_info['date'])
            self.balance_in_securities_history[transaction_info['date']
                                               ] = self.balance_in_securities
            # Update current balance
            self.update_balance()
            self.balance_history[transaction_info['date']] = self.balance
            print('Balance in Securities', self.balance_in_securities, '$')
            print('Balance in Cash', self.balance_in_cash, '$')
        else:
            print('Unable to buy', transaction_info['quantity'], 'holdings of', security+'. Requested buy:', str(
                buyCost)+'$. Account balance_in_cash:', str(round(self.balance_in_cash, 2))+'$')
            # This automatically rounds down to nearest int
            transaction_info['quantity'] = int(
                self.balance_in_cash / transaction_info['value'])
            self.buySecurity(transaction_info)
        save_account(self)
        return self

    def deposit(self, amount, date):
        """
        Method for depositing cash into the account.
        """

        self.balance_in_cash += amount
        self.balance_in_cash_history[date] = self.balance_in_cash
        self.update_balance()
        self.balance_history[date] = self.balance
        save_account(self)

    def withdraw(self, amount, date):
        """
        Method for withdrawing cash from the account.
        """

        if self.balance_in_cash >= amount:
            self.balance_in_cash = self.balance_in_cash - amount
            self.balance_in_cash_history[date] = self.balance_in_cash
            self.update_balance()
            self.balance_history[date] = self.balance
        else:
            print("Unable to withdraw", amount +
                  "$. Current cash balance is", self.balance_in_cash+"$.")
        save_account(self)

    def holdUpdate(self, transaction_info={}):
        """
        Method to update the history logs when the day's action
        is only to hold all positions.
        """

        self.logger.info(
            transaction_info['date']+": Holding all "+transaction_info['security'])
        self.update_balance_in_securities(today=transaction_info['date'])
        self.balance_in_cash_history[transaction_info['date']
                                     ] = self.balance_in_cash
        self.balance_in_securities_history[transaction_info['date']
                                           ] = self.balance_in_securities
        self.update_balance()
        self.balance_history[transaction_info['date']] = self.balance
        save_account(self)

    def update_balance_in_securities(self, today=None):
        """
        Method to update the current "value" of all
        of the securities held.
        """

        # Update dictionary containing all necessary df
        for security in self.securities.keys():
            if security not in self.dataframes.keys():
                self.dataframes[security] = import_dataframe(security, enhanced=True)

        if today is None:
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
        self.balance_in_securities = 0
        for security, quantity in self.securities.items():
            self.balance_in_securities += (self.dataframes[security].loc[today, "5. adjusted close"]) * quantity
        self.balance_in_securities = round(self.balance_in_securities, 2)
        save_account(self)

    def update_balance(self):
        """
        Method to update the total balance of the account.
        """

        self.balance = round(self.balance_in_securities +
                             self.balance_in_cash, 2)
        print("Account Value:", str(self.balance)+"$")
        save_account(self)


def save_account(account):
    """
    Function that serializes the account such
    that it can be saved.
    """

    root_dir = "./accounts/"+account.name+"/"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    with open(root_dir+account.name, "wb+") as f:
        pickle.dump(account, f)
    return 0


def open_account(name="Default", balance_in_cash=0, balance_in_securities=0, securities={}, broker_fee=0):
    """
    Function that opens an account.
    If the account doesn't exist, it creates a new one.
    """

    try:
        with open("./accounts/"+name+"/"+name, "rb+") as f:
            account = pickle.load(f)
            # create logger
            account.logger = logging.getLogger(name)
            account.logger.setLevel(logging.DEBUG)
            # create file handler
            transaction_records_folder = './accounts/' + \
                account.name+"/"+'transaction_records/'
            if not os.path.exists(transaction_records_folder):
                os.makedirs(transaction_records_folder)
            fh = logging.FileHandler(transaction_records_folder+name+'.log')
            fh.setLevel(logging.DEBUG)
            # create console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            # create formatter and add it to the handlers
            formatter = logging.Formatter(
                '%(asctime)s--%(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            # add the handlers to the logger
            account.logger.addHandler(fh)
            account.logger.addHandler(ch)
    except Exception:
        account = Account(balance_in_cash=balance_in_cash, balance_in_securities=balance_in_securities,
                          securities=securities, broker_fee=broker_fee, name=name)
    finally:
        save_account(account)
        return account


def delete_account(account):
    """
    Function for removing the account
    from system memory.
    """

    answer = None
    while answer not in ("yes", "no", "y", "n"):
        print("Deleting account named", account.name+".")
        answer = input("Please confirm [y/n]: ")
        if answer == "yes" or answer == "y":
            for handler in account.logger.handlers:
                handler.close()
                account.logger.removeHandler(handler)
            root_dir = "./accounts/"+account.name
            backup_dir = "./accounts/Backups/"+account.name
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(root_dir, backup_dir)
            if os.path.exists(root_dir):
                print("Removing directory", root_dir)
                shutil.rmtree(root_dir, ignore_errors=True)
            del account
        elif answer == "no" or answer == "n":
            print("Cancelled.")
        else:
            print("Please enter y/n")
    return 0

# TODO: Add earned dividends to total cash stack


if __name__ == "__main__":

    from pytrademl.utilities.plot_utilities import plot_account, plot_account_history

    myAccount = open_account("FutureVisionStrat", 0, 0, {}, 0)
    plot_account_history(myAccount)
    # myAccount = open_account("Test",0,0,{},0)
    # delete_account(myAccount)
    # myAccount = open_account("Test",0,0,{},0)
    pass

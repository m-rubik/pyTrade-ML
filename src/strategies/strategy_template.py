"""!
This contains the generic template for any trading strategy.
All custom trading strategies are to inherit this class and implement the run_strategy method.
"""

from abc import ABC, abstractmethod

class Strategy(ABC):

    buy_conditions = {}
    sell_conditions = {}
    hold_conditions = {}
    stop_loss = 0 # Initially, there is no stop loss. The stop loss is determined by the value of the first buy.

    def __init__(self, name=None):
        self.name = name

    def add_buy_condition(self, condition_name, condition):
        self.buy_conditions[condition_name] = condition

    def add_sell_condition(self, condition_name, condition):
        self.sell_conditions[condition_name] = condition

    def add_hold_condition(self, condition_name, condition):
        self.hold_conditions[condition_name] = condition

    @abstractmethod
    def run_strategy(self, df=None):
        """
        This is where the strategy is to be implemented.
        
        EXAMPLE:
        Iterate through a dataframe and for each day...
        1. Check HOLD conditions. If any are true, HOLD
        2. Check BUY conditions. If any are met, BUY
        3. Check SELL conditions. If any are met, SELL
        """

        


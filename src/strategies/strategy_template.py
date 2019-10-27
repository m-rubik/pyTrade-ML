"""
Generic template for any trading strategy.
Custom strategies will inherit this template.
"""


class Strategy():

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

    def run_strategy(self, df=None):
        """
        This is the default method of running a strategy.

        Iterate through a dataframe and for each day...
        1. Check HOLD conditions. If any are true, HOLD
        2. Check BUY conditions. If any are met, BUY
        3. Check SELL conditions. If any are met, SELL
        """
        raise NotImplementedError

        


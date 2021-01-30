from pytrademl.q_learning.agent.agent import QAgent
from pytrademl.q_learning.functions import get_q_state, formatPrice, getState
import sys
import pickle
import os
import pandas as pd
import pytrademl.utilities.dataframe_utilities as dataframe_utilities

class QLearner():
	def __init__(self, ticker, options):
		self.ticker = ticker
		self.options = options
		self.df = dataframe_utilities.import_dataframe(self.ticker, enhanced=True)
		# if "days_advance" in self.options.keys():
		# 	self.df = dataframe_utilities.add_future_vision(self.df, buy_threshold=0.5, sell_threshold=-0.5, days_advance=self.options["days_advance"])
		# else:
		# 	self.df = dataframe_utilities.add_future_vision(self.df)
		self.state_size = len(self.df.columns)

	def train(self, epoch_count=100, batch_size=32, resume_training=None):
		self.batch_size = batch_size
		self.epoch_count = epoch_count
		self.resume_training = resume_training

		if resume_training:
			self.agent = QAgent(self.state_size, model_name=self.resume_training)
			self.starting_epoch = int(self.resume_training.split("ep")[1]) + 1
			print("Resuming training at epoch", self.starting_epoch)
		else:
			self.agent = QAgent(self.state_size)
			self.starting_epoch = 0

		self.end_epoch = self.starting_epoch+self.epoch_count
		print("Running training from epoch", self.starting_epoch, "to", self.end_epoch)

		data = []
		for epoch in range(self.starting_epoch, self.end_epoch):
			self.epoch = epoch
			print("Running epoch " + str(self.epoch) + "/" + str(self.end_epoch) + "...")

			# self.state = getState(self.df, 0, self.state_size + 1)
			self.state = get_q_state(self.df.iloc[0])
			# print(self.state)

			total_profit = 0
			self.agent.inventory = []
			length = len(self.df)-1

			for day in range(len(self.df)-1):

				if day % 100 == 0:
					print("Processing day", day, "/", length)

				action = self.agent.act(self.state)

				# self.next_state = getState(self.df.iloc[0], day+1, self.state_size + 1)
				self.next_state = get_q_state(self.df.iloc[day+1])
				reward = 0

				if action == 1: # buy
					self.agent.inventory.append(self.df.iloc[day]['5. adjusted close'])
					# print("Buy: " + formatPrice(self.df.iloc[day]['5. adjusted close']))

				elif action == 2 and len(self.agent.inventory) > 0: # sell
					bought_price = self.agent.inventory.pop(0)
					# reward = max(self.df.iloc[day]['5. adjusted close'] - bought_price, 0)
					reward = self.df.iloc[day]['5. adjusted close'] - bought_price
					total_profit += self.df.iloc[day]['5. adjusted close'] - bought_price
					# print("Sell: " + formatPrice(self.df.iloc[day]['5. adjusted close']) + " | Profit: " + formatPrice(self.df.iloc[day]['5. adjusted close'] - bought_price))

				done = True if day == len(self.df)-1 else False
				self.agent.memory.append((self.state, action, reward, self.next_state, done))
				self.state = self.next_state

				if done:
					print("--------------------------------")
					print("Total Profit: " + formatPrice(total_profit))
					print("--------------------------------")

				if len(self.agent.memory) > self.batch_size:
					self.agent.expReplay(self.batch_size)
					# print("Total Profit: " + formatPrice(total_profit))
					data.append(total_profit)

					# import matplotlib.pyplot as plt
					# plt.plot(data)
					# plt.pause(0.05)

			self.agent.model.save("./src/pytrademl/q_learning/models/"+self.ticker+"_qmodel_ep" + str(self.epoch) + "/")

	def evaluate(self):
		pass

if __name__ == "__main__":
	learner = QLearner("XIC", {})
	learner.train(epoch_count=10, resume_training="XIC_qmodel_ep3")
	# learner.train(epoch_count=10, resume_training=None)

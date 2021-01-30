import numpy as np
import math
import pandas as pd

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	import os
	print(os.getcwd())
	lines = open("./src/q_learning/data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-1*abs(x)))

# returns an n-day state representation ending at time t
def getState(data, t, n):
	"""
	t = 0
	n = 10
	d = -10
	[1 ,2, 3, 4, 5, 6, 7, 8]
	block = 10 * [1] + 
	"""
	data = data.to_numpy()
	print(t)
	print(n)
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	print(block)
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))
	print(res)
	return np.array([res])

def get_q_state(day):
	"""
	In the current implementation,
	the state is simply the technical indicators of the day
	"""

	# data = day.to_numpy()
	# res = []
	# for i in range(len(data)-1):
	# 	res.append(sigmoid(data[i + 1] - data[i]))
	# print(res)
	# return np.array([res])
	# print(data)
	# print(data.shape)
	return day.values


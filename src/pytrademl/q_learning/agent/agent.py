"""
epsilon: balance of exploration v.s exploitation: https://stackoverflow.com/questions/53198503/epsilon-and-learning-rate-decay-in-epsilon-greedy-q-learning
gamma: https://stackoverflow.com/questions/1854659/alpha-and-gamma-parameters-in-qlearning

"""

import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class QAgent():
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		if self.model_name or self.is_eval:
			print("Loading model", model_name)
			self.model = load_model("./src/pytrademl/q_learning/models/" + model_name)
		else:
			self.model = self._model()

	def _model(self):
		model = Sequential()
		model.add(Dense(units=64, input_shape=(self.state_size,), activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))
		return model

	def act(self, state):
		if not self.is_eval and random.random() <= self.epsilon:
			return random.randrange(self.action_size) # Explore: select a random action
		
		state = state[None, :]
		options = self.model.predict(state)
		return np.argmax(options[0]) # Exploit: Select the action with the max future reward

	def expReplay(self, batch_size):
		"""
		The target is initally set to the current reward.
		If we are not done training yet, we predict the output of next_state and
		take the one that offers the highest future reward (the max of the array).
		Then, to "discount" the future reward, we apply gamma to the max.
		Then, we add the current reward to get the target reward.
		
		Next, we use the model to predict the output of the current state.
		Then, we index into the results with the action (0 hold, 1 buy, 2 sell) and
		modify the corresponding value to be the target obtained via the next_state prediction.

		Then, we re-train the model with the new target weighting so that in the future,
		when states appear that are similar to this one, it will be aware of the possible
		reward that can be achieved for a certain action, and update it's weightings
		accordingly.
		"""

		mini_batch = []
		l = len(self.memory)
		for i in range(l - batch_size + 1, l):
			mini_batch.append(self.memory[i])

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			state = state[None, :]
			next_state = next_state[None, :]
			if not done:
				res = self.model.predict(next_state)
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		self.memory = deque(maxlen=1000) # Reset memory for next batch

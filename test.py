import tensorflow as tf
print(tf.test.gpu_device_name())
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
print(tf.test.is_built_with_cuda())

def generate_lstm(self):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    import numpy as np

    self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1],1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1],1),activation='sigmoid'))
    model.add(LSTM(units=50,activation='sigmoid'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(self.X_train, self.y_train, epochs=10, batch_size=10, verbose=2)

    self.clf = model

    self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1],1))
    closing_price = model.predict(self.X_test)
    # closing_price = min_max_scaler.inverse_transform(closing_price)
    print(closing_price)
    self.test_model()
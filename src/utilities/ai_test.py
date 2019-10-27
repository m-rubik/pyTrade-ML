
# data_x: input data frame
# data_y: target variable (factor)
# GA parameters

import src.utilities.dataframe_utilities as dataframe_utilities
import numpy
import pickle

def sigmoid(inpt):
    return 1.0 / (1 + numpy.exp(-1 * inpt))

def relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    return result

def update_weights(weights, learning_rate):
    new_weights = weights - learning_rate * weights
    return new_weights

def train_network(num_iterations, weights, data_inputs, data_outputs, learning_rate, activation="relu"):
    for iteration in range(num_iterations):
        print("Iteration ", iteration)
        for sample_idx in range(data_inputs.shape[0]):
            r1 = data_inputs[sample_idx, :]
            for idx in range(len(weights) - 1):
                curr_weights = weights[idx]
                r1 = numpy.matmul(r1, curr_weights)
                if activation == "relu":
                    r1 = relu(r1)
                elif activation == "sigmoid":
                    r1 = sigmoid(r1)
            curr_weights = weights[-1]
            r1 = numpy.matmul(r1, curr_weights)
            predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
            desired_label = data_outputs[sample_idx]
            if predicted_label != desired_label:
                weights = update_weights(weights,
                                         learning_rate=0.001)
    return weights

def predict_outputs(weights, data_inputs, activation="relu"):
    predictions = numpy.zeros(shape=(data_inputs.shape[0]))
    for sample_idx in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        for curr_weights in weights:
            r1 = numpy.matmul(r1, curr_weights)
            if activation == "relu":
                r1 = relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
        predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
        predictions[sample_idx] = predicted_label
    return predictions

if __name__ == "__main__":
    df = dataframe_utilities.import_dataframe("XIC","./tickers/ETFTickers.pickle")
    data_inputs = df[['1. open', '3. low', '4. close']].values
    data_outputs = df['2. high'].values

    HL1_neurons = 150
    input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1,
                                            size=(data_inputs.shape[1], HL1_neurons))
    HL2_neurons = 60
    HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1,
                                        size=(HL1_neurons, HL2_neurons))
    output_neurons = 4
    HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1,
                                            size=(HL2_neurons, output_neurons))

    weights = numpy.array([input_HL1_weights,
                        HL1_HL2_weights,
                        HL2_output_weights])

    weights = train_network(num_iterations=10,
                            weights=weights,
                            data_inputs=data_inputs,
                            data_outputs=data_outputs,
                            learning_rate=0.01,
                            activation="relu")

    predictions = predict_outputs(weights, data_inputs)
    num_flase = numpy.where(predictions != data_outputs)[0]
    num_correct = numpy.where(predictions == data_outputs)[0]
    print("Number of false predictions", num_flase.size)
    print("Number of correct predictions", num_correct.size)
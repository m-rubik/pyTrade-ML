"""
This module contains all functions relating to the usage of Machine Learning.
Specifically, we explore classification algorithms (part of supervised learning).

Goal:
If we limit the decision scope to 3 simple options:
    1. Buy
    2. Sell
    3. Hold
then there is a 33% chance of randomly selecting the right option on any given day.
So, we are trying to develop a model that predicts with an accuracy ATLEAST >= 33%.

History of Achievements:
Date            Accuracy            Type                                Ticker
10-05-2019      54%                 Multilayer Perceptron (MLP)         XIC
05-25-2020      59%                 Multilayer Perceptron (MLP)         ACB
05-26-2020      70%                 Voting                              XUU
"""

import pickle
import os
import pandas as pd
import src.utilities.dataframe_utilities as dataframe_utilities
from src.utilities.plot_utilities import plot_confusion_matrix, plot_predictions
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


class ModelManager():
    """
    This class contains all functionalities for the generation
    and testing of machine learning models.
    """

    ticker: str
    model_type: str
    name: str
    options: dict
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, ticker, model_type, options):
        self.ticker = ticker
        self.model_type = model_type
        self.options = options
        self.name = self.model_type + "_" + self.ticker
        self.df = dataframe_utilities.import_dataframe(
            self.ticker, enhanced=True)
        if "days_advance" in self.options.keys():
            self.df = dataframe_utilities.add_future_vision(self.df, buy_threshold=0.5, sell_threshold=-0.5, days_advance=self.options["days_advance"])
        else:
            self.df = dataframe_utilities.add_future_vision(self.df)

        self.dispatch = {
            "mlp": self.generate_mlp,
            "adaboost": self.generate_adaboost,
            "voting": self.generate_voting,
            "bagging": self.generate_bagging,
        }

    def generate_model(self):
        """
        Method for generating a ML model
        by utilizing a dispatch table.
        """

        if "days_advance" in self.options.keys():
            self.X_train, self.X_test, self.y_train, self.y_test = dataframe_utilities.generate_featuresets(self.df, days_advance=self.options["days_advance"])
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = dataframe_utilities.generate_featuresets(self.df)
        self.dispatch[self.model_type]()

    def generate_mlp(self):
        """
        Method for generating a Multilayer perceptron
        Neural Network using Sklearn's MLP.
        """

        mlp = MLPClassifier(max_iter=1000, verbose=False)
        self.num_input_neurons = self.X_train[0].size
        self.num_output_neurons = 3  # Buy, sell or hold
        self.num_hidden_nodes = round(
            self.num_input_neurons*(2/3) + self.num_output_neurons)
        self.num_hn_perlayer = round(self.num_hidden_nodes/3)

        if self.options['gridsearch']:
            # Hyper-parameter optimization
            parameter_space = {
                'hidden_layer_sizes': [(self.num_hn_perlayer, self.num_hn_perlayer, self.num_hn_perlayer),
                                       (self.num_hidden_nodes,),
                                       (self.num_hn_perlayer, self.num_hn_perlayer, self.num_hn_perlayer, self.num_hn_perlayer)],
                'activation': ['tanh', 'relu', 'logistic'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive'],
            }
            # with warnings.catch_warnings():
            #     warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            print("Performing a gridsearch to find the optimal NN hyper-parameters.")
            self.clf = GridSearchCV(
                mlp, parameter_space, n_jobs=-1, cv=3, verbose=False)
            self.clf.fit(self.X_train, self.y_train)

            # Print results to console
            print('Best parameters found:\n', self.clf.best_params_)
            # print("Grid scores on development set:")
            # means = clf.cv_results_['mean_test_score']
            # stds = clf.cv_results_['std_test_score']
            # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        else:
            mlp.hidden_layer_sizes = (
                self.num_hn_perlayer, self.num_hn_perlayer, self.num_hn_perlayer)
            self.clf = mlp.fit(self.X_train, self.y_train)
        self.test_model()

    def generate_adaboost(self):
        """
        Method to generate an Adaboost
        classifier model.
        """

        self.clf = AdaBoostClassifier(n_estimators=1000)
        self.clf.fit(self.X_train, self.y_train)
        # scores = cross_val_score(self.clf, self.X_train, self.y_train, cv=5)
        # print(scores.mean())
        self.test_model()

    def generate_voting(self):
        """
        Method to generate a voting
        classifer model.
        """

        svc = LinearSVC(max_iter=10000)
        rfor = RandomForestClassifier(n_estimators=100)
        knn = KNeighborsClassifier()
        self.clf = VotingClassifier(
            [('lsvc', svc), ('knn', knn), ('rfor', rfor)])
        print("Training classifier...")
        for classifier, label in zip([svc, knn, rfor], ['lsvc', 'knn', 'rfor']):
            scores = cross_val_score(
                classifier, self.X_train, self.y_train, cv=5, scoring='accuracy')
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" %
                  (scores.mean(), scores.std(), label))
        self.clf.fit(self.X_train, self.y_train)
        self.test_model()

    def generate_bagging(self):
        """
        Method to generate a bagging
        classifer model.
        """

        self.clf = AdaBoostClassifier(
            DecisionTreeClassifier(), n_estimators=30)
        self.clf.fit(self.X_train, self.y_train)
        print(self.clf.score(self.X_test, self.y_test))
        self.test_model()

    def test_model(self, save_threshold=45):
        """
        Method for testing the performance of
        any given model.
        """

        print("Detailed classification report:")
        self.y_true, self.y_pred = self.y_test, self.clf.predict(self.X_test)
        print(classification_report(self.y_true, self.y_pred))

        self.confidence = metrics.r2_score(self.y_test, self.y_pred)
        print("R^2:", str(round(self.confidence*100, 2)) +
              "%. (Ideally as close to 0% as possible)")
        # NOTE: THESE CAN ONLY BE INTEGERS. YOU CANNOT TEST AGAINST FLOATS
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        model_accuracy = round(self.accuracy*100, 2)
        print("Accuracy:", str(model_accuracy) +
              "%. (Ideally as close to 100% as possible)")
        self.confusion_matrix = metrics.confusion_matrix(
            self.y_test, self.y_pred)

        # testIndex = self.df.shape[0]-len(self.y_pred)
        # test_df = self.df.reset_index()
        # test_df = test_df[testIndex:]
        # test_df['prediction'] = self.y_pred

        # print(sum(1 for val in y_pred if val == 1))
        # print(sum(1 for val in y_pred if val == -1))
        # print(len(y_pred))

        if model_accuracy >= save_threshold:
            self.name = self.name+"_"+str(int(round(self.accuracy*100, 0)))
            print("Saving model as", self.name)
            export_model(self)

    # def generate_lstm(self):
    #     from tensorflow.keras.models import Sequential
    #     from tensorflow.keras.layers import Dense, Dropout, LSTM
    #     import numpy as np

    #     self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1],1))

    #     # create and fit the LSTM network
    #     model = Sequential()
    #     model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1],1),activation='sigmoid'))
    #     model.add(LSTM(units=50,activation='sigmoid'))
    #     model.add(Dense(1,activation='sigmoid'))

    #     model.compile(loss='mean_squared_error', optimizer='adam')
    #     model.fit(self.X_train, self.y_train, epochs=10, batch_size=10, verbose=2)

    #     self.clf = model

    #     self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1],1))
    #     closing_price = model.predict(self.X_test)
    #     # closing_price = min_max_scaler.inverse_transform(closing_price)
    #     print(closing_price)
    #     self.test_model()

    def predict_today(self):
        """
        Method for predicting what action should be
        taken based on the most recent entry in the
        ticker's dataframe.
        """

        _, X_test, _, _ = dataframe_utilities.generate_featuresets(
            self.df, today=True)
        prediction = self.clf.predict(X_test)
        if prediction == -1:
            prediction = "SELL"
        elif prediction == 1:
            prediction = "BUY"
        elif prediction == 0:
            prediction = "HOLD"
        else:
            print("ERROR: Prediction is", prediction)
        print('Model prediction:', prediction)

        # prob = self.clf.predict_proba(X_test)
        # prob_per_class_dictionary = dict(zip(self.clf.classes_, prob))
        # if 1 in prob_per_class_dictionary.keys():
        #     print("BUY probability: ", str(round(prob_per_class_dictionary[1][0]*100,2))+"%")
        # if -1 in prob_per_class_dictionary.keys():
        #     print("SELL probability: ", str(round(prob_per_class_dictionary[-1][2]*100,2))+"%")
        # if 0 in prob_per_class_dictionary.keys():
        #     print("HOLD probability: ", str(round(prob_per_class_dictionary[0][1]*100,2))+"%")


def export_model(model_manager):
    """
    Function for exporting a model manager.
    """

    if not os.path.exists("./models/"):
        os.makedirs("./models/")
    model_file = "./models/"+model_manager.name+".pickle"
    with open(model_file, "wb+") as f:
        pickle.dump(model_manager, f)
    return 0


def import_model(model_name):
    """
    Function for loading a serialized model
    manager into runtime memory.
    """

    model_file = "./models/"+model_name+".pickle"
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    return model


if __name__ == "__main__":
    # AVAILABLE MODEL TYPES:
    # mlp, adaboost, voting, bagging

    ticker = "XIC"

    model_manager = ModelManager(ticker=ticker, model_type="mlp", options={'gridsearch': True, 'days_advance': 1})
    model_manager.generate_model()

    # model_manager = import_model("mlp_XIC_56")
    # model_manager.predict_today()

    plot_confusion_matrix(model_manager.confusion_matrix)
    plot_predictions(model_manager.y_pred, model_manager.y_test)

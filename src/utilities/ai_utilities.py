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
"""

import pickle
import os
import pandas as pd
import src.utilities.dataframe_utilities as dataframe_utilities
from src.utilities.plot_utilities import plot_confusion_matrix, plot_predictions
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn import svm, neighbors, metrics, preprocessing
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier


def export_model(model_manager):
    import os
    if not os.path.exists("./models/"):
	    os.makedirs("./models/")
    model_file = "./models/"+model_manager.name+".pickle"
    with open(model_file,"wb+") as f:
        pickle.dump(model_manager, f)
    return 0

def import_model(model_name):
    model_file = "./models/"+model_name+".pickle"
    with open(model_file,"rb") as f:
        model = pickle.load(f)
    return model

class ModelManager():

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
        self.df = dataframe_utilities.import_dataframe(self.ticker, enhanced=True)
        self.df = dataframe_utilities.add_future_vision(self.df, buy_threshold=1, sell_threshold=-1)

    def generate_model(self):
        self.X_train, self.X_test, self.y_train, self.y_test = dataframe_utilities.generate_featuresets(self.df)
        self.generate_mlp()

    def generate_mlp(self):
        """
        Generate a Multilayer perceptron Neural network using Sklearn's MLP.
        """

        mlp = MLPClassifier(max_iter=1000, verbose=False)
        self.num_input_neurons = self.X_train[0].size
        self.num_output_neurons = 3 # Buy, sell or hold
        self.num_hidden_nodes = round(self.num_input_neurons*(2/3) + self.num_output_neurons)
        self.num_hn_perlayer = round(self.num_hidden_nodes/3)

        if self.options['gridsearch']:
            # Hyper-parameter optimization
            parameter_space = {
                'hidden_layer_sizes': [(self.num_hn_perlayer,self.num_hn_perlayer,self.num_hn_perlayer), 
                                        (self.num_hidden_nodes,), 
                                        (self.num_hn_perlayer,self.num_hn_perlayer,self.num_hn_perlayer,self.num_hn_perlayer)],
                'activation': ['tanh', 'relu', 'logistic'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant','adaptive'],
            }
            # with warnings.catch_warnings():
            #     warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            print("Performing a gridsearch to find the optimal NN hyper-parameters.")
            self.clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=False)
            self.clf.fit(self.X_train, self.y_train)

            # Print results to console
            print('Best parameters found:\n', self.clf.best_params_)
            # print("Grid scores on development set:")
            # means = clf.cv_results_['mean_test_score']
            # stds = clf.cv_results_['std_test_score']
            # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        else:
            mlp.hidden_layer_sizes = (self.num_hn_perlayer,self.num_hn_perlayer,self.num_hn_perlayer)
            self.clf = mlp.fit(self.X_train, self.y_train)

    def test_model(self, save_threshold=50):
        print("Detailed classification report:")
        self.y_true, self.y_pred = self.y_test, self.clf.predict(self.X_test)
        print(classification_report(self.y_true, self.y_pred))

        self.confidence = metrics.r2_score(self.y_test, self.y_pred)
        print("R^2:", str(round(self.confidence*100,2))+"%. (Ideally as close to 0% as possible)")
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred) # NOTE: THESE CAN ONLY BE INTEGERS. YOU CANNOT TEST AGAINST FLOATS
        model_accuracy = round(self.accuracy*100,2)
        print("Accuracy:", str(model_accuracy)+"%. (Ideally as close to 100% as possible)")
        self.confusion_matrix = metrics.confusion_matrix(self.y_test, self.y_pred)

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

    def predict_today(self):
        _, X_test, _, _ = dataframe_utilities.generate_featuresets(self.df, today=True)
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


def run_lstm(df):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM

    print("Generating feature set...")
    # X = df[df.columns[9:]].values
    feature_names = ['pct_change']
    X = df[feature_names].values
    print("Normalizing feature set...")
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    # pca = decomposition.PCA(n_components=1)
    # print("Generating PCA fit to reduce feature set to",1, "dimensions...")
    # pca.fit(X)
    # print("Transforming with PCA...")
    # X = pca.transform(X)

    y = df['4. close'].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=0, shuffle=False)
    import numpy as np
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1),activation='sigmoid'))
    model.add(LSTM(units=50,activation='sigmoid'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=2)

    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    closing_price = model.predict(x_test)
    closing_price = min_max_scaler.inverse_transform(closing_price)
    return closing_price, y_test

def run_adaboost_classifier(x_train, x_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=1000)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print(scores.mean()) 

def run_voting_classifier(x_train, x_test, y_train, y_test):
    svc = svm.LinearSVC(max_iter=10000)
    rfor = RandomForestClassifier(n_estimators=100)
    knn = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc',svc),('knn',knn),('rfor',rfor)])
    print("Training classifier...")
    for clf, label in zip([svc, knn, rfor], ['lsvc', 'knn', 'rfor']):
        scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    clf.fit(x_train,y_train)
    print("Predicting with classifier...")
    predictions = clf.predict(x_test)
    export_model(clf, "voting_ACB")
    return predictions

def run_bagging_classifier(x_train, x_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
    bg = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=30)
    bg.fit(x_train, y_train)
    print(bg.score(x_test, y_test))


if __name__ == "__main__":

    ticker = "HEXO"

    # model_manager = ModelManager(ticker=ticker, model_type="mlp", options={'gridsearch': True})
    # model_manager.generate_model()
    # model_manager.test_model()

    model_manager = import_model("mlp_HEXO_51")
    model_manager.predict_today()

    # plot_confusion_matrix(model.confusion_matrix)
    # plot_predictions(model.y_pred, model.y_test)



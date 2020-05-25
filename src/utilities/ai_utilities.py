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

def export_model(model, model_name):
    import os
    if not os.path.exists("./models/"):
	    os.makedirs("./models/")
    model_file = "./models/"+model_name+".pickle"
    with open(model_file,"wb+") as f:
        pickle.dump(model,f)
    return 0

def import_model(model_name):
    model_file = "./models/"+model_name+".pickle"
    with open(model_file,"rb") as f:
        model = pickle.load(f)
    return model

def generate_featuresets(df, n_components=3, train_size=0.9, random_state=0, shuffle=False, today=False):
    from sklearn import preprocessing, decomposition

    print("Generating feature set...")
    if today:
        df = df.loc[:, df.columns != 'correct_decision']
        X = df[df.columns[0:]].values

        # Ignore the most recent value, since we don't know what tomorrow will bring
        x_test = X[-1:]
        x_train = X[:-1]

        y_test = None
        y_train = None
    else:
        y = df['correct_decision'].values
        df = df.loc[:, df.columns != 'correct_decision']
        X = df[df.columns[0:]].values

        # feature_names = ['pct_change','trend_macd_diff', 'pct_change_macd_diff', 'momentum_rsi', 'pct_change_momentum_rsi']
        # X = df[feature_names].values

        if train_size == 0:
            x_train = None
            y_train = None
            x_test = X
            y_test = y
            print("Normalizing feature set...")
            scaler = StandardScaler()
            x_test = scaler.fit_transform(x_test)
        else:
            # Ignore the most recent value, since we don't know what tomorrow will bring
            X = X[:-1]
            y = y[:-1]
            x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state, shuffle=shuffle)
            print("Normalizing feature set...")
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            
        # print("Generating PCA fit to reduce feature set to", n_components, "dimensions...")
        # pca = PCA(n_components=n_components)

        # print("Transforming with PCA...")
        # pca.fit(x_train)
        # x_train = pca.transform(x_train)
        # x_test = pca.transform(x_test)

        # principal_df = pd.DataFrame(pca.components_,columns=df[df.columns[0:]].columns)
        # principal_df.to_csv('./TSX_dfs/principals.csv')

    return x_train, x_test, y_train, y_test

def generate_mlp(x_train, x_test, y_train, y_test, gridsearch=True):
    """
    Generate a Multilayer perceptron Neural network using Sklearn's MLP.
    """

    mlp = MLPClassifier(max_iter=1000, verbose=False)
    num_input_neurons = x_train[0].size
    num_output_neurons = 3 # Buy, sell or hold
    num_hidden_nodes = round(num_input_neurons*(2/3) + num_output_neurons)
    num_hn_perlayer = round(num_hidden_nodes/3)

    if gridsearch:
        # Hyper-parameter optimization
        parameter_space = {
            'hidden_layer_sizes': [(num_hn_perlayer,num_hn_perlayer,num_hn_perlayer), (num_hidden_nodes,), (num_hn_perlayer,num_hn_perlayer,num_hn_perlayer,num_hn_perlayer)],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        print("Performing a gridsearch to find the optimal NN hyper-parameters.")
        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=False)
        clf.fit(x_train, y_train)

        # Print results to console
        print('Best parameters found:\n', clf.best_params_)
        # print("Grid scores on development set:")
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    else:
        mlp.hidden_layer_sizes = (num_hn_perlayer,num_hn_perlayer,num_hn_perlayer)
        clf = mlp.fit(x_train, y_train)
    return clf

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

def test_model(clf, df, x_test, y_test, name, save_threshold=50):
    print("Detailed classification report:")
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))

    testIndex = df.shape[0]-len(y_pred)
    df.reset_index(inplace=True)
    df = df[testIndex:]
    df['prediction'] = y_pred

    confidence = metrics.r2_score(y_test, y_pred)
    print("R^2:", str(round(confidence*100,2))+"%. (Ideally as close to 0% as possible)")
    accuracy = metrics.accuracy_score(y_test, y_pred) # NOTE: THESE CAN ONLY BE INTEGERS. YOU CANNOT TEST AGAINST FLOATS
    model_accuracy = round(accuracy*100,2)
    print("Accuracy:", str(model_accuracy)+"%. (Ideally as close to 100% as possible)")

    # print(sum(1 for val in y_pred if val == 1))
    # print(sum(1 for val in y_pred if val == -1))
    # print(len(y_pred))

    if model_accuracy >= save_threshold:
        model_name = name+"_"+str(int(round(accuracy*100, 0)))
        print("Saving model as", model_name)
        export_model(clf, model_name)

    return y_test, y_pred

def predict_today(ticker, model_name):
    # Load the dataframe object
    df = dataframe_utilities.import_dataframe(ticker,"./tickers/TSXTickers.pickle")
    df = dataframe_utilities.add_pct_change(df)
    df = dataframe_utilities.add_indicators(df)

    _, x_test, _, _ = generate_featuresets(df, n_components=5, today=True)

    model = import_model(model_name)
    print("Predicting using", model_name, "model")
    prediction = model.predict(x_test)
    if prediction == -1:
        prediction = "SELL"
    elif prediction == 1:
        prediction = "BUY"
    elif prediction == 0:
        prediction = "HOLD"
    else:
        print("ERROR: Prediction is", prediction)
    print('Model prediction:', prediction)

if __name__ == "__main__":
    import os

    # predict_today("ACB", "mlp_gridsearch_MLP_54")

    ## Load the dataframe object
    ticker = "HEXO"
    if os.path.exists('./TSX_dfs/'+ticker+"_enhanced.csv"):
        df = dataframe_utilities.import_dataframe(ticker+"_enhanced", "./tickers/TSXTickers.pickle")
    else:
        df = dataframe_utilities.import_dataframe(ticker,"./tickers/TSXTickers.pickle")
        df = dataframe_utilities.add_indicators(df)
        df = dataframe_utilities.add_historic_indicators(df)
        df = dataframe_utilities.add_pct_change(df)
        df.to_csv("./TSX_dfs/"+ticker+"_enhanced.csv")
    df = dataframe_utilities.add_future_vision(df, buy_threshold=1, sell_threshold=-1)
    x_train, x_test, y_train, y_test = generate_featuresets(df, n_components=5, today=False)

    ## To generate a MLP Neural Network
    clf = generate_mlp(x_train, x_test, y_train, y_test, gridsearch=True)

    ## To generate other classifiers...
    # predictions = run_voting_classifier(x_train, x_test, y_train, y_test)
    # run_adaboost_classifier(x_train, x_test, y_train, y_test)
    # predictions, y_test = run_lstm(df)

    ## Import a previously generated model
    # clf = import_model("mlp_ACB_59")

    y_test, y_pred = test_model(clf, df, x_test, y_test, name="mlp_"+ticker, save_threshold=45)
    conf = metrics.confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf)
    plot_predictions(y_pred, y_test)

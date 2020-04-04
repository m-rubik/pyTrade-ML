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
"""

import pickle
import pandas as pd
import src.utilities.dataframe_utilities as dataframe_utilities
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm, neighbors, metrics, preprocessing, neural_network
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

def plot_confusion_matrix(confusion_matrix):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix, cmap=plt.get_cmap('Blues'))
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    if confusion_matrix.shape[0] == 3:
        labels = ['Sell','Hold','Buy']
    else:
        labels = ['Sell','Buy']
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="black")
    fig.tight_layout()
    plt.show()

def generate_featuresets(df, n_components=3, train_size=0.9, random_state=0, shuffle=False, today=False):
    from sklearn import preprocessing, decomposition

    print("Generating feature set...")
    if today:
        # del df['correct_decision']
        df = df.loc[:, df.columns != 'correct_decision']
        X = df[df.columns[0:]].values
        x_test = X[-1:]
        x_train = X[:-1]
        y_test = None
        y_train = None
    else:
        y = df['correct_decision'].values

        df = df.loc[:, df.columns != 'correct_decision']
        # del df['correct_decision']
        X = df[df.columns[0:]].values

        # feature_names = ['pct_change','trend_macd_diff', 'pct_change_macd_diff', 'momentum_rsi', 'pct_change_momentum_rsi']
        # X = df[feature_names].values

        # Ignore the most recent value, since we don't know what tomorrow will bring
        X = X[:-1]
        y = y[:-1]

        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state, shuffle=shuffle)

        print("Normalizing feature set...")
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train = min_max_scaler.fit_transform(x_train)
        x_test = min_max_scaler.fit_transform(x_test)
        
        print("Generating PCA fit to reduce feature set to", n_components, "dimensions...")
        pca = decomposition.PCA(n_components=n_components)

        print("Transforming with PCA...")
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        # x_train = min_max_scaler.fit_transform(x_train)
        x_test = pca.transform(x_test)
        # x_test = min_max_scaler.fit_transform(x_test)

        principal_df = pd.DataFrame(pca.components_,columns=df[df.columns[0:]].columns)
        principal_df.to_csv('./TSX_dfs/principals.csv')

        df.to_csv('./TSX_dfs/temp3.csv')

    return x_train, x_test, y_train, y_test

def run_lstm(df):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from sklearn import preprocessing, decomposition

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

def run_multilayer_perceptron(x_train, x_test, y_train, y_test):
    mlp = neural_network.MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, verbose=True)
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        mlp.fit(x_train, y_train)
    print("Training set score: %f" % mlp.score(x_test, y_test))
    print("Training set loss: %f" % mlp.loss_)
    predictions = mlp.predict(x_test)
    return predictions

def run_gridsearch_mlp(x_train, x_test, y_train, y_test):
    mlp = neural_network.MLPClassifier(max_iter=1000, verbose=True)
    parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }
    from sklearn.model_selection import GridSearchCV
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(x_train, y_train)
    print('Best parameters found:\n', clf.best_params_)

    ## Print the other results that were found
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    y_true, predictions = y_test , clf.predict(x_test)
    from sklearn.metrics import classification_report
    print('Results on the test set:')
    print(classification_report(y_true, predictions))

    export_model(clf, "mlp_gridsearch_ACB")
    return predictions

def run_bagging_classifier(x_train, x_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
    bg = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=30)
    bg.fit(x_train, y_train)
    print(bg.score(x_test, y_test))

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

    # predict_today("ACB", "mlp_gridsearch_MLP_54")

    ## Load the dataframe object
    df = dataframe_utilities.import_dataframe("temp3","./tickers/TSXTickers.pickle")
    # df = dataframe_utilities.import_dataframe("XIC","./tickers/TSXTickers.pickle")
    df = dataframe_utilities.add_future_vision(df, buy_threshold=1, sell_threshold=-1)
    # df = dataframe_utilities.add_pct_change(df)
    # df = dataframe_utilities.add_indicators(df)
    # df = dataframe_utilities.add_historic_indicators(df)
    x_train, x_test, y_train, y_test = generate_featuresets(df, n_components=5, today=False)

    ## TO GENERATE OTHER CLASSIFIERS
    # predictions = run_voting_classifier(x_train, x_test, y_train, y_test)
    # run_adaboost_classifier(x_train, x_test, y_train, y_test)
    # predictions, y_test = run_lstm(df)

    ## TO GENERATE A NEW GRDISEARCH CLASSIFIER
    predictions = run_gridsearch_mlp(x_train, x_test, y_train, y_test)

    ## TO IMPORT A PRE-EXISTING CLASSIFIER
    # model = import_model("mlp_gridsearch_ACB")
    # predictions = model.predict(x_test)

    ## TO CHECK ACCURACY
    # from sklearn.metrics import classification_report
    # print('Results on the test set:')
    # print(classification_report(y_test, predictions))

    print("Classification complete.")
    testIndex = df.shape[0]-len(predictions)
    df.reset_index(inplace=True)
    df = df[testIndex:]
    df['prediction'] = predictions

    confidence = metrics.r2_score(y_test, predictions)
    print("R^2: ", confidence)
    accuracy = metrics.accuracy_score(y_test, predictions) # THESE CAN ONLY BE INTEGERS. YOU CANNOT TEST AGAINST FLOATS
    print("Accuracy:", str(round(accuracy*100,2))+"%")

    # print(sum(1 for val in predictions if val == 1))
    # print(sum(1 for val in predictions if val == -1))
    # print(len(predictions))

    conf = metrics.confusion_matrix(y_test, predictions)
    plot_confusion_matrix(conf)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(range(0,len(predictions)),predictions)
    ax.scatter(range(0,len(y_test)),y_test)
    plt.show()


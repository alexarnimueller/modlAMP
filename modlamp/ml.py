# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.ml

.. moduleauthor:: modlab Gisela Gabernet ETH Zurich <gisela.gabernet@pharma.ethz.ch>

This module contains different functions to facilitate machine learning mainly using the scikit-learn package.
Two models are available, whose parameters can be tuned. For more information of the machine learning modules please
check the scikit-learn documentation.

=============================    ==========================================================================================================================================
Model                            Reference
=============================    ==========================================================================================================================================
Support Vector Machine           `sklearn.svm.SVC <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
Random Forest                    `sklearn.ensemble.RandomForestClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
=============================    ==========================================================================================================================================

.. versionadded:: 2.2.0
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.svm import SVC

__author__ = "Alex Müller, Gisela Gabernet"
__docformat__ = "restructuredtext en"


def train_best_model(model, x_train, y_train, scaler=StandardScaler(), score=make_scorer(matthews_corrcoef),
                     param_grid=None, cv=10):
    """
    Returns pipeline that performs standard scaling and trains
    the best Support Vector Machine or Random forest classifier found by grid search.
    (see sklearn.preprocessing, sklearn.grid_search, sklearn.pipeline).

    Default parameter grids:

    ==============        ==============================================================================
    Model                 Parameter grid
    ==============        ==============================================================================
    SVM Model             param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
                          {'clf__C': param_range,    'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

    Random Forest         param_grid = [{'clf__n_estimators': [10, 50, 100, 500],
                          'clf__max_depth': [3, None],
                          'clf__max_features': [1, 2, 3, 5, 10],
                          'clf__min_samples_split': [1, 3, 5, 10],
                          'clf__min_samples_leaf': [1, 3, 5, 10],
                          'clf__bootstrap': [True, False],
                          'clf__criterion': ["gini", "entropy"]}]
    ==============        ==============================================================================



    Useful methods implemented in scikit-learn:

    =================            =============================================================
    Method                       Description
    =================            =============================================================
    fit(X,y)                     fit the model with the same parameters to new training data.
    score(X,y)                   get the score of the model for test data.
    predict(X)                   get predictions for new data.
    predict_proba(X)             get probability predicitons for [class0, class1]
    get_params()                 get parameters of the trained model
    =================            =============================================================

    :param model: {str} model to train. Choose between 'svm' (Support Vector Machine) or 'rf' (Random Forest).
    :param x_train: {array} descriptor values for training data.
    :param y_train: {array} class values for training data.
    :param scaler: {scaler} scaler to use in the pipe to scale data prior to training. Choose from sklearn.preprocessing.
                    E.g. StandardScaler(), MinMaxScaler(), Normalizer().
    :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
        (choose from `scoring-parameter <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_).
    :param param_grid: {dict} parameter grid for the gridsearch (see `sklearn.grid_search <http://scikit-learn.org/stable/modules/model_evaluation.html>`_).
    :param cv: {int} number of folds for cross-validation.
    :return: best estimator pipeline.

    :Example:

    >>> from modlamp.ml import train_best_model
    >>> from modlamp.datasets import load_ACPvsRandom
    >>> from modlamp import descriptors

    Loading a dataset for training.

    >>> data = load_ACPvsRandom()
    >>> len(data.sequences)
    826
    >>>list(data.target_names)
    ['Random', 'ACP']

    Calculating Pepcats descriptor in autocorrelation modality:

    >>> descr = descriptors.PeptideDescriptor(data.sequences,scalename='pepcats')
    >>> descr.calculate_autocorr(7)
    >>> descr.descriptor
    array([[ 1.        ,  0.15      ,  0.        , ...,  0.35714286,
         0.21428571,  0.        ],
       [ 0.64      ,  0.12      ,  0.32      , ...,  0.05263158,
         0.        ,  0.        ],
       [ 1.        ,  0.23809524,  0.        , ...,  0.53333333,
         0.26666667,  0.        ],
       ...,
       [ 0.5       ,  0.22222222,  0.44444444, ...,  0.33333333,
         0.        ,  0.        ],
       [ 0.70588235,  0.17647059,  0.23529412, ...,  0.09090909,
         0.09090909,  0.        ],
       [ 0.6875    ,  0.1875    ,  0.1875    , ...,  0.2       ,
         0.        ,  0.        ]])


    Training an SVM model with this data:

    >>> X_train = descr.descriptor
    >>> y_train = data.target
    >>> best_svm_model = train_best_model('svm', X_train, y_train)
    Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
    0.739995453978
    {'clf__gamma': 0.1, 'clf__C': 10.0, 'clf__kernel': 'rbf'}

    >>> best_svm_model.get_params()
    {'clf': SVC(C=10.0, cache_size=200, class_weight='balanced', coef0=0.0,
       decision_function_shape=None, degree=3, gamma=0.1, kernel='rbf',
       max_iter=-1, probability=True, random_state=1, shrinking=True, tol=0.001,
       verbose=False),
    ...
    'steps': [('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),
      ('clf', SVC(C=10.0, cache_size=200, class_weight='balanced', coef0=0.0,
         decision_function_shape=None, degree=3, gamma=0.1, kernel='rbf',
         max_iter=-1, probability=True, random_state=1, shrinking=True, tol=0.001,
         verbose=False))]}
    """
    print "performing grid search..."
    
    if model.lower() == 'svm':
        pipe_svc = Pipeline([('scl', scaler),
                             ('clf', SVC(class_weight='balanced', random_state=1, probability=True))])

        if param_grid is None:
            param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
            param_grid = [{'clf__C': param_range,
                           'clf__kernel': ['linear']},
                          {'clf__C': param_range,
                           'clf__gamma': param_range,
                           'clf__kernel': ['rbf']}]

        gs = GridSearchCV(estimator=pipe_svc,
                          param_grid=param_grid,
                          scoring=score,
                          cv=cv,
                          n_jobs=-1)

        gs.fit(x_train, y_train)

    elif model.lower() == 'rf':
        pipe_rf = Pipeline([('scl', scaler),
                            ('clf', RandomForestClassifier(random_state=1, class_weight='balanced'))])

        if param_grid is None:
            param_grid = [{'clf__n_estimators': [10, 100, 500],
                           'clf__max_features': ['sqrt', 'log2', None],
                           'clf__bootstrap': [True, False],
                           'clf__criterion': ["gini"]}]

        gs = GridSearchCV(estimator=pipe_rf,
                          param_grid=param_grid,
                          scoring=score,
                          cv=cv,
                          n_jobs=-1)

        gs.fit(x_train, y_train)

    else:
        print "Model not supported, please choose between 'svm' and 'rf'."

    print "Best score (scorer: %s) and parameters from a %d-fold cross validation:" % (score, cv)
    print gs.best_score_
    print gs.best_params_

    # Set the best parameters to the best estimator
    best_classifier = gs.best_estimator_
    return best_classifier.fit(x_train, y_train)


def plot_validation_curve(classifier, x_train, y_train, param_name,
                          param_range,
                          cv=10, score=make_scorer(matthews_corrcoef),
                          title="Validation Curve", xlab="parameter range", ylab="MCC", filename=None):
    """Plotting cross-validation curve for the specified classifier, training data and parameter.

    :param classifier: {classifier instance} classifier or validation curve (e.g. sklearn.svm.SVC).
    :param x_train: {array} descriptor values for training data.
    :param y_train: {array} class values for training data.
    :param param_name: {string} parameter to assess in the validation curve plot. For SVM,
        "clf__C" (C parameter), "clf__gamma" (gamma parameter). For Random Forest, "clf__n_estimators" (number of trees),
        "clf__max_depth" (max num of branches per tree, "clf__min_samples_split" (min number of samples required to split an
        internal tree node), "clf__min_samples_leaf" (min number of samples in newly created leaf).
    :param param_range: {list} parameter range for the validation curve.
    :param cv: {int} number of folds for cross-validation.
    :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
        `sklearn.model_evaluation.scoring-parameter <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_.
    :param title: {str} graph title
    :param xlab: {str} x axis label.
    :param ylab: {str} y axis label.
    :param filename: {str} if filename given the figure is stored in the specified path.
    :return: plot of the validation curve.

    :Example:

    >>> from modlamp.ml import train_best_model
    >>> from modlamp.datasets import load_ACPvsRandom
    >>> from modlamp import descriptors

    Loading a dataset for training.

    >>> data = load_ACPvsRandom()
    >>> len(data.sequences)
    826
    >>>list(data.target_names)
    ['Random', 'ACP']

    Calculating Pepcats descriptor in autocorrelation modality:

    >>> descr = descriptors.PeptideDescriptor(data.sequences,scalename='pepcats')
    >>> descr.calculate_autocorr(7)
    >>> descr.descriptor
    array([[ 1.        ,  0.15      ,  0.        , ...,  0.35714286,
         0.21428571,  0.        ],
       [ 0.64      ,  0.12      ,  0.32      , ...,  0.05263158,
         0.        ,  0.        ],
       [ 1.        ,  0.23809524,  0.        , ...,  0.53333333,
         0.26666667,  0.        ],
       ...,
       [ 0.5       ,  0.22222222,  0.44444444, ...,  0.33333333,
         0.        ,  0.        ],
       [ 0.70588235,  0.17647059,  0.23529412, ...,  0.09090909,
         0.09090909,  0.        ],
       [ 0.6875    ,  0.1875    ,  0.1875    , ...,  0.2       ,
         0.        ,  0.        ]])


    Training an SVM model with this data:

    >>> X_train = descr.descriptor
    >>> y_train = data.target
    >>> best_svm_model = train_best_model('svm', X_train, y_train)
    Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
    0.739995453978
    {'clf__gamma': 0.1, 'clf__C': 10.0, 'clf__kernel': 'rbf'}

    >>> plot_validation_curve(best_svm_model, X_train, y_train, param_name='clf__gamma', param_range=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])

    .. image:: ../docs/static/validation_curve.png
        :height: 300px

    """

    train_scores, test_scores = validation_curve(classifier, x_train, y_train, param_name, param_range,
                                                 cv=cv, scoring=score, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.clf()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.ylim(0.0, 1.1)
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="b")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="b")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def predict(classifier, x_test, seqs_test, names_test=None, y_test=np.array([]), filename=None, save_csv=True):
    """Returns pandas dataframe with predictions using the specified estimator and test data. If true class is provided,
    it returns the scoring value for the test data.

    :param classifier: {classifier instance} classifier used for predictions.
    :param x_test: {array} descriptor values for testing data.
    :param names_test: {list} names of the peptides in test data.
    :param seqs_test: {list} sequences of the peptides in test data.
    :param y_test: {array} true classes for testing data (optional).
    :param filename: {string} valid path for the file to store the predictions.
    :param save_csv: {bool} if true additionally saves csv file with predicitons.
    :return: pandas dataframe containing predictions for test data. Pred_prob_class0 and Pred_prob_class1
        are the predicted probability of the peptide belonging to class 0 and class 1, respectively.

    :Example:

    >>> from modlamp.ml import train_best_model, predict
    >>> from modlamp.datasets import load_ACPvsRandom
    >>> from modlamp import descriptors
    >>> from modlamp.sequences import Helices
    >>> data = load_ACPvsRandom()

    Calculating descriptor from the data
    >>> desc = descriptors.PeptideDescriptor(data.sequences, scalename='pepcats')
    >>> desc.calculate_autocorr(7)
    >>> best_svm_model = train_best_model('svm', desc.descriptor, data.target)

    Generating 10 de novo Helical sequences to predict their activity
    >>> H = Helices(seqnum=10, lenmin=7, lenmax=30)
    >>> H.generate_sequences()

    Calculating descriptor for the newly generated sequences
    >>> descH = descriptors.PeptideDescriptor(H.sequences, scalename='pepcats')
    >>> descH.calculate_autocorr(7)

    >>> df = predict(best_svm_model, x_test=descH.descriptor, seqs_test=H.sequences)
    >>> df.head(3)
       ID                   Sequence  Pred_prob_class0  Pred_prob_class1
    0   0  IAGKLAKVGLKIGKIGGKLVKGVLK          0.009167          0.990833
    1   1                LGVRVLRIIIR          0.007239          0.992761
    2   2              VGIRLARGVGRIG          0.071436          0.928564

    """
    if filename is None:
        filename = 'probability_predictions'

    pred_probs = classifier.predict_proba(x_test)

    if not (y_test.size and names_test):
        dictpred = {'ID': range(len(x_test)), 'Sequence': seqs_test,
                    'Pred_prob_class0': pred_probs[:, 0], 'Pred_prob_class1': pred_probs[:, 1]}
        dfpred = pd.DataFrame(dictpred, columns=['ID', 'Sequence', 'Pred_prob_class0', 'Pred_prob_class1'])

    elif not y_test.size:
        dictpred = {'ID': range(len(x_test)), 'Name': names_test, 'Sequence': seqs_test,
                    'Pred_prob_class0': pred_probs[:, 0], 'Pred_prob_class1': pred_probs[:, 1]}
        dfpred = pd.DataFrame(dictpred, columns=['ID', 'Name', 'Sequence', 'Pred_prob_class0', 'Pred_prob_class1'])

    elif not names_test:
        dictpred = {'ID': range(len(x_test)), 'Sequence': seqs_test,
                    'Pred_prob_class0': pred_probs[:, 0], 'Pred_prob_class1': pred_probs[:, 1],
                    'True_class': y_test}
        dfpred = pd.DataFrame(dictpred, columns=['ID', 'Sequence', 'Pred_prob_class0',
                                                 'Pred_prob_class1', 'True_class'])

    else:
        dictpred = {'ID': range(len(x_test)), 'Name': names_test, 'Sequence': seqs_test,
                    'Pred_prob_class0': pred_probs[:, 0], 'Pred_prob_class1': pred_probs[:, 1],
                    'True_class': y_test}
        dfpred = pd.DataFrame(dictpred, columns=['ID', 'Name', 'Sequence', 'Pred_prob_class0',
                                                 'Pred_prob_class1', 'True_class'])

    if save_csv:
        dfpred.to_csv(filename + time.strftime("-%Y%m%d-%H%M%S.csv"))

    return dfpred


def score_cv(classifier, X, y, cv=10, metrics=None, names=None):
    """ Returns the cross validation scores for the specified scoring metrics as a pandas data frame.

    :param classifier: {classifier instance} trained classifier used for predictions.
    :param X: {array} descriptor values for training data.
    :param y: {array} class values for training data.
    :param cv: {int} number of folds for cross-validation.
    :param metrics: {list} metrics to consider for calculating the cv_scores. Choose from
        `sklearn.metrics.scorers <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_.
    :param names: {list} names of the metrics to display on the dataframe.
    :return: pandas dataframe containing the cross validation scores for the specified metrics.
    :Example:

    >>> from modlamp.ml import train_best_model, score_cv
    >>> from modlamp.datasets import load_ACPvsRandom
    >>> from modlamp import descriptors
    >>> data = load_ACPvsRandom()

    Calculating descriptor from the data
    >>> desc = descriptors.PeptideDescriptor(data.sequences, scalename='pepcats')
    >>> desc.calculate_autocorr(7)
    >>> best_svm_model = train_best_model('svm', desc.descriptor, data.target)

    Cross validation scores
    >>> score_cv(best_svm_model, desc.descriptor, data.target, cv=5)
    ID   Metrics  Mean CV score     StDev
    0   accuracy       0.841199  0.051708
    1  precision       0.930872  0.024897
    2     recall       0.735763  0.093979
    3         f1       0.819435  0.064995
    4    roc_auc       0.914607  0.039345
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    means = []
    sd = []
    for metric in metrics:
        scores = cross_val_score(classifier, X, y, cv=cv, scoring=metric)
        means.append(scores.mean())
        sd.append(scores.std())

    if names is None:
        dict_scores = {'Metrics': metrics,
                       'Mean CV score': means,
                       'StDev': sd}
    else:
        dict_scores = {'Metrics': names,
                       'Mean CV score': means,
                       'StDev': sd}

    df_scores = pd.DataFrame(dict_scores)
    df_scores = df_scores[['Metrics', 'Mean CV score', 'StDev']]

    return df_scores


def score_testset(classifier, X_test, y_test):
    """ Returns the test set scores for the specified scoring metrics as a pandas data frame. The calculated metrics
    are Matthews correlation coefficient, accuracy, precision, recall, f1 and Area under the Receiver-Operator curve
    (roc_auc). See `sklearn.metrics <http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics>`_
    for more information.

    :param classifier: {classifier instance} trained classifier used for predictions.
    :param X_test: {array} descriptor values for the test data.
    :param y_test: {array} class values for the test data.
    :return: pandas dataframe containing the cross validation scores for the specified metrics.
    :Example:

    >>> from modlamp.ml import train_best_model, score_testset
    >>> from sklearn.model_selection import train_test_split
    >>> from modlamp.datasets import load_ACPvsRandom
    >>> from modlamp import descriptors
    >>> data = load_ACPvsRandom()

    Calculating descriptor from the data
    >>> desc = descriptors.PeptideDescriptor(data.sequences, scalename='pepcats')
    >>> desc.calculate_autocorr(7)

    Splitting into train and test sets
    >>> X_train, X_test, y_train, y_test = train_test_split(desc.descriptor, data.target, test_size = 0.33)

    Training an SVM model with the training set
    >>> best_svm_model = train_best_model('svm', X_train,y_train)

    Calculating the test set scores
    >>> score_testset(best_svm_model, X_test, y_test)
    ID  Metrics   Scores
    0        MCC  0.838751
    1   accuracy  0.919414
    2  precision  0.923664
    3     recall  0.909774
    4         f1  0.916667
    5    roc_auc  0.919173
    """

    metrics = ['MCC', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    scores = []

    MCC = matthews_corrcoef(y_test, classifier.predict(X_test))
    scores.append(MCC)

    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    scores.append(accuracy)

    precision = precision_score(y_test, classifier.predict(X_test))
    scores.append(precision)

    recall = recall_score(y_test, classifier.predict(X_test))
    scores.append(recall)

    f1 = f1_score(y_test, classifier.predict(X_test))
    scores.append(f1)

    roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
    scores.append(roc_auc)

    dict_scores = {'Metrics': metrics,
                   'Scores': scores}

    df_scores = pd.DataFrame(dict_scores)
    df_scores = df_scores[['Metrics', 'Scores']]

    return df_scores



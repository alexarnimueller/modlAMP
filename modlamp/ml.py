# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.ml

.. moduleauthor:: modlab Gisela Gabernet ETH Zurich <gisela.gabernet@pharma.ethz.ch>,
                  Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module contains different functions to facilitate machine learning with peptides, mainly making use of the
scikit-learn Python package. Two machine learning models are available, whose parameters can be tuned. For more
information on the machine learning modules please check the `scikit-learn documentation <http://scikit-learn.org>`_.

**The two available machine learning models in this module are:**

=========================   =======================================================================================
Model                       Reference :sup:`[1]`
=========================   =======================================================================================
Support Vector Machine      `sklearn.svm.SVC <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
Random Forest               `sklearn.ensemble.RandomForestClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
=========================   =======================================================================================

[1] F. Pedregosa *et al., J. Mach. Learn. Res.* **2011**, 12, 2825–2830.

**The following functions are included in this module:**

============================================    ========================================================================
Function name                                   Description
============================================    ========================================================================
:py:func:`modlamp.ml.train_best_model`          Performs a grid search on different model parameters and returns the
                                                best fitted model and its performance as the cross-validation MCC.
:py:func:`modlamp.ml.plot_validation_curve`     Plotting a validation curve for any parameter from the grid search.
:py:func:`modlamp.ml.predict`                   Predict the class labels or class probabilities for given peptides.
:py:func:`modlamp.ml.score_cv`                  Evaluate the performance of a given model through cross-validation.
:py:func:`modlamp.ml.score_testset`             Evaluate the performance of a given model through test-set prediction.
============================================    ========================================================================


.. versionadded:: 2.2.0
.. versionchanged: 2.7.8
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn import metrics as mets
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.svm import SVC
from sklearn.base import clone

__author__ = "Alex Müller, Gisela Gabernet"
__docformat__ = "restructuredtext en"


def train_best_model(model, x_train, y_train, sample_weights=None, scaler=StandardScaler(),
                     score=make_scorer(matthews_corrcoef), param_grid=None, n_jobs=-1, cv=10):
    """
    This function performs a parameter grid search on a selected classifier model and peptide training data set.
    It returns a scikit-learn pipeline that performs standard scaling and contains the best model found by the
    grid search according to the Matthews correlation coefficient.
    (see `sklearn.preprocessing <http://scikit-learn.org/stable/modules/preprocessing.html>`_, `sklearn.grid_search
    <http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search>`_, `sklearn.pipeline.Pipeline
    <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_).
    
    :param model: {str} model to train. Choose between ``'svm'`` (Support Vector Machine) or ``'rf'`` (Random Forest).
    :param x_train: {array} descriptor values for training data.
    :param y_train: {array} class values for training data.
    :param sample_weights: {array} sample weights for training data.
    :param scaler: {scaler} scaler to use in the pipe to scale data prior to training. Choose from
        ``sklearn.preprocessing``, e.g. ``StandardScaler()``, ``MinMaxScaler()``, ``Normalizer()``.
    :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
        (choose from the scikit-learn
        `scoring-parameters <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_).
    :param param_grid: {dict} parameter grid for the gridsearch (see
        `sklearn.grid_search <http://scikit-learn.org/stable/modules/model_evaluation.html>`_).
    :param n_jobs: {int} number of parallel jobs to use for calculation. if ``-1``, all available cores are used.
    :param cv: {int} number of folds for cross-validation.
    :return: best estimator pipeline.


    **Default parameter grids:**

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


    **Useful methods implemented in scikit-learn:**

    =================            =============================================================
    Method                       Description
    =================            =============================================================
    fit(X, y)                    fit the model with the same parameters to new training data.
    score(X, y)                  get the score of the model for test data.
    predict(X)                   get predictions for new data.
    predict_proba(X)             get probability predictions for [class0, class1]
    get_params()                 get parameters of the trained model
    =================            =============================================================

    :Example:

    >>> from modlamp.ml import train_best_model
    >>> from modlamp.datasets import load_ACPvsRandom
    >>> from modlamp.descriptors import PeptideDescriptor

    Loading a dataset for training:

    >>> data = load_ACPvsRandom()
    >>> len(data.sequences)
    826
    >>>list(data.target_names)
    ['Random', 'ACP']

    Calculating the pepCATS descriptor values in auto-correlation modality:

    >>> descr = PeptideDescriptor(data.sequences, scalename='pepcats')
    >>> descr.calculate_autocorr(7)
    >>> descr.descriptor.shape
    (826, 42)
    >>> descr.descriptor
    array([[ 1.    ,  0.15      ,  0.        , ...,  0.35714286,  0.21428571,  0.        ],
       [ 0.64      ,  0.12      ,  0.32      , ...,  0.05263158,  0.        ,  0.        ],
       [ 1.        ,  0.23809524,  0.        , ...,  0.53333333,  0.26666667,  0.        ],
       ...,
       [ 0.5       ,  0.22222222,  0.44444444, ...,  0.33333333,  0.        ,  0.        ],
       [ 0.70588235,  0.17647059,  0.23529412, ...,  0.09090909,  0.09090909,  0.        ],
       [ 0.6875    ,  0.1875    ,  0.1875    , ...,  0.2       ,  0.        ,  0.        ]])

    Training an SVM model on this descriptor data:

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
    print("performing grid search...")
    
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
                          fit_params={'clf__sample_weight': sample_weights},
                          scoring=score,
                          cv=cv,
                          n_jobs=n_jobs)
        
        gs.fit(x_train, y_train)
        print("Best score (scorer: %s) and parameters from a %d-fold cross validation:" % (score, cv))
        print("MCC score:\t%.3f" % gs.best_score_)
        print("Parameters:\t%s" % gs.best_params_)
        
        # Set the best parameters to the best estimator
        best_classifier = gs.best_estimator_
        return best_classifier.fit(x_train, y_train)
    
    elif model.lower() == 'rf':
        pipe_rf = Pipeline([('scl', scaler),
                            ('clf', RandomForestClassifier(random_state=1, class_weight='balanced'))])
        
        if param_grid is None:
            param_grid = [{'clf__n_estimators': [10, 100, 500],
                           'clf__max_features': ['sqrt', 'log2'],
                           'clf__bootstrap': [True],
                           'clf__criterion': ["gini"]}]
        
        gs = GridSearchCV(estimator=pipe_rf,
                          param_grid=param_grid,
                          fit_params={'clf__sample_weight': sample_weights},
                          scoring=score,
                          cv=cv,
                          n_jobs=n_jobs)
        
        gs.fit(x_train, y_train)
        print("Best score (scorer: %s) and parameters from a %d-fold cross validation:" % (score, cv))
        print("MCC score:\t%.3f" % gs.best_score_)
        print("Parameters:\t%s" % gs.best_params_)
        
        # Set the best parameters to the best estimator
        best_classifier = gs.best_estimator_
        return best_classifier.fit(x_train, y_train)
    
    else:
        print("Model not supported, please choose between 'svm' and 'rf'.")


def plot_validation_curve(classifier, x_train, y_train, param_name, param_range, cv=10, score=make_scorer(
        matthews_corrcoef), title="Validation Curve", xlab="parameter range", ylab="MCC", n_jobs=-1, filename=None):
    """This function plots a cross-validation curve for the specified classifier on all tested parameters given in the
    option ``param_range``.

    :param classifier: {classifier instance} classifier or validation curve (e.g. sklearn.svm.SVC).
    :param x_train: {array} descriptor values for training data.
    :param y_train: {array} class values for training data.
    :param param_name: {string} parameter to assess in the validation curve plot. For SVM,
        "clf__C" (C parameter), "clf__gamma" (gamma parameter). For Random Forest, "clf__n_estimators" (number of trees)
        "clf__max_depth" (max num of branches per tree, "clf__min_samples_split" (min number of samples required to
        split an internal tree node), "clf__min_samples_leaf" (min number of samples in newly created leaf).
    :param param_range: {list} parameter range for the validation curve.
    :param cv: {int} number of folds for cross-validation.
    :param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
        `sklearn.model_evaluation.scoring-parameter
        <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_.
    :param title: {str} graph title
    :param xlab: {str} x axis label.
    :param ylab: {str} y axis label.
    :param n_jobs: {int} number of parallel jobs to use for calculation. if ``-1``, all available cores are used.
    :param filename: {str} if filename given the figure is stored in the specified path.
    :return: plot of the validation curve.

    :Example:

    >>> from modlamp.ml import train_best_model
    >>> from modlamp.datasets import load_ACPvsRandom
    >>> from modlamp.descriptors import PeptideDescriptor

    Loading a dataset for training:

    >>> data = load_ACPvsRandom()
    >>> len(data.sequences)
    826
    >>>list(data.target_names)
    ['Random', 'ACP']

    Calculating the pepCATS descriptor values in auto-correlation modality:

    >>> descr = PeptideDescriptor(data.sequences, scalename='pepcats')
    >>> descr.calculate_autocorr(7)
    >>> descr.descriptor.shape
    (826, 42)
    >>> descr.descriptor
    array([[ 1.    ,  0.15      ,  0.        , ...,  0.35714286,  0.21428571,  0.        ],
       [ 0.64      ,  0.12      ,  0.32      , ...,  0.05263158,  0.        ,  0.        ],
       [ 1.        ,  0.23809524,  0.        , ...,  0.53333333,  0.26666667,  0.        ],
       ...,
       [ 0.5       ,  0.22222222,  0.44444444, ...,  0.33333333,  0.        ,  0.        ],
       [ 0.70588235,  0.17647059,  0.23529412, ...,  0.09090909,  0.09090909,  0.        ],
       [ 0.6875    ,  0.1875    ,  0.1875    , ...,  0.2       ,  0.        ,  0.        ]])

    Training an SVM model with this data:

    >>> X_train = descr.descriptor
    >>> y_train = data.target
    >>> best_svm_model = train_best_model('svm', X_train, y_train)
    Best score (scorer: make_scorer(matthews_corrcoef)) and parameters from a 10-fold cross validation:
    0.739995453978
    {'clf__gamma': 0.1, 'clf__C': 10.0, 'clf__kernel': 'rbf'}

    >>> plot_validation_curve(best_svm_model, X_train, y_train, param_name='clf__gamma',
                              param_range=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])

    .. image:: ../docs/static/validation_curve.png
        :height: 300px

    """
    train_scores, test_scores = validation_curve(classifier, x_train, y_train, param_name, param_range,
                                                 cv=cv, scoring=score, n_jobs=n_jobs)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # plotting
    plt.clf()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.ylim(0.0, 1.1)
    plt.semilogx(param_range, train_mean, label="Training score", color="b")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="b")
    plt.semilogx(param_range, test_mean, label="Cross-validation score", color="g")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def predict(classifier, X, seqs, names=None, y=None, filename=None):
    """This function can be used to predict novel peptides with a trained classifier model. The function returns a
    ``pandas.DataFrame`` with predictions using the specified estimator and test data. If true class is provided,
    it returns the scoring value for the test data.

    :param classifier: {classifier instance} classifier used for predictions.
    :param X: {array} descriptor values of the peptides to be predicted.
    :param seqs: {list} sequences of the peptides in ``X``.
    :param names: {list} (optional) names of the peptides in ``X``.
    :param y: {array} (optional) true (known) classes of the peptides.
    :param filename: {string} (optional) output filename to store the predictions to (``.csv`` format); if ``None``:
        not saved.
    :return: ``pandas.DataFrame`` containing predictions for ``X``. ``P_class0`` and ``P_class1``
        are the predicted probability of the peptide belonging to class 0 and class 1, respectively.

    :Example:

    >>> from modlamp.ml import train_best_model, predict
    >>> from modlamp.datasets import load_ACPvsRandom
    >>> from modlamp.descriptors import PeptideDescriptor
    >>> from modlamp.sequences import Helices
    
    Loading data for model training:
    
    >>> data = load_ACPvsRandom()

    Calculating descriptor values from the data:
    
    >>> desc = PeptideDescriptor(data.sequences, scalename='pepcats')
    >>> desc.calculate_autocorr(7)
    >>> best_svm_model = train_best_model('svm', desc.descriptor, data.target)

    Generating 10 *de novo* helical sequences to predict their activity:
    
    >>> H = Helices(seqnum=10, lenmin=7, lenmax=30)
    >>> H.generate_sequences()

    Calculating descriptor values for the newly generated sequences:
    
    >>> descH = PeptideDescriptor(H.sequences, scalename='pepcats')
    >>> descH.calculate_autocorr(7)

    >>> df = predict(best_svm_model, X=descH.descriptor, seqs=descH.sequences)
    >>> df.head(3)  # all three shown sequences are predicted active (class 1)
                     Sequence       P_class0        P_class1
    IAGKLAKVGLKIGKIGGKLVKGVLK       0.009167        0.990833
                  LGVRVLRIIIR       0.007239        0.992761
                VGIRLARGVGRIG       0.071436        0.928564

    """
    preds = classifier.predict_proba(X)
    
    if not (y and names):
        d_pred = {'P_class0': preds[:, 0], 'P_class1': preds[:, 1]}
        df_pred = pd.DataFrame(d_pred, index=seqs)
    
    elif not y:
        d_pred = {'Name': names, 'P_class0': preds[:, 0], 'P_class1': preds[:, 1]}
        df_pred = pd.DataFrame(d_pred, index=seqs)
    
    elif not names:
        d_pred = {'P_class0': preds[:, 0], 'P_class1': preds[:, 1], 'True_class': y}
        df_pred = pd.DataFrame(d_pred, index=seqs)
    
    else:
        d_pred = {'Name': names, 'P_class0': preds[:, 0], 'P_class1': preds[:, 1], 'True_class': y}
        df_pred = pd.DataFrame(d_pred, index=seqs)
    
    if filename:
        df_pred.to_csv(filename + time.strftime("-%Y%m%desc-%H%M%S.csv"))
    
    return df_pred


def score_cv(classifier, X, y, sample_weights=None, cv=10, shuffle=True):
    """This function can be used to evaluate the performance of selected classifier model. It returns the average
    **cross-validation scores** for the specified scoring metrics in a ``pandas.DataFrame``.

    :param classifier: {classifier instance} a classifier model to be evaluated.
    :param X: {array} descriptor values for training data.
    :param y: {array} class values for training data.
    :param sample_weights: {array} weights for training data.
    :param cv: {int} number of folds for cross-validation.
    :param shuffle: {bool} suffle data before making the K-fold split.
    :return: ``pandas.DataFrame`` containing the cross validation scores for the specified metrics.
    :Example:

    >>> from modlamp.ml import train_best_model, score_cv
    >>> from modlamp.datasets import load_ACPvsRandom
    >>> from modlamp.descriptors import PeptideDescriptor
    
    Loading data for model training:
    
    >>> data = load_ACPvsRandom()

    Calculating descriptor values from the data:

    >>> desc = PeptideDescriptor(data.sequences, scalename='pepcats')
    >>> desc.calculate_autocorr(7)
    >>> best_svm_model = train_best_model('svm', desc.descriptor, data.target)

    Get the cross-validation scores:
    
    >>> score_cv(best_svm_model, desc.descriptor, data.target, cv=5)
                   CV_0   CV_1   CV_2   CV_3   CV_4   mean    std
        MCC        0.785  0.904  0.788  0.757  0.735  0.794  0.059
        accuracy   0.892  0.952  0.892  0.880  0.867  0.896  0.029
        precision  0.927  0.974  0.953  0.842  0.884  0.916  0.048
        recall     0.864  0.925  0.854  0.889  0.864  0.879  0.026
        f1         0.894  0.949  0.901  0.865  0.874  0.896  0.029
        roc_auc    0.893  0.951  0.899  0.881  0.868  0.898  0.028
    """

    cv_names = []
    for i in range(cv):
        cv_names.append("CV_%i" % i)

    cv_scores = []

    funcs = ['matthews_corrcoef', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'roc_auc_score',
             'confusion_matrix']
    metrics = ['MCC', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
               'TN', 'FP', 'FN', 'TP', 'FDR', 'sensitivity', 'specificity']

    kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=shuffle)
    clf = clone(classifier)

    for fold_train_index, fold_test_index in kf.split(X, y):
        Xcv_train, Xcv_test = X[fold_train_index], X[fold_test_index]
        ycv_train, ycv_test = y[fold_train_index], y[fold_test_index]
        scores = []
        if sample_weights is not None:
            weightcv_train, weightcv_test = sample_weights[fold_train_index], sample_weights[fold_test_index]
            clf.fit(Xcv_train, ycv_train, sample_weight=weightcv_train)
            for f in funcs:
                scores.append(getattr(mets, f)(ycv_test, clf.predict(Xcv_test), sample_weight=weightcv_test))
            tn, fp, fn, tp = scores.pop().ravel()
            scores = scores + [tn, fp, fn, tp]
            fdr = float(fp) / (tp + fp)
            scores.append(fdr)
            sn = float(tp) / (tp + fn)
            scores.append(sn)
            sp = float(tn) / (tn + fp)
            scores.append(sp)
        else:
            clf.fit(Xcv_train, ycv_train)
            for f in funcs:
                scores.append(getattr(mets, f)(ycv_test, clf.predict(Xcv_test)))
            tn, fp, fn, tp = scores.pop().ravel()
            scores = scores + [tn, fp, fn, tp]
            fdr = float(fp) / (tp + fp)
            scores.append(fdr)
            sn = float(tp) / (tp + fn)
            scores.append(sn)
            sp = float(tn) / (tn + fp)
            scores.append(sp)

        cv_scores.append(scores)

    dict_scores = dict()
    for colname, score in zip(cv_names, cv_scores):
        dict_scores.update({colname: score})

    df_scores = pd.DataFrame(dict_scores, index=metrics)

    df_scores['mean'] = df_scores.mean(axis=1)
    df_scores['std'] = df_scores.std(axis=1)

    return df_scores.round(2)


def score_testset(classifier, x_test, y_test, sample_weights=None):
    """ Returns the test set scores for the specified scoring metrics in a ``pandas.DataFrame``. The calculated metrics
    are Matthews correlation coefficient, accuracy, precision, recall, f1 and area under the Receiver-Operator Curve
    (roc_auc). See `sklearn.metrics <http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics>`_
    for more information.

    :param classifier: {classifier instance} pre-trained classifier used for predictions.
    :param x_test: {array} descriptor values of the test data.
    :param y_test: {array} true class values of the test data.
    :param sample_weights: {array} weights for the test data.
    :return: ``pandas.DataFrame`` containing the cross validation scores for the specified metrics.
    :Example:

    >>> from modlamp.ml import train_best_model, score_testset
    >>> from sklearn.model_selection import train_test_split
    >>> from modlamp.datasets import load_ACPvsRandom
    >>> from modlamp import descriptors
    >>> data = load_ACPvsRandom()

    Calculating descriptor values from the data
    
    >>> desc = descriptors.PeptideDescriptor(data.sequences, scalename='pepcats')
    >>> desc.calculate_autocorr(7)

    Splitting data into train and test sets
    
    >>> X_train, X_test, y_train, y_test = train_test_split(desc.descriptor, data.target, test_size = 0.33)

    Training a SVM model on the training set
    
    >>> best_svm_model = train_best_model('svm', X_train,y_train)

    Calculating the scores of the predictions on the test set
    
    >>> score_testset(best_svm_model, X_test, y_test)
       Metrics   Scores
           MCC  0.839
      accuracy  0.920
     precision  0.924
        recall  0.910
            f1  0.917
       roc_auc  0.919
    """
    scores = []
    metrics = ['MCC', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
               'TN', 'FP', 'FN', 'TP', 'FDR', 'sensitivity', 'specificity']
    funcs = ['matthews_corrcoef', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'roc_auc_score',
             'confusion_matrix']
    
    for f in funcs:
        scores.append(getattr(mets, f)(y_test, classifier.predict(x_test), sample_weight=sample_weights))  # fore every metric, calculate the scores
    
    tn, fp, fn, tp = scores.pop().ravel()
    scores = scores + [tn, fp, fn, tp]
    fdr = float(fp) / (tp + fp)
    scores.append(fdr)
    sn = float(tp) / (tp + fn)
    scores.append(sn)
    sp = float(tn) / (tn + fp)
    scores.append(sp)
    df_scores = pd.DataFrame({'Scores': scores}, index=metrics)
    
    return df_scores.round(2)

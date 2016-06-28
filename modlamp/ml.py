# -*- coding: utf-8 -*-
"""
.. module:: modlamp.ml

.. moduleauthor:: modlab Gisela Gabernet ETH Zurich <gisela.gabernet@pharma.ethz.ch>

This module contains different functions to facilitate machine learning mainly using the scikit-learn package.
Two models are available, whose parameters can be tuned. For more information of the machine learning modules please
check the scikit-learn documentation.

=============================	=============================================================================================
Model  							Reference
=============================	=============================================================================================
Support Vector Machine          http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
Random Forest                   http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
=============================	=============================================================================================

.. versionadded:: 2.2.0
"""

import numpy as np
from sklearn.preprocessing import *
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import validation_curve
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import time
import pandas as pd

__author__ = "modlab"
__docformat__ = "restructuredtext en"

def train_best_model(model, x_train, y_train, scaler=StandardScaler(), score=make_scorer(matthews_corrcoef),
					param_grid=None, cv=10):
	"""
	Returns pipeline that performs standard scaling and trains
	the best Support Vector Machine or Random forest classifier found by grid search.
	(see sklearn.preprocessing, sklearn.grid_search, sklearn.pipeline).

	Default parameter grids:

	==============		============================================================================
	Model				Parameter grid
	==============		============================================================================
	SVM Model			param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
						{'clf__C': param_range,	'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

	Random Forest		param_grid = [{'clf__n_estimators': [10, 50, 100, 500],
						'clf__max_depth': [3, None],
						'clf__max_features': [1, 2, 3, 5, 10],
						'clf__min_samples_split': [1, 3, 5, 10],
						'clf__min_samples_leaf': [1, 3, 5, 10],
						'clf__bootstrap': [True, False],
						'clf__criterion': ["gini", "entropy"]}]
	==============		============================================================================



	Useful methods implemented in scikit-learn:

	=================			=============================================================
	Method						Description
	=================			=============================================================
	fit(X,y) 					fit the model with the same parameters to new training data.
	score(X,y) 					get the score of the model for test data.
	predict(X) 					get predictions for new data.
	predict_proba(X)				get probability predicitons for [class0, class1]
	get_params()					get parameters of the trained model
	=================			=============================================================

	:param model: {str} model to train. Choose between 'svm' (Support Vector Machine) or 'rf' (Random Forest).
	:param x_train: {array} descriptor values for training data.
	:param y_train: {array} class values for training data.
	:param scaler: {scaler} scaler to use in the pipe to scale data prior to training. Choose from sklearn.preprocessing.
					E.g. StandardScaler(), MinMaxScaler(), Normalizer().
	:param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
		(choose from http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
	:param param_grid: {dict} parameter grid for the gridsearch (see sklearn.grid_search).
	:param cv: {int} number of folds for cross-validation.
	:return: best estimator pipeline.

	:Example:

	>>> from modlamp.ml import train_best_model
	>>> from modlamp.datasets import load_ACPvsNeg
	>>> from modlamp import descriptors

	Loading a dataset for training.

	>>> data = load_ACPvsNeg()
	>>> data.sequences[188]
	'FLFKLIPKAIKGLVKAIRK'
	>>> data.target[188]
	'1'
	>>> data.target_names
	array(['Neg', 'ACP'], dtype='|S3')

	Calculating Pepcats descriptor in autocorrelation:

	>>> descr = descriptors.PeptideDescriptor(data.sequences,scalename='pepcats')
	>>> descr.calculate_autocorr(7)
	>>> descr.descriptor[0]
	array([ 0.77777778,  0.11111111,  0.22222222,  0.16666667,  0.        ,
        0.        ,  0.52941176,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.5625    ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.6       ,  0.        ,
        0.13333333,  0.        ,  0.        ,  0.        ,  0.71428571,
        0.        ,  0.07142857,  0.07142857,  0.        ,  0.        ,
        0.53846154,  0.07692308,  0.        ,  0.        ,  0.        ,
        0.        ,  0.66666667,  0.        ,  0.08333333,  0.08333333,
        0.        ,  0.        ])

	Training an SVM model with this data:

	>>> X_train = descr.descriptor
	>>> y_train = data.target
	>>> best_svm_model = train_best_model('svm', X_train, y_train)
	Best score and parameters from a 10-fold cross validation:
	mean: 0.86932, std: 0.10581, params: {'clf__gamma': 0.001, 'clf__C': 100.0, 'clf__kernel': 'rbf'}

	>>> best_svm_model.get_params()
	{'clf': SVC(C=100.0, cache_size=200, class_weight='balanced', coef0=0.0,
	   decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
	   max_iter=-1, probability=True, random_state=1, shrinking=True, tol=0.001,
	   verbose=False),
	 'clf__C': 100.0,
	 'clf__cache_size': 200,
	 'clf__class_weight': None,
	 'clf__coef0': 0.0,
	 'clf__decision_function_shape': None,
	 'clf__degree': 3,
	 'clf__gamma': 0.001,
	 'clf__kernel': 'rbf',
	 'clf__max_iter': -1,
	 'clf__probability': True,
	 'clf__random_state': 1,
	 'clf__shrinking': True,
	 'clf__tol': 0.001,
	 'clf__verbose': False,
	 'scl': StandardScaler(copy=True, with_mean=True, with_std=True),
	 'scl__copy': True,
	 'scl__with_mean': True,
	 'scl__with_std': True,
	 'steps': [('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),
	  ('clf', SVC(C=100.0, cache_size=200, class_weight='balanced', coef0=0.0,
		 decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
		 max_iter=-1, probability=True, random_state=1, shrinking=True, tol=0.001,
		 verbose=False))]}

	"""
	if model == 'svm':

		pipe_svc = Pipeline([('scl', scaler),
							('clf', SVC(class_weight='balanced', random_state=1, probability=True))])

		if param_grid is None:
			param_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
			param_grid = [{'clf__C': param_range,
						'clf__kernel': ['linear']},
						{'clf__C': param_range,
						'clf__gamma': param_range,
						'clf__kernel': ['rbf']}]

		gs = GridSearchCV(estimator=pipe_svc,
						param_grid=param_grid,
						scoring=score,
						cv=cv,
						n_jobs=1)

		gs.fit(x_train, y_train)

	elif model == 'rf':

		pipe_rf = Pipeline([('scl', scaler),
							('clf', RandomForestClassifier(random_state=1, class_weight='balanced'))])

		if param_grid is None:
			param_grid = [{'clf__n_estimators': [10, 50, 100, 500],
						'clf__max_depth': [3, None],
						'clf__max_features': [1, 2, 3, 5, 10],
						'clf__min_samples_split': [1, 3, 5, 10],
						'clf__min_samples_leaf': [1, 3, 5, 10],
						'clf__bootstrap': [True, False],
						'clf__criterion': ["gini", "entropy"]}]

		gs = GridSearchCV(estimator=pipe_rf,
						param_grid=param_grid,
						scoring=score,
						cv=cv,
						n_jobs=1)

		gs.fit(x_train, y_train)

	else:
		print "Model not supported, please choose between 'svm' and 'rf'."

	print "Best score and parameters from a %d-fold cross validation:" % cv
	for row in range(len(gs.grid_scores_)):
		if gs.grid_scores_[row][0] == gs.best_params_:
			print gs.grid_scores_[row]
			print "\n"

	# Set the best parameters to the best estimator
	best_classifier = gs.best_estimator_
	return best_classifier.fit(x_train, y_train)


def plot_validation_curve(classifier, x_train, y_train, param_name,
					param_range=None,
					cv=10, score=make_scorer(matthews_corrcoef),
					title="Validation Curve", xlab="parameter range", ylab="MCC"):
	"""Plotting cross-validation curve for the specified classifier, training data and parameter.

	:param classifier: {classifier instance} classifier or validation curve (e.g. sklearn.svm.SVC).
	:param x_train: {array} descriptor values for training data.
	:param y_train: {array} class values for training data.
	:param param_name: {string} parameter to assess in the validation curve plot. For SVM,
		"clf__C" (C parameter), "clf__gamma" (gamma parameter). For RF, "clf__n_estimators" (number of trees),
		"clf__max_depth" (max num of branches per tree, "clf__min_samples_split" (min number of samples required to split an
		internal tree node), "clf__min_samples_leaf" (min number of samples in newly created leaf).
	:param param_range: {list} parameter range for the validation curve.
	:param cv: {int} number of folds for cross-validation.
	:param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
		(choose from http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
	:param title: {str} graph title
	:param xlab: {str} x axis label.
	:param ylab: {str} y axis label.
	:return: plot of the validation curve.

	"""

	if param_range is None:
		param_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

	train_scores, test_scores = validation_curve(classifier, x_train, y_train, param_name, param_range,
								cv=cv, scoring=score, n_jobs=1)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.title(title)
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	plt.ylim(0.0, 1.1)
	plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
	plt.fill_between(param_range, train_scores_mean - train_scores_std,
					train_scores_mean + train_scores_std, alpha=0.2, color="r")
	plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
				color="g")
	plt.fill_between(param_range, test_scores_mean - test_scores_std,
					test_scores_mean + test_scores_std, alpha=0.2, color="g")
	plt.legend(loc="best")
	plt.show()


def df_predictions(classifier, x_test, seqs_test, names_test=None, y_test=None, filename=None, save_csv=True):
	"""	Returns pandas dataframe with predictions using the specified estimator and test data. If true class is provided,
	it returns the scoring value for the test data.

	:param classifier: {classifier instance} classifier used for predictions.
	:param x_test: {array} descriptor values for testing data.
	:param names_test: {list} names of the peptides in test data.
	:param seqs_test: {list} sequences of the peptides in test data.
	:param y_test: {array} true classes for testing data (optional).
	:param filename: {string} valid path for the file to store the predictions.
	:param save_csv: {bool} if true additionally saves csv file with predicitons.
	:return: pandas dataframe containing predictions for test data. Pred_prob_class0 and Pred_prob_class1
		are the predicted probability of the peptide belonging to class0 and class1, respectively.

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


def cv_scores(classifier, X, y, cv=10, metrics=None):
	""" Returns the cross validation scores for the specified scoring metrics as a pandas data frame.

	:param classifier: {classifier instance} classifier used for predictions.
	:param X: {array} descriptor values for training data.
	:param y: {array} class values for training data.
	:param cv: {int} number of folds for cross-validation.
	:param metrics: {list} metrics to consider for calculating the cv_scores. Choose from sklearn.metrics.scorers
					(http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
	:return: pandas dataframe containing the cross validation scores for the specified metrics.


	"""
	if metrics is None:
		metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

	means = []
	sd = []
	for metric in metrics:
		scores = cross_val_score(classifier, X, y, cv=cv, scoring=metric)
		means.append(scores.mean())
		sd.append(scores.std())

	dict_scores = {'Metrics' : metrics,
				   'Mean CV score': means,
				   'StDev': sd}

	df_scores = pd.DataFrame(dict_scores)
	df_scores = df_scores[['Metrics', 'Mean CV score', 'StDev']]

	return df_scores
# -*- coding: utf-8 -*-
"""
.. module:: modlamp.ml

.. moduleauthor:: modlab Gisela Gabernet ETH Zurich <gisela.gabernet@pharma.ethz.ch>

This module contains different functions to facilitate machine learning mainly using the scikit-learn package.
Two models are available, whose parameters can be tuned. For more information of the machine learning modules please
check the scikit-learn documentation.

=============================		=============================================================================================
Model  								Reference
=============================		=============================================================================================
Support Vector Machine              http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
Random Forest                       http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
=============================		=============================================================================================

"""

import numpy as np
from sklearn.preprocessing import *
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt
import time
import pandas as pd

__author__ = "modlab"
__docformat__ = "restructuredtext en"

def train_best_model(model, x_train, y_train, scaler=StandardScaler(), score=make_scorer(matthews_corrcoef),
					param_grid=None, param_range=None, cv=10):
	"""
	Returns estimator pipeline that performs standard scaling and trains
	the best Support Vector Machine or Random forest classifier found by grid search.
	(see sklearn.preprocessing, sklearn.grid_search, sklearn.pipeline).
	:param model: {str} model to train. Choose between 'svm' (Support Vector Machine) or 'rf' (Random Forest).
	:param x_train: {array} descriptor values for training data.
	:param y_train: {array} class values for training data.
	:param scaler: {scaler} scaler to use in the pipe to scale data prior to training. Choose from sklearn.preprocessing.
					E.g. StandardScaler(), MinMaxScaler(), Normalizer().
	:param score: {metrics instance} scoring function built from make_scorer() or a predefined value in string form
		(choose from http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
	:param param_grid: {dict} parameter grid for the gridsearch (see sklearn.grid_search).
	:param param_range: {list} parameter range for the parameter grid.
	:param cv: {int} number of folds for cross-validation.
	:return: best estimator pipeline.
	"""
	if model == 'svm':

		pipe_svc = Pipeline([('scl', scaler),
							('clf', SVC(random_state=1, probability=True))])

		if param_range is None:
			param_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

		if param_grid is None:
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
							('clf', RandomForestClassifier(random_state=1))])

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


def validation_curve(classifier, x_train, y_train, param_name,
					param_range=None,
					cv=10, score=make_scorer(matthews_corrcoef),
					title="Validation Curve", xlab="parameter range", ylab="MCC"):
	"""
	Plotting cross-validation curve for the specified classifier, training data and parameter.
	:param classifier: {classifier instance} classifier or validation curve (e.g. sklearn.svm.SVC).
	:param x_train: {array} descriptor values for training data.
	:param y_train: {array} class values for training data.
	:param param_name: {string} parameter to assess in the validation curve plot. For SVM,
		"clf__C" (C parameter), "clf__gamma" (gamma parameter). For RF, "n_estimators" (number of trees), "max_depth" (max num of
		branches per tree, "min_samples_split" (min number of samples required to split an internal tree node),
		"min_samples_leaf" (min number of samples in newly created leaf).
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


def predictions(classifier, x_test, names_test, seqs_test, y_test=None, filename=None):
	"""
	Returns predictions using the specified estimator and test data. If true class is provided,
	it returns the scoring value for the test data.
	:param classifier: {classifier instance} classifier used for predictions.
	:param x_test: {array} descriptor values for testing data.
	:param names_test: {list} names of the peptides in test data.
	:param seqs_test: {list} sequences of the peptides in test data.
	:param y_test: {array} true classes for testing data (optional).
	:param filename: {string} valid path for the file to store the predictions.
	:return: csv file containing predictions for test data. Pred_prob_class0 and Pred_prob_class1
		are the predicted probability of the peptide belonging to class0 and class1, respectively.
	"""

	if filename is None:
		filename = 'probability_predictions'

	pred_probs = classifier.predict_proba(x_test)

	if y_test is None:
		dictpred = {'ID': range(len(x_test)), 'Name': names_test, 'Sequence': seqs_test,
					'Pred_prob_class0': pred_probs[:, 0], 'Pred_prob_class1': pred_probs[:, 1]}
		dfpred = pd.DataFrame(dictpred, columns=['ID', 'Name', 'Sequence', 'Pred_prob_class0', 'Pred_prob_class1'])
	else:
		dictpred = {'ID': range(len(x_test)), 'Name': names_test, 'Sequence': seqs_test,
					'Pred_prob_class0': pred_probs[:, 0], 'Pred_prob_class1': pred_probs[:, 1],
					'True_class': y_test}
		dfpred = pd.DataFrame(dictpred, columns=['ID', 'Name', 'Sequence', 'Pred_prob_class0',
												'Pred_prob_class1', 'True_class'])

	dfpred.to_csv(filename + time.strftime("-%Y%m%d-%H%M%S.csv"))
	return dfpred

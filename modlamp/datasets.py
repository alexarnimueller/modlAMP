# -*- coding: utf-8 -*-
"""
.. module:: modlamp.datasets

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to load different peptide datasets used for classification.
"""

from os.path import dirname
from os.path import join
import csv
import numpy as np

__author__ = 'modlab'

class Bunch(dict):
	"""Container object for datasets

	Dictionary-like object that exposes its keys as attributes. Taken from the ``sklearn`` package.

	:Example:

	>>> b = Bunch(a=1, b=2)
	>>> b['b']
	2
	>>> b.b
	2
	>>> b.a = 3
	>>> b['a']
	3
	>>> b.c = 6
	>>> b['c']
	6
	"""

	def __init__(self, **kwargs):
		dict.__init__(self, kwargs)

	def __setattr__(self, key, value):
		self[key] = value

	def __getattr__(self, key):
		try:
			return self[key]
		except KeyError:
			raise AttributeError(key)

	def __setstate__(self, state):
		# Bunch pickles generated with scikit-learn 0.16.* have an non
		# empty __dict__. This causes a surprising behaviour when
		# loading these pickles scikit-learn 0.17: reading bunch.key
		# uses __dict__ but assigning to bunch.key use __setattr__ and
		# only changes bunch['key']. More details can be found at:
		# https://github.com/scikit-learn/scikit-learn/issues/6196.
		# Overriding __setstate__ to be a noop has the effect of
		# ignoring the pickled __dict__
		pass


def load_AMPvsTM():
	"""Function to load a dataset consisting of AMP sequences and transmembrane regions of proteins for classification.

	The AMP class consists of an intersection of all activity annotations of the `APD2 <http://aps.unmc.edu/AP/>`_ and
	`CAMP <http://camp.bicnirrh.res.in/>`_ databases, where for gram positive, gram negative and antifungal exact
	matches were observed. A detailed description of how the dataset was compiled can be found in the following
	publication: *Schneider P. et al. 2016, Mol. Inf.*

	=================	====
	Classes				2
	Samples per class	206
	Samples total		412
	Dimensionality		1
	=================	====

	:return: Bunch, a dictionary-like object, the interesting attributes are: ``data``, the data to learn, ``target``, the
		classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the features.
	:Example:

	>>> from modlamp.datasets import load_AMPvsTM
	>>> data = load_AMPvsTM()
	>>> data.sequences[:5]
	array([['AAGAATVLLVIVLLAGSYLAVLA'],['LWIVIACLACVGSAAALTLRA'],['FYRFYMLREGTAVPAVWFSIELIFGLFA'],['GTLELGVDYGRAN'],['KLFWRAVVAEFLATTLFVFISIGSALGFK']])
	>>> list(data.target_names)
	['TM', 'AMP']
	>>> data.sequences.shape
	(412, 1)
	"""

	module_path = dirname(__file__)
	with open(join(module_path, 'data', 'AMPvsTM.csv')) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		n_samples = int(temp[0])
		n_features = int(temp[1])
		target_names = np.array(temp[2:])
		sequences = np.empty((n_samples,n_features), dtype='|S100')
		target = np.empty((n_samples,), dtype=np.int)

		for i, ir in enumerate(data_file):
			sequences[i] = ir[0]
			target[i] = ir[-1]

	return Bunch(sequences=sequences, target=target,
				 target_names=target_names,
				 feature_names=['Sequence'])


#TODO: add more data set loading method, e.g helical vs AMPs
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

__author__ = "modlab"
__docformat__ = "restructuredtext en"


class Bunch(dict):
	"""Container object for datasets

	Dictionary-like object that exposes its keys as attributes. Taken from the ``sklearn`` package.

	:Example:

	>>> b = Bunch(a=1, b=2)
	>>> b['b']
	2
	>>> b.b  # key can also be called as attribute
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
		pass


def load_AMPvsTMset():
	"""Function to load a dataset consisting of **AMP sequences and transmembrane regions of proteins** for classification.

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

	:return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``, the
		classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the features.
	:Example:

	>>> from modlamp.datasets import load_AMPvsTMset
	>>> data = load_AMPvsTMset()
	>>> data.sequences[:5]
	array([['AAGAATVLLVIVLLAGSYLAVLA'],['LWIVIACLACVGSAAALTLRA'],['FYRFYMLREGTAVPAVWFSIELIFGLFA'],['GTLELGVDYGRAN'],['KLFWRAVVAEFLATTLFVFISIGSALGFK']])
	>>> list(data.target_names)
	['TM', 'AMP']
	>>> data.sequences.shape
	(412, 1)
	"""

	module_path = dirname(__file__)
	with open(join(module_path, 'data', 'AMPvsTMset.csv')) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		# n_samples = int(temp[0])
		# n_features = int(temp[1])
		target_names = np.array(temp[2:])
		sequences = []  # np.empty((n_samples, n_features), dtype='|S100')
		target = []  # np.empty((n_samples,), dtype=np.int)

		for i, ir in enumerate(data_file):
			sequences.append(ir[0])  # sequences[i] = ir[0]
			target.append(ir[-1])  # target[i] = ir[-1]

	return Bunch(sequences=sequences, target=target,
				 target_names=target_names,
				 feature_names=['Sequence'])


def load_helicalAMPset():
	"""Function to load a dataset consisting of **helical AMP sequences and other helical peptides** for classification.

	The AMP class consists of 363 helical annotated sequences from the `APD2 <http://aps.unmc.edu/AP/>`_
	(extracted Dez. 14 2015). The HEL class is constructed out of extracted all alpha annotated proteins from the
	`PDB <http://www.rcsb.org/pdb/home/home.do>`_, out of which alpha helical regions were extracted. 363 sequences
	were then randomly chosen from this extracted set to get equally balanced datasets in terms of numbers of sequences.

	=================	====
	Classes				2
	Samples per class	363
	Samples total		726
	Dimensionality		1
	=================	====

	:return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``, the
			classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the features.
	:Example:

	>>> from modlamp.datasets import load_helicalAMPset
	>>> data = load_helicalAMPset()
	>>> data.sequences[:5]
	array([['FDQAQTEIQATMEEN'],['DVDAALHYLARLVEAG'],['RCPLVIDYLIDLATRS'],['NPATLMMFFK'],['NLEDSIQILRTD']])
	>>> list(data.target_names)
	['HEL', 'AMP']
	>>> data.sequences.shape
	(726, 1)
	"""

	module_path = dirname(__file__)
	with open(join(module_path, 'data', 'helicalAMPset.csv')) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		# n_samples = int(temp[0])
		# n_features = int(temp[1])
		target_names = np.array(temp[2:])
		sequences = []  # np.empty((n_samples, n_features), dtype='|S100')
		target = []  # np.empty((n_samples,), dtype=np.int)

		for i, ir in enumerate(data_file):
			sequences.append(ir[0])  # sequences[i] = ir[0]
			target.append(ir[-1])  # target[i] = ir[-1]

	return Bunch(sequences=sequences, target=target,
				 target_names=target_names,
				 feature_names=['Sequence'])


def load_ACPvsTMandNoTMset():
	"""Function to load a dataset consisting of **alpha-helical ACP sequences and negative peptides extracted from
	alpha-helical transmembrane and non-transmembrane regions of proteins** for classification.

	The ACP class consists of a collection of ACPs from the `APD2 <http://aps.unmc.edu/AP/>`_ and
	`CancerPPD <http://crdd.osdd.net/raghava/cancerppd/index.php>`_ databases, manually curated by Gisela Gabernet at
	modlab ETH Zuerich, checking the original literature and annotated active against at least one of the following cancer types at a concentration of 50 µM:
	breast, lung, skin, haematological, and cervical. Selected sequences with length between 7 and 30 aa and without
	cisteines to facilitate synthesis.

	The Negative peptide set contains a mixture of a random selection of 47 transmembrane alpha-helices (extracted from the
	 `PDBTM <http://pdbtm.enzim.hu/>` ) and 47 non-transmembrane helices (extracted from the `PDB
	<http://www.rcsb.org/pdb/home/home.do>`) isolated directly from the proteins crystal structure.

	=================	====
	Classes				2
	ACP peptides 		95
	Negative peptides	94
	Total peptides		189
	Dimensionality		1
	=================	====

	:return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``, the
		classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the features.
	:Example:

	>>> from modlamp.datasets import load_ACPvsTMandNoTMset
	>>> data = load_ACPvsTMandNoTMset()
	>>> data.sequences[:5]
	['VLTIIATIFMPLTFIAGI', 'QLGAGLSVGLSGLAAGFAIGIVG', 'WLYLILGIIFGIFGPIFNKWVL', 'VTWLLFLLGFVAILI', 'TRELFLNFTIVLITVILMWLLV']
	>>> list(data.target_names)
	['Neg', 'ACP']
	>>> len(data.sequences)
	189
	"""

	module_path = dirname(__file__)
	with open(join(module_path, 'data', 'ACPvsTMandNoTMset.csv')) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		# n_samples = int(temp[0])
		# n_features = int(temp[1])
		target_names = np.array(temp[2:])
		sequences = []  # np.empty((n_samples, n_features), dtype='|S100')
		target = []  # np.empty((n_samples,), dtype=np.int)

		for i, ir in enumerate(data_file):
			sequences.append(ir[0])  # sequences[i] = ir[0]
			target.append(ir[-1])  # target[i] = ir[-1]

	return Bunch(sequences=sequences, target=target,
				 target_names=target_names,
				 feature_names=['Sequence'])


# TODO: add more data set loading functions

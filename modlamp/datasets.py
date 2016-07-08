# -*- coding: utf-8 -*-
"""
.. module:: modlamp.datasets

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to load different peptide datasets used for classification.

=============================		============================================================================
Function							Data
=============================		============================================================================
:py:func:`load_AMPvsTMset`			Antimicrobial peptides versus trans-membrane sequences
:py:func:`load_helicalAMPset`		Helical antimicrobial peptides versus other helical peptides
:py:func:`load_ACPvsNeg`			Helical anticancer peptides versus other mixed sequences
:py:func:`load_AMPvsUniProt`		AMPs from the *APD3* versus other peptides from *UniProt*
=============================		============================================================================
"""

import csv
from os.path import dirname
from os.path import join

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

	:return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``,
		the	classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the
		features.
	:Example:

	>>> from modlamp.datasets import load_AMPvsTMset
	>>> data = load_AMPvsTMset()
	>>> data.sequences
	['AAGAATVLLVIVLLAGSYLAVLA','LWIVIACLACVGSAAALTLRA','FYRFYMLREGTAVPAVWFSIELIFGLFA','GTLELGVDYGRAN',...]
	>>> list(data.target_names)
	['TM', 'AMP']
	>>> len(data.sequences)
	412
	>>> data.target[:10]
	array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	"""

	module_path = dirname(__file__)
	with open(join(module_path, 'data', 'AMPvsTMset.csv')) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		n_samples = int(temp[0])
		n_features = int(temp[1])
		target_names = np.array(temp[2:])
		sequences = np.empty((n_samples, n_features), dtype='|S100')
		target = np.empty((n_samples,), dtype=np.int)

		for i, ir in enumerate(data_file):
			sequences[i] = np.asarray(ir[0], dtype=np.str)
			target[i] = np.asarray(ir[-1], dtype=np.int)

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

	:return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``,
		the classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the
		features.
	:Example:

	>>> from modlamp.datasets import load_helicalAMPset
	>>> data = load_helicalAMPset()
	>>> data.sequences[:5]
	['FDQAQTEIQATMEEN','DVDAALHYLARLVEAG','RCPLVIDYLIDLATRS','NPATLMMFFK','NLEDSIQILRTD']
	>>> list(data.target_names)
	['HEL', 'AMP']
	>>> len(data.sequences)
	726
	"""

	module_path = dirname(__file__)
	with open(join(module_path, 'data', 'helicalAMPset.csv')) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		n_samples = int(temp[0])
		n_features = int(temp[1])
		target_names = np.array(temp[2:])
		sequences = np.empty((n_samples, n_features), dtype='|S100')
		target = np.empty((n_samples,), dtype=np.int)

		for i, ir in enumerate(data_file):
			sequences[i] = np.asarray(ir[0], dtype=np.str)
			target[i] = np.asarray(ir[-1], dtype=np.int)

	return Bunch(sequences=sequences, target=target,
				 target_names=target_names,
				 feature_names=['Sequence'])


def load_ACPvsNeg():
	"""Function to load a dataset consisting of **alpha-helical ACP sequences and negative peptides extracted from
	alpha-helical transmembrane and non-transmembrane regions of proteins** for classification.

	The ACP class consists of a collection of ACPs from the `APD2 <http://aps.unmc.edu/AP/>`_ and
	`CancerPPD <http://crdd.osdd.net/raghava/cancerppd/index.php>`_ databases, manually curated by Gisela Gabernet at
	modlab ETH Zuerich <gisela.gabernet@pharma.ethz.ch>, checking the original literature and annotated active against
	at least one of the following cancer types at a concentration of 50 µM: breast, lung, skin, haematological, and
	cervical. Selected sequences with length between 7 and 30 aa and without Cysteines to facilitate synthesis.

	The Negative peptide set contains a mixture of a random selection of 47 transmembrane alpha-helices (extracted from
	the `PDBTM <http://pdbtm.enzim.hu/>` ) and 47 non-transmembrane helices (extracted from the `PDB
	<http://www.rcsb.org/pdb/home/home.do>`) isolated directly from the proteins crystal structure.

	=================	====
	Classes				2
	ACP peptides 		95
	Negative peptides	94
	Total peptides		189
	Dimensionality		1
	=================	====

	:return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``,
		the classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the
		features.
	:Example:

	>>> from modlamp.datasets import load_ACPvsNeg
	>>> data = load_ACPvsNeg()
	>>> data.sequences
	['VLTIIATIFMPLTFIAGI', 'QLGAGLSVGLSGLAAGFAIGIVG', 'WLYLILGIIFGIFGPIFNKWVL', 'VTWLLFLLGFVAILI'...]
	>>> list(data.target_names)
	['Neg', 'ACP']
	>>> len(data.sequences)
	189
	"""

	module_path = dirname(__file__)
	with open(join(module_path, 'data', 'ACPvsNeg.csv')) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		n_samples = int(temp[0])
		n_features = int(temp[1])
		target_names = np.array(temp[2:])
		sequences = np.empty((n_samples, n_features), dtype='|S100')
		target = np.empty((n_samples,), dtype=np.int)

		for i, ir in enumerate(data_file):
			sequences[i] = np.asarray(ir[0], dtype=np.str)
			target[i] = np.asarray(ir[-1], dtype=np.int)

	return Bunch(sequences=sequences, target=target,
				 target_names=target_names,
				 feature_names=['Sequence'])


def load_AMPvsUniProt():
	"""Function to load a dataset consisting of **APD3 < 80% internal similarity versus UniProt not antimicrobial,
	not secretory, 10-50 AA and verified** for classification.

	The AMP class consists of 1609 AMP sequences from the `APD3 <http://aps.unmc.edu/AP/>`_. The whole APD3 was
	extracted (Mar. 3rd  2016) and then submitted to an internal similarity filtering with a threshold of 80% by the
	`CD-Hit tool <http://cd-hit.org>`_. The UniProt class is constructed out of extracted protein sequences from the
	`UniProt Database <http://uniprot.org/>`_ with the search query *NOT secretory NOT antimicrobial
	AND length 10 TO 50*. These sequences were as well subjected to a similarity filtering of 80% yielding 4279
	sequences.

	=================	=====
	Classes				2
	AMP Samples			1609
	UniProt Samples		4279
	Samples total		5888
	Dimensionality		1
	=================	=====

	:return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``,
		the classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the
		features.
	:Example:

	>>> from modlamp.datasets import load_AMPvsUniProt
	>>> data = load_AMPvsUniProt()
	>>> data.sequences[:5]
	['GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV',
	'YVPLPNVPQPGRRPFPTFPGQGPFNPKIKWPQGY',
	'DGVKLCDVPSGTWSGHCGSSSKCSQQCKDREHFAYGGACHYQFPSVKCFCKRQC',
	'NLCERASLTWTGNCGNTGHCDTQCRNWESAKHGACHKRGNWKCFCYFDC',
	'VFIDILDKVENAIHNAAQVGIGFAKPFEKLINPK']
	>>> list(data.target_names)
	['AMP', 'UniProt']
	>>> len(data.sequences)
	5888
	>>> data.target[:10]
	array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
	"""

	module_path = dirname(__file__)
	with open(join(module_path, 'data', 'AMPvsUniProt.csv')) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		n_samples = int(temp[0])
		n_features = int(temp[1])
		target_names = np.array(temp[2:])
		sequences = np.empty((n_samples, n_features), dtype='|S100')
		target = np.empty((n_samples,), dtype=np.int)

		for i, ir in enumerate(data_file):
			sequences[i] = np.asarray(ir[0], dtype=np.str)
			target[i] = np.asarray(ir[-1], dtype=np.int)

	return Bunch(sequences=sequences, target=target,
				 target_names=target_names,
				 feature_names=['Sequence'])

# TODO: add more data set loading functions

# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.datasets

.. moduleauthor:: ETH Zurich Alex Mueller <alex.mueller@pharma.ethz.ch>

This module incorporates functions to load different peptide datasets used for classification.

=============================        ============================================================================
Function                             Data
=============================        ============================================================================
:py:func:`load_AMPvsTM`              Antimicrobial peptides versus trans-membrane sequences
:py:func:`load_AMPvsUniProt`         AMPs from the *APD3* versus other peptides from the *UniProt* database
:py:func:`load_ACPvsTM`              Anticancer peptides (*CancerPPD*) versus helical transmembrane sequences
:py:func:`load_ACPvsRandom`          Anticancer peptides (*CancerPPD*) versus random scrambled AMP sequences
:py:func:`load_custom`               A custom data set provided in ``modlamp/data`` as a ``.csv`` file
=============================        ============================================================================
"""

import csv
from os.path import dirname, join

import numpy as np

__author__ = "Alex Müller, Gisela Gabernet"
__docformat__ = "restructuredtext en"


class Bunch(dict):
    """Container object for datasets

    Dictionary-like object that exposes its keys as attributes. Taken from the `sklearn <http://scikit-learn.org>`_
    package.

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


def load_AMPvsTM():
    """Function to load a dataset consisting of **AMP sequences** and **transmembrane regions of proteins** for
    classification.

    The AMP class consists of an intersection of all activity annotations of the `APD2 <http://aps.unmc.edu/AP/>`_ and
    `CAMP <http://camp.bicnirrh.res.in/>`_ databases, where for gram positive, gram negative and antifungal exact
    matches were observed. A detailed description of how the dataset was compiled can be found in the following
    publication: Schneider, P., Müller, A. T., Gabernet, G., Button, A. L., Posselt, G., Wessler, S., Hiss, J. A. and
    Schneider, G. (2016), Hybrid Network Model for “Deep Learning” of Chemical Data: Application to Antimicrobial
    Peptides. Mol. Inf.. `doi:10.1002/minf.201600011 <http://onlinelibrary.wiley.com/doi/10.1002/minf.201600011/full>`_

    =================    ===
    Classes                2
    Samples per class    206
    Samples total        412
    Dimensionality         1
    =================    ===

    :return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``,
        the classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the
        features.
    :Example:

    >>> from modlamp.datasets import load_AMPvsTM
    >>> data = load_AMPvsTM()
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

    return Bunch(sequences=sequences.reshape(1, -1)[0], target=target,
                 target_names=target_names,
                 feature_names=['Sequence'])


def load_AMPvsUniProt():
    """Function to load a dataset consisting of the whole **APD3** versus the same number of sequences randomly
    extracted from the **UniProt** database, to be used for classification.

    The AMP class consists of 2600 AMP sequences from the `APD3 <http://aps.unmc.edu/AP/>`_ (extracted Jan. 2016).
    The UniProt class consists of 2600 randomly extracted protein sequences from the `UniProt Database
    <http://uniprot.org/>`_ with the search query *length 10 TO 50* filtered for unnatural amino acids.

    =================    =====
    Classes                 2
    AMP Samples          2600
    UniProt Samples      2600
    Samples total        5200
    Dimensionality          1
    =================    =====

    :return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``,
        the classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the
        features.
    :Example:

    >>> from modlamp.datasets import load_AMPvsUniProt
    >>> data = load_AMPvsUniProt()
    >>> data.sequences[:10]
    ['GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV', 'YVPLPNVPQPGRRPFPTFPGQGPFNPKIKWPQGY', ... ]
    >>> list(data.target_names)
    ['AMP', 'UniProt']
    >>> len(data.sequences)
    5200
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

    return Bunch(sequences=sequences.reshape(1, -1)[0], target=target,
                 target_names=target_names,
                 feature_names=['Sequence'])


def load_ACPvsTM():
    """Function to load a dataset consisting of ACP sequences from the CancerPPD database and negative peptides
    extracted from alpha-helical transmembrane regions of proteins for classification.

    The ACP class consists of a collection of 413 ACPs from the `CancerPPD
    <http://crdd.osdd.net/raghava/cancerppd/index.php>`_ database with length between 7 and 30 aa and without cysteines
    to facilitate peptide synthesis.

    The Negative peptide set contains a random selection of 413 transmembrane alpha-helices (extracted from
    the `PDBTM <http://pdbtm.enzim.hu/>`_ ) isolated directly from the proteins crystal structure.

    =================    ===
    Classes                2
    ACP peptides         413
    Negative peptides    413
    Total peptides       826
    Dimensionality         1
    =================    ===

    :return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``,
        the classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the
        features.
    :Example:

    >>> from modlamp.datasets import load_ACPvsTM
    >>> data = load_ACPvsTM()
    >>> data.sequences[:4]
    ['AAKKWAKAKWAKAKKWAKAA', 'AAVPIVNLKDELLFPSWEALFSGSE', 'AAWKWAWAKKWAKAKKWAKAA', 'AFGMALKLLKKVL']
    >>> list(data.target_names)
    ['TM', 'ACP']
    >>> len(data.sequences)
    826
    """

    module_path = dirname(__file__)
    with open(join(module_path, 'data', 'ACP_CancPPD_vs_TM.csv')) as csv_file:
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

    return Bunch(sequences=sequences.reshape(1, -1)[0], target=target,
                 target_names=target_names,
                 feature_names=['Sequence'])


def load_ACPvsRandom():
    """Function to load a dataset consisting of ACP sequences from the CancerPPD database and negative peptides generated
     randomly with the amino acid composition of AMPs.

    The ACP class consists of a collection of 413 ACPs from the `CancerPPD
    <http://crdd.osdd.net/raghava/cancerppd/index.php>`_ database with length between 7 and 30 aa and without cysteines
    to facilitate peptide synthesis.

    The Negative peptide set contains a random selection of 413 randomly generated peptides with the amino acid
    composition of AMPs in the APD2 database.

    =================    ===
    Classes                2
    ACP peptides         413
    Negative peptides    413
    Total peptides       826
    Dimensionality         1
    =================    ===

    :return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``,
        the classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the
        features.
    :Example:

    >>> from modlamp.datasets import load_ACPvsRandom
    >>> data = load_ACPvsRandom()
    >>> data.sequences[:3]
    ['AAKKWAKAKWAKAKKWAKAA', 'AAVPIVNLKDELLFPSWEALFSGSE', 'AAWKWAWAKKWAKAKKWAKAA']
    >>> list(data.target_names)
    ['Random', 'ACP']
    >>> len(data.sequences)
    826
    """

    module_path = dirname(__file__)
    with open(join(module_path, 'data', 'ACP_CancPPD_vs_Random.csv')) as csv_file:
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

    return Bunch(sequences=sequences.reshape(1, -1)[0], target=target,
                 target_names=target_names,
                 feature_names=['Sequence'])


def load_custom(filename):
    """Function to load a custom dataset saved in ``modlamp/data/`` as a ``.csv`` file.
    
    The following header needs to be included: *Nr. of sequences*, *Nr. of columns - 1*, *Class name for 0*,
    *Class name for 1*
    
    Example ``.csv`` file structure::
    
        4, 1, TM, AMP
        GTLEFDVTIGRAN, 0
        GSNVHLASNLLA, 0
        GLFDIVKKVVGALGSL, 0
        GLFDIIKKIAESF, 0
    
    :param filename: {str} filename of the data file to be loaded; the file must be located in ``modlamp/data/``
    :return: Bunch, a dictionary-like object, the interesting attributes are: ``sequences``, the sequences, ``target``,
        the classification labels, ``target_names``, the meaning of the labels and ``feature_names``, the meaning of the
        features.
    :Example:

    >>> from modlamp.datasets import load_custom
    >>> data = load_custom('custom_data.csv')
    """

    module_path = dirname(__file__)
    with open(join(module_path, 'data', filename)) as csv_file:
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

    return Bunch(sequences=sequences.reshape(1, -1)[0], target=target,
                 target_names=target_names,
                 feature_names=['Sequence'])

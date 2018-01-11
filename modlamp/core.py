# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.core

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

Core helper functions and classes for other modules. The two main classes are:

=============================    =======================================================================================
Class                            Characteristics
=============================    =======================================================================================
:py:class:`BaseSequence`         Base class inheriting to all sequence classes in the module :py:mod:`modlamp.sequences`
:py:class:`BaseDescriptor`       Base class inheriting to the two descriptor classes in :py:mod:`modlamp.descriptors`
=============================    =======================================================================================
"""

import os
import random
import re

import numpy as np
import pandas as pd
import collections
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle

__author__ = "Alex Müller, Gisela Gabernet"
__docformat__ = "restructuredtext en"


class BaseSequence(object):
    """Base class for sequence classes in the module :mod:`modlamp.sequences`.
    It contains amino acid probabilities for different sequence generation classes.
    
    The following amino acid probabilities are used: (extracted from the
    `APD3 <http://aps.unmc.edu/AP/statistic/statistic.php>`_, March 17, 2016)

    ===  ====    ======   =========    ==========
    AA   rand    AMP      AMPnoCM      randnoCM
    ===  ====    ======   =========    ==========
    A    0.05    0.0766   0.0812275    0.05555555
    C    0.05    0.071    0.0          0.0
    D    0.05    0.026    0.0306275    0.05555555
    E    0.05    0.0264   0.0310275    0.05555555
    F    0.05    0.0405   0.0451275    0.05555555
    G    0.05    0.1172   0.1218275    0.05555555
    H    0.05    0.021    0.0256275    0.05555555
    I    0.05    0.061    0.0656275    0.05555555
    K    0.05    0.0958   0.1004275    0.05555555
    L    0.05    0.0838   0.0884275    0.05555555
    M    0.05    0.0123   0.0          0.0
    N    0.05    0.0386   0.0432275    0.05555555
    P    0.05    0.0463   0.0509275    0.05555555
    Q    0.05    0.0251   0.0297275    0.05555555
    R    0.05    0.0545   0.0591275    0.05555555
    S    0.05    0.0613   0.0659275    0.05555555
    T    0.05    0.0455   0.0501275    0.05555555
    V    0.05    0.0572   0.0618275    0.05555555
    W    0.05    0.0155   0.0201275    0.05555555
    Y    0.05    0.0244   0.0290275    0.05555555
    ===  ====    ======   =========    ==========
    
    """
    
    def __init__(self, seqnum, lenmin=7, lenmax=28):
        """
        :param seqnum: number of sequences to generate
        :param lenmin: minimal length of the generated sequences
        :param lenmax: maximal length of the generated sequences
        :return: attributes :py:attr:`seqnum`, :py:attr:`lenmin` and :py:attr:`lenmax`.
        :Example:
        
        >>> b = BaseSequence(10, 7, 28)
        >>> b.seqnum
        10
        >>> b.lenmin
        7
        >>> b.lenmax
        28
        """
        self.sequences = list()
        self.names = list()
        self.lenmin = int(lenmin)
        self.lenmax = int(lenmax)
        self.seqnum = int(seqnum)
        
        # AA classes:
        self.AA_hyd = ['G', 'A', 'L', 'I', 'V']
        self.AA_basic = ['K', 'R']
        self.AA_acidic = ['D', 'E']
        self.AA_aroma = ['W', 'Y', 'F']
        self.AA_polar = ['S', 'T', 'Q', 'N']
        # AA labels:
        self.AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        # AA probability from the APD3 database:
        self.prob_AMP = [0.0766, 0.071, 0.026, 0.0264, 0.0405, 0.1172, 0.021, 0.061, 0.0958, 0.0838, 0.0123, 0.0386,
                         0.0463,0.0251, 0.0545, 0.0613, 0.0455, 0.0572, 0.0155, 0.0244]
        # AA probability from the APD2 database without Cys and Met (synthesis reasons)
        self.prob_AMPnoCM = [0.081228, 0., 0.030627, 0.031027, 0.045128, 0.121828, 0.025627, 0.065628, 0.100428,
                             0.088428, 0., 0.043228, 0.050928, 0.029728, 0.059128, 0.065927, 0.050128, 0.061828,
                             0.020128, 0.029028]
        # equal AA probabilities:
        self.prob = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                     0.05, 0.05, 0.05, 0.05]
        # equal AA probabilities but 0 for Cys and Met:
        self.prob_randnoCM = [0.05555555555, 0.0, 0.05555555555, 0.05555555555, 0.05555555555, 0.05555555555,
                              0.05555555555, 0.05555555555, 0.05555555555, 0.05555555555, 0.0, 0.05555555555,
                              0.05555555555, 0.05555555555, 0.05555555555, 0.05555555555, 0.05555555555, 0.05555555555,
                              0.05555555555, 0.05555555555]

        # AA probability from the linear CancerPPD peptides:
        self.prob_ACP = [0.14526966, 0., 0.00690031, 0.00780824, 0.06991102, 0.04957327, 0.01725077, 0.05647358,
                         0.27637552, 0.17759216, 0.00998729, 0.00798983, 0.01307427, 0.00381333, 0.02941711,
                         0.02651171, 0.0154349, 0.04013074, 0.0406755, 0.00581079]

        # AA probabilities for perfect amphipathic helix of different arc sizes
        self.prob_amphihel = [[0.04545455, 0., 0.04545454, 0.04545455, 0., 0.04545455, 0.04545455, 0., 0.25, 0., 0.,
                               0.04545454, 0.04545455, 0.04545454, 0.25, 0.04545454, 0.04545454, 0., 0., 0.04545454],
                              [0., 0., 0., 0., 0.16666667, 0., 0., 0.16666667, 0., 0.16666667, 0., 0., 0., 0., 0., 0., 0.,
                               0.16666667, 0.16666667, (1. - 0.16666667 * 5)]]

        # helical ACP AA probabilities, depending on the position of the AA in the helix.
        self.prob_ACPhel = np.array([[0.0483871, 0., 0., 0.0483871, 0.01612903, 0.12903226, 0.03225807, 0.09677419,
                                      0.19354839, 0.5, 0.0483871, 0.11290323, 0.1, 0.18518519, 0.07843137, 0.12,
                                      0.17073172, 0.16666667],
                                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01612903, 0., 0., 0., 0., 0.,
                                      0.02439024,
                                      0.19444444],
                                     [0., 0.01612903, 0., 0.27419355, 0.01612903, 0., 0., 0.01612903, 0., 0., 0., 0.,
                                      0.,
                                      0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0., 0., 0.06451613, 0., 0.01612903, 0.0483871, 0.01612903, 0.,
                                      0.01851852, 0., 0., 0., 0.],
                                     [0.16129032, 0.0483871, 0.30645161, 0., 0.0483871, 0., 0., 0.01612903, 0.,
                                      0.01612903,
                                      0., 0.09677419, 0.06666667, 0.01851852, 0., 0.02, 0.14634146, 0.],
                                     [0.64516129, 0., 0.17741936, 0.14516129, 0., 0.01612903, 0.25806452, 0.11290323,
                                      0.06451613, 0.08064516, 0.22580645, 0.03225807, 0.06666667, 0.2037037, 0.1372549,
                                      0.1, 0., 0.05555556],
                                     [0., 0., 0., 0.01612903, 0., 0., 0.01612903, 0., 0.03225807, 0., 0., 0.20967742,
                                      0.,
                                      0., 0., 0.16, 0., 0.],
                                     [0.0483871, 0.11290323, 0.01612903, 0.08064516, 0.33870968, 0.27419355, 0.,
                                      0.0483871, 0.14516129, 0.06451613, 0.03225807, 0.06451613, 0.18333333, 0., 0.,
                                      0.1, 0.26829268, 0.],
                                     [0., 0.03225807, 0.01612903, 0.12903226, 0.12903226, 0., 0.38709677, 0.33870968,
                                      0.0483871, 0.03225807, 0.41935484, 0.08064516, 0., 0.03703704, 0.29411765,
                                      0.04, 0.02439024, 0.02777778],
                                     [0.0483871, 0.70967742, 0.12903226, 0.0483871, 0.09677419, 0.32258064, 0.20967742,
                                      0.06451613, 0.11290323, 0.06451613, 0.03225807, 0.03225807, 0.28333333,
                                      0.24074074,
                                      0.03921569, 0.28, 0.07317073, 0.22222222],
                                     [0., 0.01612903, 0.01612903, 0.0483871, 0.01612903, 0.03225807, 0., 0., 0., 0.,
                                      0., 0., 0.03333333, 0., 0.01960784, 0.02, 0., 0.],
                                     [0., 0.01612903, 0., 0., 0., 0., 0., 0., 0.01612903, 0., 0.03225807, 0., 0., 0.,
                                      0.01960784, 0.02, 0., 0.],
                                     [0., 0., 0.14516129, 0.01612903, 0.03225807, 0.01612903, 0., 0., 0., 0.,
                                      0.01612903, 0., 0., 0.12962963, 0.17647059, 0., 0., 0.],
                                     [0., 0., 0.01612903, 0.01612903, 0., 0., 0.01612903, 0., 0.01612903, 0., 0.,
                                      0.01612903, 0., 0.01851852, 0., 0., 0., 0.],
                                     [0., 0.01612903, 0.01612903, 0., 0.01612903, 0., 0.01612903, 0., 0.01612903,
                                      0.01612903, 0.01612903, 0.01612903, 0., 0.01851852, 0.01960784, 0., 0.04878049,
                                      0.],
                                     [0.01612903, 0., 0.01612903, 0.12903226, 0.03225807, 0.03225807, 0.0483871,
                                      0.17741936, 0., 0.03225807, 0.09677419, 0.0483871, 0.01666667, 0., 0.15686274,
                                      0.1, 0., 0.05555556],
                                     [0.01612903, 0.01612903, 0., 0.01612903, 0.0483871, 0.01612903, 0., 0.01612903, 0.,
                                      0.01612903, 0.01612903, 0.11290323, 0., 0.01851852, 0.03921569, 0.02, 0.,
                                      0.05555556],
                                     [0.01612903, 0.01612903, 0.01612903, 0.01612903, 0.20967742, 0.16129032,
                                      0.01612903,
                                      0.0483871, 0.33870968, 0.16129032, 0., 0.14516129, 0.25, 0.11111111, 0.01960784,
                                      0.02, 0.21951219, 0.22222222],
                                     [0., 0., 0.12903226, 0.01612903, 0., 0., 0., 0., 0.01612903, 0., 0., 0., 0., 0.,
                                      0.,
                                      0., 0.02439024, 0.],
                                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01612903, 0., 0., 0., 0., 0., 0.]])
    
    def save_fasta(self, filename, names=False):
        """Method to save generated sequences in a ``.FASTA`` formatted file.

        :param filename: output filename in which the sequences from :py:attr:`sequences` are safed in fasta format.
        :param names: {bool} whether sequence names from :py:attr:`names` should be saved as sequence identifiers
        :return: a FASTA formatted file containing the generated sequences
        :Example:
        
        >>> b = BaseSequence(2)
        >>> b.sequences = ['KLLSLSLALDLLS', 'KLPERTVVNSSDF']
        >>> b.names = ['Sequence1', 'Sequence2']
        >>> b.save_fasta('/location/of/fasta/file.fasta', names=True)
        """
        if names:
            save_fasta(filename, self.sequences, self.names)
        else:
            save_fasta(filename, self.sequences)
    
    def mutate_AA(self, nr, prob):
        """Method to mutate with **prob** probability a **nr** of positions per sequence randomly.

        :param nr: number of mutations to perform per sequence
        :param prob: probability of mutating a sequence
        :return: mutated sequences in the attribute :py:attr:`sequences`.
        :Example:

        >>> b = BaseSequence(1)
        >>> b.sequences = ['IAKAGRAIIK']
        >>> b.mutate_AA(3, 1.)
        >>> b.sequences
        ['NAKAGRAWIK']
        """
        for s in range(len(self.sequences)):
            # mutate: yes or no? prob = mutation probability
            mutate = np.random.choice([1, 0], 1, p=[prob, 1 - float(prob)])
            if mutate == 1:
                seq = list(self.sequences[s])
                cnt = 0
                while cnt < nr:  # mutate "nr" AA
                    seq[random.choice(range(len(seq)))] = random.choice(self.AAs)
                    cnt += 1
                self.sequences[s] = ''.join(seq)
    
    def filter_duplicates(self):
        """Method to filter duplicates in the sequences from the class attribute :py:attr:`sequences`

        :return: filtered sequences list in the attribute :py:attr:`sequences` and corresponding names.
        :Example:
        
        >>> b = BaseSequence(4)
        >>> b.sequences = ['KLLKLLKKLLKLLK', 'KLLKLLKKLLKLLK', 'KLAKLAKKLAKLAK', 'KLAKLAKKLAKLAK']
        >>> b.filter_duplicates()
        >>> b.sequences
        ['KLLKLLKKLLKLLK', 'KLAKLAKKLAKLAK']

        .. versionadded:: v2.2.5
        """
        if not self.names:
            self.names = ['Seq_' + str(i) for i in range(len(self.sequences))]
        df = pd.DataFrame(zip(self.sequences, self.names), columns=['Sequences', 'Names'])
        df = df.drop_duplicates('Sequences', 'first')  # keep first occurrence of duplicate
        self.sequences = df['Sequences'].get_values().tolist()
        self.names = df['Names'].get_values().tolist()
    
    def keep_natural_aa(self):
        """Method to filter out sequences that do not contain natural amino acids. If the sequence contains a character
        that is not in ``['A','C','D,'E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']``.

        :return: filtered sequence list in the attribute :py:attr:`sequences`. The other attributes are also filtered
            accordingly (if present).
        :Example:
        
        >>> b = BaseSequence(2)
        >>> b.sequences = ['BBBsdflUasUJfBJ', 'GLFDIVKKVVGALGSL']
        >>> b.keep_natural_aa()
        >>> b.sequences
        ['GLFDIVKKVVGALGSL']
        """
        natural_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                      'Y']
        
        seqs = []
        names = []
        
        for i, s in enumerate(self.sequences):
            seq = list(s.upper())
            if all(c in natural_aa for c in seq):
                seqs.append(s.upper())
                if hasattr(self, 'names') and self.names:
                    names.append(self.names[i])
        
        self.sequences = seqs
        self.names = names
    
    def filter_aa(self, amino_acids):
        """Method to filter out corresponding names and descriptor values of sequences with given amino acids in the
        argument list *aminoacids*.

        :param amino_acids: {list} amino acids to be filtered
        :return: filtered list of sequences names in the corresponding attributes.
        :Example:
        
        >>> b = BaseSequence(3)
        >>> b.sequences = ['AAALLLIIIKKK', 'CCEERRT', 'LLVVIIFFFQQ']
        >>> b.filter_aa(['C'])
        >>> b.sequences
        ['AAALLLIIIKKK', 'LLVVIIFFFQQ']
        """
        
        pattern = re.compile('|'.join(amino_acids))
        seqs = []
        names = []
        
        for i, s in enumerate(self.sequences):
            if not pattern.search(s):
                seqs.append(s)
                if hasattr(self, 'names') and self.names:
                    names.append(self.names[i])
        
        self.sequences = seqs
        self.names = names
    
    def clean(self):
        """Method to clean / clear / empty the attributes :py:attr:`sequences` and :py:attr:`names`.

        :return: freshly initialized, empty class attributes.
        """
        self.__init__(self.seqnum, self.lenmin, self.lenmax)


class BaseDescriptor(object):
    """
    Base class inheriting to both peptide descriptor classes :py:class:`modlamp.descriptors.GlobalDescriptor` and
    :py:class:`modlamp.descriptors.PeptideDescriptor`.
    """
    
    def __init__(self, seqs):
        """
        :param seqs: a ``.FASTA`` file with sequences, a list / array of sequences or a single sequence as string to
            calculate the descriptor values for.
        :return: initialized attributes :py:attr:`sequences` and :py:attr:`names`.
        :Example:

        >>> AMP = BaseDescriptor('KLLKLLKKLLKLLK','pepCATS')
        >>> AMP.sequences
        ['KLLKLLKKLLKLLK']
        >>> seqs = BaseDescriptor('/Path/to/file.fasta', 'eisenberg')  # load sequences from .fasta file
        >>> seqs.sequences
        ['AFDGHLKI','KKLQRSDLLRTK','KKLASCNNIPPR'...]
        """
        if type(seqs) == list and seqs[0].isupper():
            self.sequences = seqs
            self.names = []
        elif type(seqs) == np.ndarray and seqs[0].isupper():
            self.sequences = seqs.tolist()
            self.names = []
        elif type(seqs) == str and seqs.isupper():
            self.sequences = [seqs]
            self.names = []
        elif os.path.isfile(seqs):
            if seqs.endswith('.fasta'):  # read .fasta file
                self.sequences, self.names = read_fasta(seqs)
            elif seqs.endswith('.csv'):  # read .csv file with sequences every line
                with open(seqs) as f:
                    self.sequences = list()
                    cntr = 0
                    self.names = []
                    for line in f:
                        if line.isupper():
                            self.sequences.append(line.strip())
                            self.names.append('seq_' + str(cntr))
                            cntr += 1
            else:
                print("Sorry, currently only .fasta or .csv files can be read!")
        else:
            print("%s does not exist, is not a valid list of AA sequences or is not a valid sequence string" % seqs)
        
        self.descriptor = np.array([[]])
        self.target = np.array([], dtype='int')
        self.scaler = None
        self.featurenames = []
    
    def read_fasta(self, filename):
        """Method for loading sequences from a ``.FASTA`` formatted file into the attributes :py:attr:`sequences` and
        :py:attr:`names`.

        :param filename: {str} ``.FASTA`` file with sequences and headers to read
        :return: {list} sequences in the attribute :py:attr:`sequences` with corresponding sequence names in
            :py:attr:`names`.
        """
        self.sequences, self.names = read_fasta(filename)
    
    def save_fasta(self, filename, names=False):
        """Method for saving sequences from :py:attr:`sequences` to a ``.FASTA`` formatted file.

        :param filename: {str} filename of the output ``.FASTA`` file
        :param names: {bool} whether sequence names from self.names should be saved as sequence identifiers
        :return: a FASTA formatted file containing the generated sequences
        """
        if names:
            save_fasta(filename, self.sequences, self.names)
        else:
            save_fasta(filename, self.sequences)
    
    def count_aa(self, scale='relative', average=False, append=False):
        """Method for producing the amino acid distribution for the given sequences as a descriptor

        :param scale: {'absolute' or 'relative'} defines whether counts or frequencies are given for each AA
        :param average: {boolean} whether the averaged amino acid counts for all sequences should be returned
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: the amino acid distributions for every sequence individually in the attribute :py:attr:`descriptor`
        :Example:

        >>> AMP = PeptideDescriptor('ACDEFGHIKLMNPQRSTVWY') # aa_count() does not depend on the descriptor scale
        >>> AMP.count_aa()
        >>> AMP.descriptor
        array([[ 0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05, ... ]])
        >>> AMP.descriptor.shape
        (1, 20)

        .. seealso:: :py:func:`modlamp.core.count_aa()`
        """
        desc = list()
        for seq in self.sequences:
            od = count_aas(seq, scale)
            desc.append(od.values())
        
        desc = np.array(desc)
        self.featurenames = od.keys()
        
        if append:
            self.descriptor = np.hstack((self.descriptor, desc))
        elif average:
            self.descriptor = np.mean(desc, axis=0)
        else:
            self.descriptor = desc
    
    def feature_scaling(self, stype='standard', fit=True):
        """Method for feature scaling of the calculated descriptor matrix.

        :param stype: {'standard' or 'minmax'} type of scaling to be used
        :param fit: {boolean} defines whether the used scaler is first fitting on the data (True) or
            whether the already fitted scaler in :py:attr:`scaler` should be used to transform (False).
        :return: scaled descriptor values in :py:attr:`descriptor`
        :Example:

        >>> D.descriptor
        array([[0.155],[0.34],[0.16235294],[-0.08842105],[0.116]])
        >>> D.feature_scaling(type='minmax',fit=True)
        array([[0.56818182],[1.],[0.5853447],[0.],[0.47714988]])
        """
        if stype in ['standard', 'minmax']:
            if stype == 'standard':
                self.scaler = StandardScaler()
            elif stype == 'minmax':
                self.scaler = MinMaxScaler()
            
            if fit:
                self.descriptor = self.scaler.fit_transform(self.descriptor)
            else:
                self.descriptor = self.scaler.transform(self.descriptor)
        else:
            print("Unknown scaler type!\nAvailable: 'standard', 'minmax'")
    
    def feature_shuffle(self):
        """Method for shuffling feature columns randomly.

        :return: descriptor matrix with shuffled feature columns in :py:attr:`descriptor`
        :Example:

        >>> D.descriptor
        array([[0.80685625,167.05234375,39.56818125,-0.26338667,155.16888667,33.48778]])
        >>> D.feature_shuffle()
        array([[155.16888667,-0.26338667,167.05234375,0.80685625,39.56818125,33.48778]])
        """
        self.descriptor = shuffle(self.descriptor.transpose()).transpose()
    
    def sequence_order_shuffle(self):
        """Method for shuffling sequence order in the attribute :py:attr:`sequences`.

        :return: sequences in :py:attr:`sequences` with shuffled order in the list.
        :Example:

        >>> D.sequences
        ['LILRALKGAARALKVA','VKIAKIALKIIKGLG','VGVRLIKGIGRVARGAI','LRGLRGVIRGGKAIVRVGK','GGKLVRLIARIGKGV']
        >>> D.sequence_order_shuffle()
        >>> D.sequences
        ['VGVRLIKGIGRVARGAI','LILRALKGAARALKVA','LRGLRGVIRGGKAIVRVGK','GGKLVRLIARIGKGV','VKIAKIALKIIKGLG']
        """
        self.sequences = shuffle(self.sequences)
    
    def random_selection(self, num):
        """Method to randomly select a specified number of sequences (with names and descriptors if present) out of a given
        descriptor instance.

        :param num: {int} number of entries to be randomly selected
        :return: updated instance
        :Example:

        >>> h = Helices(7, 28, 100)
        >>> h.generate_helices()
        >>> desc = PeptideDescriptor(h.sequences, 'eisenberg')
        >>> desc.calculate_moment()
        >>> len(desc.sequences)
        100
        >>> len(desc.descriptor)
        100
        >>> desc.random_selection(10)
        >>> len(desc.descriptor)
        10
        >>> len(desc.descriptor)
        10

        .. versionadded:: v2.2.3
        """
        
        sel = np.random.choice(len(self.sequences), size=num, replace=False)
        self.sequences = np.array(self.sequences)[sel].tolist()
        if hasattr(self, 'descriptor') and self.descriptor.size:
            self.descriptor = self.descriptor[sel]
        if hasattr(self, 'names') and self.names:
            self.names = np.array(self.names)[sel].tolist()
        if hasattr(self, 'target') and self.target.size:
            self.target = self.target[sel]
    
    def minmax_selection(self, iterations, distmetric='euclidean', seed=0):
        """Method to select a specified number of sequences according to the minmax algorithm.

        :param iterations: {int} Number of sequences to retrieve.
        :param distmetric: Distance metric to calculate the distances between the sequences in descriptor space.
            Choose from 'euclidean' or 'minkowsky'.
        :param seed: {int} Set a random seed for numpy to pick the first sequence.
        :return: updated instance

        .. seealso:: **SciPy** http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        """
        
        # Storing M into pool, where selections get deleted
        pool = self.descriptor  # Store pool where selections get deleted
        minmaxidx = list()  # Store original indices of selections to return
        
        # Randomly selecting first peptide into the sele
        np.random.seed(seed)
        idx = int(np.random.random_integers(0, len(pool), 1))
        sele = pool[idx:idx + 1, :]
        minmaxidx.append(int(*np.where(np.all(self.descriptor == pool[idx:idx + 1, :], axis=1))))
        
        # Deleting peptide in selection from pool
        pool = np.delete(pool, idx, axis=0)
        
        for i in range(iterations - 1):
            # Calculating distance from sele to the rest of the peptides
            dist = distance.cdist(pool, sele, distmetric)
            
            # Choosing maximal distances for every sele instance
            maxidx = np.argmax(dist, axis=0)
            maxcols = np.max(dist, axis=0)
            
            # Choosing minimal distance among the maximal distances
            minmax = np.argmin(maxcols)
            maxidx = int(maxidx[minmax])
            
            # Adding it to selection and removing from pool
            sele = np.append(sele, pool[maxidx:maxidx + 1, :], axis=0)
            pool = np.delete(pool, maxidx, axis=0)
            minmaxidx.append(int(*np.where(np.all(self.descriptor == pool[maxidx:maxidx + 1, :], axis=1))))
        
        self.sequences = np.array(self.sequences)[minmaxidx].tolist()
        if hasattr(self, 'descriptor') and self.descriptor.size:
            self.descriptor = self.descriptor[minmaxidx]
        if hasattr(self, 'names') and self.names:
            self.names = np.array(self.names)[minmaxidx].tolist()
        if hasattr(self, 'target') and self.target.size:
            self.target = self.descriptor[minmaxidx]
    
    def filter_sequences(self, sequences):
        """Method to filter out entries for given sequences in *sequences* out of a descriptor instance. All
        corresponding attribute values of these sequences (e.g. in :py:attr:`descriptor`, :py:attr:`name`) are deleted
        as well. The method returns an updated descriptor instance.

        :param sequences: {list} sequences to be filtered out of the whole instance, including corresponding data
        :return: updated instance without filtered sequences
        :Example:

        >>> sequences = ['KLLKLLKKLLKLLK', 'ACDEFGHIK', 'GLFDIVKKVV', 'GLFDIVKKVVGALG', 'GLFDIVKKVVGALGSL']
        >>> desc = PeptideDescriptor(sequences, 'pepcats')
        >>> desc.calculate_crosscorr(7)
        >>> len(desc.descriptor)
        5
        >>> desc.filter_sequences('KLLKLLKKLLKLLK')
        >>> len(desc.descriptor)
        4
        >>> desc.sequences
        ['ACDEFGHIK', 'GLFDIVKKVV', 'GLFDIVKKVVGALG', 'GLFDIVKKVVGALGSL']
        """
        indices = list()
        if isinstance(sequences, basestring):  # check if sequences is only one sequence string and convert it to a list
            sequences = [sequences]
        for s in sequences:  # get indices of queried sequences
            indices.append(self.sequences.index(s))
        
        self.sequences = np.delete(np.array(self.sequences), indices, 0).tolist()
        if hasattr(self, 'descriptor') and self.descriptor.size:
            self.descriptor = np.delete(self.descriptor, indices, 0)
        if hasattr(self, 'names') and self.names:
            self.names = np.delete(np.array(self.names), indices, 0).tolist()
        if hasattr(self, 'target') and self.target.size:
            self.target = np.delete(self.target, indices, 0)
    
    def filter_values(self, values, operator='=='):
        """Method to filter the descriptor matrix in the attribute :py:attr:`descriptor` for a given list of values (same
        size as the number of features in the descriptor matrix!) The operator option tells the method whether to
        filter for values equal, lower, higher ect. to the given values in the *values* array.

        :param values: {list} values to filter the attribute :py:attr:`descriptor` for
        :param operator: {str} filter criterion, available the operators ``==``, ``<``, ``>``, ``<=``and ``>=``.
        :return: descriptor matrix and updated sequences containing only entries with descriptor values given in
            *values* in the corresponding attributes.
        :Example:

        >>> desc.descriptor  # desc = BaseDescriptor instance
        array([[ 0.7666517 ],
               [ 0.38373498]])
        >>> desc.filter_values([0.5], '<')
        >>> desc.descriptor
        array([[ 0.38373498]])
        """
        dim = self.descriptor.shape[1]
        for d in range(dim):  # for all the features in self.descriptor
            if operator == '==':
                indices = np.where(self.descriptor[:, d] == values[d])[0]
            elif operator == '<':
                indices = np.where(self.descriptor[:, d] < values[d])[0]
            elif operator == '>':
                indices = np.where(self.descriptor[:, d] > values[d])[0]
            elif operator == '<=':
                indices = np.where(self.descriptor[:, d] <= values[d])[0]
            elif operator == '>=':
                indices = np.where(self.descriptor[:, d] >= values[d])[0]
            
            # filter descriptor matrix, sequence list and names list according to obtained indices
            self.sequences = np.array(self.sequences)[indices].tolist()
            if hasattr(self, 'descriptor') and self.descriptor.size:
                self.descriptor = self.descriptor[indices]
            if hasattr(self, 'names') and self.names:
                self.names = np.array(self.names)[indices].tolist()
            if hasattr(self, 'target') and self.target.size:
                self.target = self.target[indices]
    
    def filter_aa(self, amino_acids):
        """Method to filter out corresponding names and descriptor values of sequences with given amino acids in the
        argument list *aminoacids*.

        :param amino_acids: list of amino acids to be filtered
        :return: filtered list of sequences, descriptor values, target values and names in the corresponding attributes.
        :Example:

        >>> b = BaseSequence(3)
        >>> b.sequences = ['AAALLLIIIKKK', 'CCEERRT', 'LLVVIIFFFQQ']
        >>> b.filter_aa(['C'])
        >>> b.sequences
        ['AAALLLIIIKKK', 'LLVVIIFFFQQ']
        """
        
        pattern = re.compile('|'.join(amino_acids))
        seqs = []
        desc = []
        names = []
        target = []
        
        for i, s in enumerate(self.sequences):
            if not pattern.search(s):
                seqs.append(s)
                if hasattr(self, 'descriptor') and self.descriptor.size:
                    desc.append(self.descriptor[i])
                if hasattr(self, 'names') and self.names:
                    names.append(self.names[i])
                if hasattr(self, 'target') and self.target.size:
                    target.append(self.target[i])
        
        self.sequences = seqs
        self.names = names
        self.descriptor = np.array(desc)
        self.target = np.array(target, dtype='int')
    
    def filter_duplicates(self):
        """Method to filter duplicates in the sequences from the class attribute :py:attr:`sequences`

        :return: filtered sequences list in the attribute :py:attr:`sequences` and corresponding names.
        :Example:

        >>> b = BaseDescriptor(['KLLKLLKKLLKLLK', 'KLLKLLKKLLKLLK', 'KLAKLAKKLAKLAK', 'KLAKLAKKLAKLAK'])
        >>> b.filter_duplicates()
        >>> b.sequences
        ['KLLKLLKKLLKLLK', 'KLAKLAKKLAKLAK']

        .. versionadded:: v2.2.5
        """
        if not self.names:
            self.names = ['Seq_' + str(i) for i in range(len(self.sequences))]
        if not self.target:
            self.target = [0] * len(self.sequences)
        if not self.descriptor:
            self.descriptor = np.zeros(len(self.sequences))
        df = pd.DataFrame(np.array([self.sequences, self.names, self.descriptor, self.target]).T,
                          columns=['Sequences', 'Names', 'Descriptor', 'Target'])
        df = df.drop_duplicates('Sequences', 'first')  # keep first occurrence of duplicate
        self.sequences = df['Sequences'].get_values().tolist()
        self.names = df['Names'].get_values().tolist()
        self.descriptor = df['Descriptor'].get_values()
        self.target = df['Target'].get_values()
    
    def keep_natural_aa(self):
        """Method to filter out sequences that do not contain natural amino acids. If the sequence contains a character
        that is not in ['A','C','D,'E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'].

        :return: filtered sequence list in the attribute :py:attr:`sequences`. The other attributes are also filtered
            accordingly (if present).
        :Example:

        >>> b = BaseSequence(2)
        >>> b.sequences = ['BBBsdflUasUJfBJ', 'GLFDIVKKVVGALGSL']
        >>> b.keep_natural_aa()
        >>> b.sequences
        ['GLFDIVKKVVGALGSL']
        """
        
        natural_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                      'Y']
        
        seqs = []
        desc = []
        names = []
        target = []
        
        for i, s in enumerate(self.sequences):
            seq = list(s.upper())
            if all(c in natural_aa for c in seq):
                seqs.append(s.upper())
                if hasattr(self, 'descriptor') and self.descriptor.size:
                    desc.append(self.descriptor[i])
                if hasattr(self, 'names') and self.names:
                    names.append(self.names[i])
                if hasattr(self, 'target') and self.target.size:
                    target.append(self.target[i])
        
        self.sequences = seqs
        self.names = names
        self.descriptor = np.array(desc)
        self.target = np.array(target, dtype='int')
    
    def load_descriptordata(self, filename, delimiter=",", targets=False, skip_header=0):
        """Method to load any data file with sequences and descriptor values and save it to a new insatnce of the
        class :class:`modlamp.descriptors.PeptideDescriptor`.

        .. note:: Headers are not considered. To skip initial lines in the file, use the *skip_header* option.

        :param filename: {str} filename of the data file to be loaded
        :param delimiter: {str} column delimiter
        :param targets: {boolean} whether last column in the file contains a target class vector
        :param skip_header: {int} number of initial lines to skip in the file
        :return: loaded sequences, descriptor values and targets in the corresponding attributes.
        """
        data = np.genfromtxt(filename, delimiter=delimiter, skip_header=skip_header)
        data = data[:, 1:]  # skip sequences as they are "nan" when read as float
        seqs = np.genfromtxt(filename, delimiter=delimiter, dtype="str")
        seqs = seqs[:, 0]
        if targets:
            self.target = np.array(data[:, -1], dtype='int')
        self.sequences = seqs
        self.descriptor = data
    
    def save_descriptor(self, filename, delimiter=',', targets=None, header=None):
        """Method to save the descriptor values to a .csv/.txt file

        :param filename: filename of the output file
        :param delimiter: column delimiter
        :param targets: target class vector to be added to descriptor (same length as :py:attr:`sequences`)
        :param header: {str} header to be written at the beginning of the file (if ``None``: feature names are taken)
        :return: output file with peptide names and descriptor values
        """
        seqs = np.array(self.sequences, dtype='|S80')[:, np.newaxis]
        ids = np.array(self.names, dtype='|S80')[:, np.newaxis]
        if ids.shape == seqs.shape:
            names = np.hstack((ids, seqs))
        else:
            names = seqs
        if targets and len(targets) == len(self.sequences):
            target = np.array(targets)[:, np.newaxis]
            data = np.hstack((names, self.descriptor, target))
        else:
            data = np.hstack((names, self.descriptor))
        if not header:
            featurenames = [['Sequence']] + self.featurenames
            header = ', '.join([f[0] for f in featurenames])
        np.savetxt(filename, data, delimiter=delimiter, fmt='%s', header=header)


def load_scale(scalename):
    """Method to load scale values for a given amino acid scale

    :param scalename: amino acid scale name, for available scales see the
        :class:`modlamp.descriptors.PeptideDescriptor()` documentation.
    :return: amino acid scale values in dictionary format.
    """
    # predefined amino acid scales dictionary
    scales = {
        'aasi': {'A': [1.89], 'C': [1.73], 'D': [3.13], 'E': [3.14], 'F': [1.53], 'G': [2.67], 'H': [3], 'I': [1.97],
                 'K': [2.28], 'L': [1.74], 'M': [2.5], 'N': [2.33], 'P': [0.22], 'Q': [3.05], 'R': [1.91], 'S': [2.14],
                 'T': [2.18], 'V': [2.37], 'W': [2], 'Y': [2.01]},
        'abhprk': {'A': [0, 0, 0, 0, 0, 0], 'C': [0, 0, 0, 0, 0, 0], 'D': [1, 0, 0, 1, 0, 0], 'E': [1, 0, 0, 1, 0, 0],
                   'F': [0, 0, 1, 0, 1, 0], 'G': [0, 0, 0, 0, 0, 0], 'H': [0, 0, 0, 1, 1, 0], 'I': [0, 0, 1, 0, 0, 0],
                   'K': [0, 1, 0, 1, 0, 0], 'L': [0, 0, 1, 0, 0, 0], 'M': [0, 0, 1, 0, 0, 0], 'N': [0, 0, 0, 1, 0, 0],
                   'P': [0, 0, 0, 0, 0, 1], 'Q': [0, 0, 0, 1, 0, 0], 'R': [0, 1, 0, 1, 0, 0], 'S': [0, 0, 0, 1, 0, 0],
                   'T': [0, 0, 0, 1, 0, 0], 'V': [0, 0, 1, 0, 0, 0], 'W': [0, 0, 1, 0, 1, 0], 'Y': [0, 0, 0, 1, 1, 0]},
        'argos': {'I': [0.77], 'F': [1.2], 'V': [0.14], 'L': [2.3], 'W': [0.07], 'M': [2.3], 'A': [0.64], 'G': [-0.48],
                  'C': [0.25], 'Y': [-0.41], 'P': [-0.31], 'T': [-0.13], 'S': [-0.25], 'H': [-0.87], 'E': [-0.94],
                  'N': [-0.89], 'Q': [-0.61], 'D': [-1], 'K': [-1], 'R': [-0.68]},
        'bulkiness': {'A': [0.443], 'C': [0.551], 'D': [0.453], 'E': [0.557], 'F': [0.898], 'G': [0], 'H': [0.563],
                      'I': [0.985], 'K': [0.674], 'L': [0.985], 'M': [0.703], 'N': [0.516], 'P': [0.768], 'Q': [0.605],
                      'R': [0.596], 'S': [0.332], 'T': [0.677], 'V': [0.995], 'W': [1], 'Y': [0.801]},
        'charge_phys': {'A': [0.], 'C': [-.1], 'D': [-1.], 'E': [-1.], 'F': [0.], 'G': [0.], 'H': [0.1],
                        'I': [0.], 'K': [1.], 'L': [0.], 'M': [0.], 'N': [0.], 'P': [0.], 'Q': [0.],
                        'R': [1.], 'S': [0.], 'T': [0.], 'V': [0.], 'W': [0.], 'Y': [0.]},
        'charge_acid': {'A': [0.], 'C': [-.1], 'D': [-1.], 'E': [-1.], 'F': [0.], 'G': [0.], 'H': [1.],
                        'I': [0.], 'K': [1.], 'L': [0.], 'M': [0.], 'N': [0.], 'P': [0.], 'Q': [0.],
                        'R': [1.], 'S': [0.], 'T': [0.], 'V': [0.], 'W': [0.], 'Y': [0.]},
        'cougar': {'A': [0.25, 0.62, 1.89], 'C': [0.208, 0.29, 1.73], 'D': [0.875, -0.9, 3.13],
                   'E': [0.833, -0.74, 3.14], 'F': [0.042, 1.2, 1.53], 'G': [1, 0.48, 2.67], 'H': [0.083, -0.4, 3],
                   'I': [0.667, 1.4, 1.97], 'K': [0.708, -1.5, 2.28], 'L': [0.292, 1.1, 1.74], 'M': [0, 0.64, 2.5],
                   'N': [0.667, -0.78, 2.33], 'P': [0.875, 0.12, 0.22], 'Q': [0.792, -0.85, 3.05],
                   'R': [0.958, -2.5, 1.91], 'S': [0.875, -0.18, 2.14], 'T': [0.583, -0.05, 2.18],
                   'V': [0.375, 1.1, 2.37], 'W': [0.042, 0.81, 2], 'Y': [0.5, 0.26, 2.01]},
        'eisenberg': {'I': [1.4], 'F': [1.2], 'V': [1.1], 'L': [1.1], 'W': [0.81], 'M': [0.64], 'A': [0.62],
                      'G': [0.48], 'C': [0.29], 'Y': [0.26], 'P': [0.12], 'T': [-0.05], 'S': [-0.18], 'H': [-0.4],
                      'E': [-0.74], 'N': [-0.78], 'Q': [-0.85], 'D': [-0.9], 'K': [-1.5], 'R': [-2.5]},
        'ez': {'A': [-0.29, 10.22, 4.67], 'C': [0.95, 13.69, 5.77], 'D': [1.19, 14.25, 8.98], 'E': [1.3, 14.66, 4.16],
               'F': [-0.8, 19.67, 7.12], 'G': [-0.01, 13.86, 6], 'H': [0.75, 12.26, 2.77], 'I': [-0.56, 14.34, 10.69],
               'K': [1.66, 11.11, 2.09], 'L': [-0.64, 17.34, 8.61], 'M': [-0.28, 18.04, 7.13], 'N': [0.89, 12.78, 6.28],
               'P': [0.83, 18.09, 3.53], 'Q': [1.21, 10.46, 2.59], 'R': [1.55, 9.34, 4.68], 'S': [0.1, 13.86, 6],
               'T': [0.01, 13.86, 6], 'V': [-0.47, 11.35, 4.97], 'W': [-0.85, 11.65, 7.2], 'Y': [-0.42, 13.04, 6.2]},
        'flexibility': {'A': [0.25], 'C': [0.208], 'D': [0.875], 'E': [0.833], 'F': [0.042], 'G': [1], 'H': [0.083],
                        'I': [0.667], 'K': [0.708], 'L': [0.292], 'M': [0.], 'N': [0.667], 'P': [0.875], 'Q': [0.792],
                        'R': [0.958], 'S': [0.875], 'T': [0.583], 'V': [0.375], 'W': [0.042], 'Y': [0.5]},
        'grantham': {'A': [0, 8.1, 31], 'C': [2.75, 5.5, 55], 'D': [1.38, 13.0, 54], 'E': [0.92, 12.3, 83],
               'F': [0, 5.2, 132], 'G': [0.74, 9.0, 3], 'H': [0.58, 10.4, 96], 'I': [0, 5.2, 111],
               'K': [0.33, 11.3, 119], 'L': [0, 4.9, 111], 'M': [0, 5.7, 105], 'N': [1.33, 11.6, 56],
               'P': [0.39, 8.0, 32.5], 'Q': [0.89, 10.5, 85], 'R': [0.65, 10.5, 124], 'S': [1.42, 9.2, 32],
               'T': [0.71, 8.6, 61], 'V': [0, 5.9, 84], 'W': [0.13, 5.4, 170], 'Y': [0.20, 6.2, 136]},
        'gravy': {'I': [4.5], 'V': [4.2], 'L': [3.8], 'F': [2.8], 'C': [2.5], 'M': [1.9], 'A': [1.8], 'G': [-0.4],
                  'T': [-0.7], 'W': [-0.9], 'S': [-0.8], 'Y': [-1.3], 'P': [-1.6], 'H': [-3.2], 'E': [-3.5],
                  'Q': [-3.5], 'D': [-3.5], 'N': [-3.5], 'K': [-3.9], 'R': [-4.5]},
        'hopp-woods': {'A': [-0.5], 'C': [-1], 'D': [3], 'E': [3], 'F': [-2.5], 'G': [0], 'H': [-0.5], 'I': [-1.8],
                       'K': [3], 'L': [-1.8], 'M': [-1.3], 'N': [0.2], 'P': [0], 'Q': [0.2], 'R': [3], 'S': [0.3],
                       'T': [-0.4], 'V': [-1.5], 'W': [-3.4], 'Y': [-2.3]},
        'isaeci': {'A': [62.9, 0.05], 'C': [78.51, 0.15], 'D': [18.46, 1.25], 'E': [30.19, 1.31], 'F': [189.42, 0.14],
                   'G': [19.93, 0.02], 'H': [87.38, 0.56], 'I': [149.77, 0.09], 'K': [102.78, 0.53], 'L': [154.35, 0.1],
                   'M': [132.22, 0.34], 'N': [19.53, 1.36], 'P': [122.35, 0.16], 'Q': [17.87, 1.31], 'R': [52.98, 1.69],
                   'S': [19.75, 0.56], 'T': [59.44, 0.65], 'V': [120.91, 0.07], 'W': [179.16, 1.08],
                   'Y': [132.16, 0.72]},
        'janin': {'I': [1.2], 'F': [0.87], 'V': [1], 'L': [0.87], 'W': [0.59], 'M': [0.73], 'A': [0.59], 'G': [0.59],
                  'C': [1.4], 'Y': [-0.4], 'P': [-0.26], 'T': [-0.12], 'S': [0.02], 'H': [0.02], 'E': [-0.83],
                  'N': [-0.55], 'Q': [-0.83], 'D': [-0.69], 'K': [-2.4], 'R': [-1.8]},
        'kytedoolittle': {'I': [1.7], 'F': [1.1], 'V': [1.6], 'L': [1.4], 'W': [-0.14], 'M': [0.8], 'A': [0.77],
                          'G': [0.03], 'C': [1], 'Y': [-0.27], 'P': [-0.37], 'T': [-0.07], 'S': [-0.1], 'H': [-0.91],
                          'E': [-1], 'N': [-1], 'Q': [-1], 'D': [-1], 'K': [-1.1], 'R': [-1.3]},
        'levitt_alpha': {'A': [1.29], 'C': [1.11], 'D': [1.04], 'E': [1.44], 'F': [1.07], 'G': [0.56], 'H': [1.22],
                         'I': [0.97], 'K': [1.23], 'L': [1.3], 'M': [1.47], 'N': [0.9], 'P': [0.52], 'Q': [1.27],
                         'R': [0.96], 'S': [0.82], 'T': [0.82], 'V': [0.91], 'W': [0.99], 'Y': [0.72]},
        'mss': {'A': [13.02], 'C': [23.7067], 'D': [22.02], 'E': [20.0233], 'F': [23.5288], 'G': [1.01], 'H': [23.5283],
                'I': [22.3611], 'K': [18.9756], 'L': [19.6944], 'M': [21.92], 'N': [21.8567], 'P': [19.0242],
                'Q': [19.9689], 'R': [19.0434], 'S': [18.3533], 'T': [22.3567], 'V': [21.0267], 'W': [26.1975],
                'Y': [24.1954]},
        'msw': {'A': [-0.73, 0.2, -0.62], 'C': [-0.66, 0.26, -0.27], 'D': [0.11, -1, -0.96], 'E': [0.24, -0.39, -0.04],
                'F': [0.76, 0.85, -0.34], 'G': [-0.31, -0.28, -0.75], 'H': [0.84, 0.67, -0.78],
                'I': [-0.91, 0.83, -0.25], 'K': [-0.51, 0.08, 0.6], 'L': [-0.74, 0.72, -0.16], 'M': [-0.7, 1, -0.32],
                'N': [0.14, 0.2, -0.66], 'P': [-0.43, 0.73, -0.6], 'Q': [0.3, 1, -0.3], 'R': [-0.22, 0.27, 1],
                'S': [-0.8, 0.61, -1], 'T': [-0.58, 0.85, -0.89], 'V': [-1, 0.79, -0.58], 'W': [1, 0.98, -0.47],
                'Y': [0.97, 0.66, -0.16]},
        'pepcats': {'A': [1, 0, 0, 0, 0, 0], 'C': [1, 0, 1, 1, 0, 0], 'D': [0, 0, 1, 0, 0, 1], 'E': [0, 0, 1, 0, 0, 1],
                    'F': [1, 1, 0, 0, 0, 0], 'G': [0, 0, 0, 0, 0, 0], 'H': [1, 1, 0, 1, 1, 0], 'I': [1, 0, 0, 0, 0, 0],
                    'K': [1, 0, 0, 1, 1, 0], 'L': [1, 0, 0, 0, 0, 0], 'M': [1, 0, 1, 0, 0, 0], 'N': [0, 0, 1, 1, 0, 0],
                    'P': [1, 0, 0, 0, 0, 0], 'Q': [0, 0, 1, 1, 0, 0], 'R': [1, 0, 0, 1, 1, 0], 'S': [0, 0, 1, 1, 0, 0],
                    'T': [0, 0, 1, 1, 0, 0], 'V': [1, 0, 0, 0, 0, 0], 'W': [1, 1, 0, 1, 0, 0], 'Y': [1, 1, 1, 1, 0, 0]},
        'peparc': {'A': [1, 0, 0, 0, 0], 'C': [0, 1, 0, 0, 0], 'D': [0, 1, 0, 1, 0], 'E': [0, 1, 0, 1, 0],
                   'F': [1, 0, 0, 0, 0], 'G': [0, 0, 0, 0, 0], 'H': [0, 1, 1, 0, 0], 'I': [1, 0, 0, 0, 0],
                   'K': [0, 1, 1, 0, 0], 'L': [1, 0, 0, 0, 0], 'M': [1, 0, 0, 0, 0], 'N': [0, 1, 0, 0, 0],
                   'P': [0, 0, 0, 0, 1], 'Q': [0, 1, 0, 0, 0], 'R': [0, 1, 1, 0, 0], 'S': [0, 1, 0, 0, 0],
                   'T': [0, 1, 0, 0, 0], 'V': [1, 0, 0, 0, 0], 'W': [1, 0, 0, 0, 0], 'Y': [1, 0, 0, 0, 0]},
        'polarity': {'A': [0.395], 'C': [0.074], 'D': [1.], 'E': [0.914], 'F': [0.037], 'G': [0.506], 'H': [0.679],
                     'I': [0.037], 'K': [0.79], 'L': [0.], 'M': [0.099], 'N': [0.827], 'P': [0.383], 'Q': [0.691],
                     'R': [0.691], 'S': [0.531], 'T': [0.457], 'V': [0.123], 'W': [0.062], 'Y': [0.16]},
        'ppcali': {
            'A': [0.070781, 0.036271, 2.042, 0.083272, 0.69089, 0.15948, -0.80893, 0.24698, 0.86525, 0.68563, -0.24665,
                  0.61314, -0.53343, -0.50878, -1.3646, 2.2679, -1.5644, -0.75043, -0.65875],
            'C': [0.61013, -0.93043, -0.85983, -2.2704, 1.5877, -2.0066, -0.30314, 1.2544, -0.2832, -1.2844, -0.73449,
                  -0.11235, -0.41152, -0.0050164, 0.28307, 0.20522, -0.021084, -0.15627, -0.32689],
            'D': [-1.3215, 0.24063, -0.032754, -0.37863, 1.2051, 1.0001, 2.1827, 0.19212, -0.60529, 0.37639, -0.46451,
                  -0.46788, 1.4077, -2.1661, 0.72604, -0.12332, -0.8243, -0.082989, 0.053476],
            'E': [-0.87713, 1.4905, 1.0755, 0.35944, 1.567, 0.41365, 1.0944, 0.72634, -0.74957, 0.038939, 0.075057,
                  0.78637, -1.4543, 1.6667, -0.097439, -0.24293, 1.7687, 0.36174, -0.11585],
            'F': [1.3557, -0.10336, -0.4309, 0.41269, -0.083356, 0.83783, 0.095381, -0.65222, -0.3119, 0.43293, -1.0011,
                  -0.66855, -0.10242, 1.2066, 2.6234, 1.9981, -0.25016, 0.71979, 0.21569],
            'G': [-1.0818, -2.1561, 0.77082, -0.92747, -1.0748, 1.7997, -1.3708, 1.279, -1.2098, 0.46065, 0.43076,
                  0.20037, -0.2302, 0.2646, 0.57149, -0.68432, 0.19341, -0.061606, -0.08071],
            'H': [-0.050161, 0.69246, -0.88397, -0.64601, 0.24622, 0.10487, -1.1317, -2.3661, -0.89918, 0.46391,
                  -0.62359, 2.5478, -0.34737, -0.52062, 0.17522, -0.88648, -0.4755, 0.023187, -0.28261],
            'I': [1.4829, -0.46435, 0.50189, 0.55724, -0.51535, -0.29914, 0.97236, -0.15793, -0.98246, -0.54347,
                  0.97806, 0.37577, 1.618, 0.62323, -0.59359, -0.35483, -0.085017, 0.55825, -2.7542],
            'K': [-0.85344, 1.529, 0.27747, 0.32993, -1.1786, -0.16633, -1.0459, 0.44621, 0.41027, -2.5318, 0.91329,
                  0.53385, 0.61417, -1.111, 1.1323, 0.95105, 0.76769, -0.016115, 0.054995],
            'L': [1.2857, 0.039488, 1.5378, 0.87969, -0.21419, 0.40389, -0.20426, -0.14351, 0.61024, -1.1927, -2.2149,
                  -0.84248, -0.5061, -0.48548, 0.10791, -2.1503, -0.12006, -0.60222, 0.26546],
            'M': [1.137, 0.64388, 0.13724, -0.2988, 1.2288, 0.24981, -1.6427, -0.75868, -0.54902, 1.0571, 1.272,
                  -1.9104, 0.70919, -0.93575, -0.6314, -0.079654, 1.634, -0.0021923, 0.49825],
            'N': [-1.084, -0.176, -0.47062, -0.92245, -0.32953, 0.74278, 0.34551, -1.4605, 0.25219, -1.2107, -0.59978,
                  -0.79183, 1.3268, 1.9839, -1.6137, 0.5333, 0.033889, -1.0331, 0.83019],
            'P': [-1.1823, -1.6911, -1.1331, 3.073, 1.1942, -0.93426, -0.72985, -0.042441, -0.19264, -0.21603, -0.1239,
                  0.054016, 0.15241, -0.019691, -0.20543, 0.10206, 0.07671, -0.081968, 0.20348],
            'Q': [-0.57747, 0.97452, -0.077547, -0.0033488, 0.17184, -0.52537, -0.27362, -0.1366, 0.2057, -0.013066,
                  1.8834, -1.2736, -0.84991, 1.0445, 0.69027, -1.2866, -2.6776, 0.1683, 0.086105],
            'R': [-0.62245, 1.545, -0.61966, 0.19057, -1.7485, -1.3909, -0.47526, 1.3938, -0.84556, 1.7344, -1.6516,
                  -0.52678, 0.6791, 0.24374, -0.62551, -0.0028271, -0.053884, 0.14926, -0.17232],
            'S': [-0.86409, -0.77147, 0.38542, -0.59389, -0.53313, -0.47585, 0.31966, -0.89716, 1.8029, 0.26431,
                  -0.23173, -0.37626, -0.47349, -0.42878, -0.47297, -0.079826, 0.57043, 3.2057, -0.18413],
            'T': [-0.33027, -0.57447, 0.18653, -0.28941, -0.62681, -1.0737, 0.80363, -0.59525, 1.8786, 1.3971, 0.63929,
                  0.21281, -0.067048, 0.096271, 1.323, -0.36173, 1.2261, -2.2771, -0.65412],
            'V': [1.1675, -0.61554, 0.95405, 0.11662, -0.74473, -1.1482, 1.1309, 0.12079, -0.77171, 0.18597, 0.93442,
                  1.201, 0.3826, -0.091573, -0.31269, 0.074367, -0.22946, 0.24322, 2.9836],
            'W': [1.1881, 0.43789, -1.7915, 0.138, 0.43088, 1.6467, -0.11987, 1.7369, 2.0818, 0.33122, 0.31829, 1.1586,
                  0.67649, 0.30819, -0.55772, -0.54491, -0.17969, 0.24477, 0.38674],
            'Y': [0.54671, -0.1468, -1.5688, 0.19001, -1.2736, 0.66162, 1.1614, -0.18614, -0.70654, -0.43634, 0.44775,
                  -0.71366, -2.5907, -1.1649, -1.1576, 0.66572, 0.21019, -0.61016, -0.34844]},
        'refractivity': {'A': [0.102045615], 'C': [0.841053374], 'D': [0.282153774], 'E': [0.405831178],
                         'F': [0.691276746], 'G': [0], 'H': [0.512814484], 'I': [0.448154244], 'K': [0.50058782],
                         'L': [0.441570656], 'M': [0.508817305], 'N': [0.282153774], 'P': [0.256995062],
                         'Q': [0.405831178], 'R': [0.626851634], 'S': [0.149306372], 'T': [0.258876087],
                         'V': [0.327298378], 'W': [1], 'Y': [0.741359041]},
        't_scale': {'A': [-8.4, -8.01, -3.73, -3.65, -6.12, -1.59, 1.56],
                    'C': [-2.44, -1.96, 0.93, -2.35, 1.31, 2.29, -1.52],
                    'D': [-6.84, -0.94, 17.68, -0.03, 3.44, 9.07, 4.32],
                    'E': [-6.5, 16.2, 17.28, 3.11, -4.75, -2.54, 4.72],
                    'F': [21.59, -5.73, 1.03, -3.3, 2.64, -5.02, 1.7],
                    'G': [-8.48, -10.37, -5.14, -6.51, -11.84, -3.6, 2.01],
                    'H': [15.28, -3.67, 6.72, -6.38, 4.12, -1.55, -2.85],
                    'I': [-2.97, 4.64, -0.77, 11, 3.26, -4.36, -7.88],
                    'K': [2.7, 13.46, -14.03, -2.55, 2.77, 0.15, 3.19],
                    'L': [2.61, 5.96, 1.97, 2.59, -4.77, -4.84, -5.44],
                    'M': [3.38, 12.43, -4.77, 0.45, -1.55, -0.6, 3.26],
                    'N': [-3.11, -1.22, 6.26, -9.38, 9.94, 7.66, -4.81],
                    'P': [-5.35, -9.07, -1.52, -8.79, -8.73, 4.29, -9.91],
                    'Q': [-5.31, 15.64, 8.44, 1.03, -4.32, -4.4, -0.52],
                    'R': [-2.27, 18.9, -18.24, -3.47, 3.03, 6.64, 0.45],
                    'S': [-15.88, -11.21, -2.44, -3.61, 3.46, -0.37, 8.98],
                    'T': [-17.81, -13.64, -5.19, 10.57, 6.91, -4.43, 3.49],
                    'V': [-5.8, -6.15, -2.26, 9.87, 5.28, -1.49, -7.54],
                    'W': [21.68, -8.78, -2.53, 15.53, -8.15, 11.98, 3.23],
                    'Y': [23.9, -6.47, 0.31, -4.14, 4.08, -7.28, 3.59]},
        'tm_tend': {'A': [0.38], 'C': [-0.3], 'D': [-3.27], 'E': [-2.9], 'F': [1.98], 'G': [-0.19], 'H': [-1.44],
                    'I': [1.97], 'K': [-3.46], 'L': [1.82], 'M': [1.4], 'N': [-1.62], 'P': [-1.44], 'Q': [-1.84],
                    'R': [-2.57], 'S': [-0.53], 'T': [-0.32], 'V': [1.46], 'W': [1.53], 'Y': [0.49]},
        'z3': {'A': [0.07, -1.73, 0.09], 'C': [0.71, -0.97, 4.13], 'D': [3.64, 1.13, 2.36], 'E': [3.08, 0.39, -0.07],
               'F': [-4.92, 1.3, 0.45], 'G': [2.23, -5.36, 0.3], 'H': [2.41, 1.74, 1.11], 'I': [-4.44, -1.68, -1.03],
               'K': [2.84, 1.41, -3.14], 'L': [-4.19, -1.03, -0.98], 'M': [-2.49, -0.27, -0.41],
               'N': [3.22, 1.45, 0.84], 'P': [-1.22, 0.88, 2.23], 'Q': [2.18, 0.53, -1.14], 'R': [2.88, 2.52, -3.44],
               'S': [1.96, -1.63, 0.57], 'T': [0.92, -2.09, -1.4], 'V': [-2.69, -2.53, -1.29], 'W': [-4.75, 3.65, 0.85],
               'Y': [-1.39, 2.32, 0.01]},
        'z5': {'A': [0.24, -2.32, 0.6, -0.14, 1.3], 'C': [0.84, -1.67, 3.71, 0.18, -2.65],
               'D': [3.98, 0.93, 1.93, -2.46, 0.75], 'E': [3.11, 0.26, -0.11, -3.04, -0.25],
               'F': [-4.22, 1.94, 1.06, 0.54, -0.62], 'G': [2.05, -4.06, 0.36, -0.82, -0.38],
               'H': [2.47, 1.95, 0.26, 3.9, 0.09], 'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
               'K': [2.29, 0.89, -2.49, 1.49, 0.31], 'L': [-4.28, -1.3, -1.49, -0.72, 0.84],
               'M': [-2.85, -0.22, 0.47, 1.94, -0.98], 'N': [3.05, 1.62, 1.04, -1.15, 1.61],
               'P': [-1.66, 0.27, 1.84, 0.7, 2], 'Q': [1.75, 0.5, -1.44, -1.34, 0.66],
               'R': [3.52, 2.5, -3.5, 1.99, -0.17], 'S': [2.39, -1.07, 1.15, -1.39, 0.67],
               'T': [0.75, -2.18, -1.12, -1.46, -0.4], 'V': [-2.59, -2.64, -1.54, -0.85, -0.02],
               'W': [-4.36, 3.94, 0.59, 3.44, -1.59], 'Y': [-2.54, 2.44, 0.43, 0.04, -1.47]}
    }
    if scalename == 'all':
        d = {'I': [], 'F': [], 'V': [], 'L': [], 'W': [], 'M': [], 'A': [], 'G': [], 'C': [], 'Y': [], 'P': [],
             'T': [], 'S': [], 'H': [], 'E': [], 'N': [], 'Q': [], 'D': [], 'K': [], 'R': []}
        for scale in scales.keys():
            for k, v in scales[scale].items():
                d[k].extend(v)
        return 'all', d

    elif scalename == 'instability':
        d = {
            "A": {"A": 1.0, "C": 44.94, "E": 1.0, "D": -7.49, "G": 1.0, "F": 1.0, "I": 1.0, "H": -7.49, "K": 1.0,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": 1.0,
                  "Y": 1.0},
            "C": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 20.26, "G": 1.0, "F": 1.0, "I": 1.0, "H": 33.6, "K": 1.0,
                  "M": 33.6, "L": 20.26, "N": 1.0, "Q": -6.54, "P": 20.26, "S": 1.0, "R": 1.0, "T": 33.6, "W": 24.68,
                  "V": -6.54, "Y": 1.0},
            "E": {"A": 1.0, "C": 44.94, "E": 33.6, "D": 20.26, "G": 1.0, "F": 1.0, "I": 20.26, "H": -6.54, "K": 1.0,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 20.26, "R": 1.0, "T": 1.0, "W": -14.03,
                  "V": 1.0, "Y": 1.0},
            "D": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": -6.54, "I": 1.0, "H": 1.0, "K": -7.49,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 1.0, "S": 20.26, "R": -6.54, "T": -14.03, "W": 1.0,
                  "V": 1.0, "Y": 1.0},
            "G": {"A": -7.49, "C": 1.0, "E": -6.54, "D": 1.0, "G": 13.34, "F": 1.0, "I": -7.49, "H": 1.0, "K": -7.49,
                  "M": 1.0, "L": 1.0, "N": -7.49, "Q": 1.0, "P": 1.0, "S": 1.0, "R": 1.0, "T": -7.49, "W": 13.34,
                  "V": 1.0, "Y": -7.49},
            "F": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 13.34, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": -14.03,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": 1.0,
                  "Y": 33.601},
            "I": {"A": 1.0, "C": 1.0, "E": 44.94, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 13.34, "K": -7.49,
                  "M": 1.0, "L": 20.26, "N": 1.0, "Q": 1.0, "P": -1.88, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0,
                  "V": -7.49, "Y": 1.0},
            "H": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -9.37, "F": -9.37, "I": 44.94, "H": 1.0, "K": 24.68,
                  "M": 1.0, "L": 1.0, "N": 24.68, "Q": 1.0, "P": -1.88, "S": 1.0, "R": 1.0, "T": -6.54, "W": -1.88,
                  "V": 1.0, "Y": 44.94},
            "K": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -7.49, "F": 1.0, "I": -7.49, "H": 1.0, "K": 1.0,
                  "M": 33.6, "L": -7.49, "N": 1.0, "Q": 24.64, "P": -6.54, "S": 1.0, "R": 33.6, "T": 1.0, "W": 1.0,
                  "V": -7.49, "Y": 1.0},
            "M": {"A": 13.34, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 58.28, "K": 1.0,
                  "M": -1.88, "L": 1.0, "N": 1.0, "Q": -6.54, "P": 44.94, "S": 44.94, "R": -6.54, "T": -1.88, "W": 1.0,
                  "V": 1.0, "Y": 24.68},
            "L": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": -7.49, "M": 1.0,
                  "L": 1.0, "N": 1.0, "Q": 33.6, "P": 20.26, "S": 1.0, "R": 20.26, "T": 1.0, "W": 24.68, "V": 1.0,
                  "Y": 1.0},
            "N": {"A": 1.0, "C": -1.88, "E": 1.0, "D": 1.0, "G": -14.03, "F": -14.03, "I": 44.94, "H": 1.0, "K": 24.68,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": -6.54, "P": -1.88, "S": 1.0, "R": 1.0, "T": -7.49, "W": -9.37,
                  "V": 1.0, "Y": 1.0},
            "Q": {"A": 1.0, "C": -6.54, "E": 20.26, "D": 20.26, "G": 1.0, "F": -6.54, "I": 1.0, "H": 1.0, "K": 1.0,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 44.94, "R": 1.0, "T": 1.0, "W": 1.0,
                  "V": -6.54, "Y": -6.54},
            "P": {"A": 20.26, "C": -6.54, "E": 18.38, "D": -6.54, "G": 1.0, "F": 20.26, "I": 1.0, "H": 1.0, "K": 1.0,
                  "M": -6.54, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 20.26, "R": -6.54, "T": 1.0, "W": -1.88,
                  "V": 20.26, "Y": 1.0},
            "S": {"A": 1.0, "C": 33.6, "E": 20.26, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": 1.0, "M": 1.0,
                  "L": 1.0, "N": 1.0, "Q": 20.26, "P": 44.94, "S": 20.26, "R": 20.26, "T": 1.0, "W": 1.0, "V": 1.0,
                  "Y": 1.0},
            "R": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -7.49, "F": 1.0, "I": 1.0, "H": 20.26, "K": 1.0,
                  "M": 1.0, "L": 1.0, "N": 13.34, "Q": 20.26, "P": 20.26, "S": 44.94, "R": 58.28, "T": 1.0, "W": 58.28,
                  "V": 1.0, "Y": -6.54},
            "T": {"A": 1.0, "C": 1.0, "E": 20.26, "D": 1.0, "G": -7.49, "F": 13.34, "I": 1.0, "H": 1.0, "K": 1.0,
                  "M": 1.0, "L": 1.0, "N": -14.03, "Q": -6.54, "P": 1.0, "S": 1.0, "R": 1.0, "T": 1.0, "W": -14.03,
                  "V": 1.0, "Y": 1.0},
            "W": {"A": -14.03, "C": 1.0, "E": 1.0, "D": 1.0, "G": -9.37, "F": 1.0, "I": 1.0, "H": 24.68, "K": 1.0,
                  "M": 24.68, "L": 13.34, "N": 13.34, "Q": 1.0, "P": 1.0, "S": 1.0, "R": 1.0, "T": -14.03, "W": 1.0,
                  "V": -7.49, "Y": 1.0},
            "V": {"A": 1.0, "C": 1.0, "E": 1.0, "D": -14.03, "G": -7.49, "F": 1.0, "I": 1.0, "H": 1.0, "K": -1.88,
                  "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": -7.49, "W": 1.0,
                  "V": 1.0, "Y": -6.54},
            "Y": {"A": 24.68, "C": 1.0, "E": -6.54, "D": 24.68, "G": -7.49, "F": 1.0, "I": 1.0, "H": 13.34, "K": 1.0,
                  "M": 44.94, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 13.34, "S": 1.0, "R": -15.91, "T": -7.49, "W": -9.37,
                  "V": 1.0, "Y": 13.34}}
        return 'instability', d

    else:
        return scalename, scales[scalename]


def read_fasta(inputfile):
    """Method for loading sequences from a FASTA formatted file into :py:attr:`sequences` & :py:attr:`names`.
    This method is used by the base class :class:`modlamp.descriptors.PeptideDescriptor` if the input is a FASTA file.

    :param inputfile: .fasta file with sequences and headers to read
    :return: list of sequences in the attribute :py:attr:`sequences` with corresponding sequence names in
        :py:attr:`names`.
    """
    names = list()  # list for storing names
    sequences = list()  # list for storing sequences
    seq = str()
    with open(inputfile) as f:
        all = f.readlines()
        last = all[-1]
        for line in all:
            if line.startswith('>'):
                names.append(line.split(' ')[0][1:].strip())  # add FASTA name without description as molecule name
                sequences.append(seq.strip())
                seq = str()
            elif line == last:
                seq += line.strip()  # remove potential white space
                sequences.append(seq.strip())
            else:
                seq += line.strip()  # remove potential white space
    return sequences[1:], names


def save_fasta(filename, sequences, names=None):
    """Method for saving sequences in the instance :py:attr:`sequences` to a file in FASTA format.

    :param filename: {str} output filename (ending .fasta)
    :param sequences: {list} sequences to be saved to file
    :param names: {list} whether sequence names from self.names should be saved as sequence identifiers
    :return: a FASTA formatted file containing the generated sequences
    """
    if os.path.exists(filename):
        os.remove(filename)  # remove outputfile, it it exists

    with open(filename, 'w') as o:
        for n, seq in enumerate(sequences):
            if names:
                o.write('>' + str(names[n]) + '\n')
            else:
                o.write('>Seq_' + str(n) + '\n')
            o.write(seq + '\n')


def aa_weights():
    """Function holding molecular weight data on all natural amino acids.

    :return: dictionary with amino acid letters and corresponding weights

    .. versionadded:: v2.4.1
    """
    weights = {'A': 89.093, 'C': 121.158, 'D': 133.103, 'E': 147.129, 'F': 165.189, 'G': 75.067,
               'H': 155.155, 'I': 131.173, 'K': 146.188, 'L': 131.173, 'M': 149.211, 'N': 132.118,
               'P': 115.131, 'Q': 146.145, 'R': 174.20, 'S': 105.093, 'T': 119.119, 'V': 117.146,
               'W': 204.225, 'Y': 181.189}
    return weights


def count_aas(seq, scale='relative'):
    """Function to count the amino acids occuring in a given sequence.

    :param seq: {str} amino acid sequence
    :param scale: {'absolute' or 'relative'} defines whether counts or frequencies are given for each AA
    :return: {dict} dictionary with amino acids as keys and their counts in the sequence as values.
    """
    if seq == '':  # error if len(seq) == 0
        seq = ' '
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    scl = 1.
    if scale == 'relative':
        scl = len(seq)
    aa = {a: (float(seq.count(a)) / scl) for a in aas}
    aa = collections.OrderedDict(sorted(aa.items()))
    return aa


def aa_energies():
    """Function holding free energies of transfer between cyclohexane and water for all natural amino acids.
    H. G. Boman, D. Wade, I. a Boman, B. Wåhlin, R. B. Merrifield, *FEBS Lett*. **1989**, *259*, 103–106.

    :return: dictionary with amino acid letters and corresponding energies.
    """
    energies = {'L': -4.92, 'I': -4.92, 'V': -4.04, 'F': -2.98, 'M': -2.35, 'W': -2.33, 'A': -1.81, 'C': -1.28,
                'G': -0.94, 'Y': 0.14, 'T': 2.57, 'S': 3.40, 'H': 4.66, 'Q': 5.54, 'K': 5.55, 'N': 6.64, 'E': 6.81,
                'D': 8.72, 'R': 14.92, 'P': 0.}
    return energies


def ngrams_apd():
    """Function returning the most frequent 2-, 3- and 4-grams from all sequences in the `APD3
    <http://aps.unmc.edu/AP/>`_, version August 2016 with 2727 sequences.
    For all 2, 3 and 4grams, all possible ngrams were generated from all sequences and the top 50 most frequent
    assembled into a list. Finally, leading and tailing spaces were striped and duplicates as well as ngrams containing
    spaces were removed.

    :return: numpy.array containing most frequent ngrams
    """
    ngrams = np.array(['AGK', 'CKI', 'RR', 'YGGG', 'LSGL', 'RG', 'YGGY', 'PRP', 'LGGG',
                       'GV', 'GT', 'GS', 'GR', 'IAG', 'GG', 'GF', 'GC', 'GGYG', 'GA', 'GL',
                       'GK', 'GI', 'IPC', 'KAA', 'LAK', 'GLGG', 'GGLG', 'CKIT', 'GAGK',
                       'LLSG', 'LKK', 'FLP', 'LSG', 'SCK', 'LLS', 'GETC', 'VLG', 'GKLL',
                       'LLG', 'C', 'KCKI', 'G', 'VGK', 'CSC', 'TKKC', 'GCS', 'GKA', 'IGK',
                       'GESC', 'KVCY', 'KKL', 'KKI', 'KKC', 'LGGL', 'GLL', 'CGE', 'GGYC',
                       'GLLS', 'GLF', 'AKK', 'GKAA', 'ESCV', 'GLP', 'CGES', 'PCGE', 'FL',
                       'CGET', 'GLW', 'KGAA', 'KAAL', 'GGY', 'GGG', 'IKG', 'LKG', 'GGL',
                       'CK', 'GTC', 'CG', 'SKKC', 'CS', 'CR', 'KC', 'AGKA', 'KA', 'KG',
                       'LKCK', 'SCKL', 'KK', 'KI', 'KN', 'KL', 'SK', 'KV', 'SL', 'SC',
                       'SG', 'AAA', 'VAK', 'AAL', 'AAK', 'GGGG', 'KNVA', 'GGGL', 'GYG',
                       'LG', 'LA', 'LL', 'LK', 'LS', 'LP', 'GCSC', 'TC', 'GAA', 'AA', 'VA',
                       'VC', 'AG', 'VG', 'AI', 'AK', 'VL', 'AL', 'TPGC', 'IK', 'IA', 'IG',
                       'YGG', 'LGK', 'CSCK', 'GYGG', 'LGG', 'KGA'],
                      dtype='|S4')
    return ngrams


def aa_formulas():
    """
    Function returning the molecular formulas of all amino acids. All amino acids are considered in the neutral form
    (uncharged).
    """
    formulas = {'A': {'C': 3, 'H': 7, 'N': 1, 'O': 2, 'S': 0},
                'C': {'C': 3, 'H': 7, 'N': 1, 'O': 2, 'S': 1},
                'D': {'C': 4, 'H': 7, 'N': 1, 'O': 4, 'S': 0},
                'E': {'C': 5, 'H': 9, 'N': 1, 'O': 4, 'S': 0},
                'F': {'C': 9, 'H': 11, 'N': 1, 'O': 2, 'S': 0},
                'G': {'C': 2, 'H': 5, 'N': 1, 'O': 2, 'S': 0},
                'H': {'C': 6, 'H': 9, 'N': 3, 'O': 2, 'S': 0},
                'I': {'C': 6, 'H': 13, 'N': 1, 'O': 2, 'S': 0},
                'K': {'C': 6, 'H': 14, 'N': 2, 'O': 2, 'S': 0},
                'L': {'C': 6, 'H': 13, 'N': 1, 'O': 2, 'S': 0},
                'M': {'C': 5, 'H': 11, 'N': 1, 'O': 2, 'S': 1},
                'N': {'C': 4, 'H': 8, 'N': 2, 'O': 3, 'S': 0},
                'P': {'C': 5, 'H': 9, 'N': 1, 'O': 2, 'S': 0},
                'Q': {'C': 4, 'H': 7, 'N': 1, 'O': 4, 'S': 0},
                'R': {'C': 6, 'H': 14, 'N': 4, 'O': 2, 'S': 0},
                'S': {'C': 3, 'H': 7, 'N': 1, 'O': 3, 'S': 0},
                'T': {'C': 4, 'H': 9, 'N': 1, 'O': 3, 'S': 0},
                'V': {'C': 5, 'H': 11, 'N': 1, 'O': 2, 'S': 0},
                'W': {'C': 11, 'H': 12, 'N': 2, 'O': 2, 'S': 0},
                'Y': {'C': 9, 'H': 11, 'N': 1, 'O': 3, 'S': 0}
                }
    return formulas

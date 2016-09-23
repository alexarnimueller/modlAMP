# -*- coding: utf-8 -*-
"""
.. module:: modlamp.descriptors

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates different classes to calculate peptide descriptor values. The following classes are available:

=============================        ============================================================================
Class                                Characteristics
=============================        ============================================================================
:py:class:`GlobalDescriptor`         Global one-dimensional peptide descriptors calculated from the AA sequence.
:py:class:`PeptideDescriptor`        AA scale based global or convoluted descriptors (auto-/cross-correlated).
=============================        ============================================================================

"""

import collections
import os
from os.path import dirname, join
import sys
import json

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
from sklearn.externals.joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle

from core import load_scale, read_fasta, save_fasta, filter_unnatural, filter_values, filter_aa, count_aa, \
    random_selection, minmax_selection, filter_sequences, filter_duplicates, keep_natural_aa, aa_weights, aa_energies

__author__ = 'modlab'
__docformat__ = "restructuredtext en"


def _one_autocorr(seq, window, scale):
    """Private function used for calculating auto-correlated descriptors for 1 given sequence, window and an AA scale.
    This function is used by the :py:func:`calculate_autocorr` method of :py:class:`PeptideDescriptor`.

    :param seq: {str} amino acid sequence to calculate descriptor for
    :param window: {int} correlation-window size
    :param scale: {str} amino acid scale to be used to calculate descriptor
    :return: {numpy.array} calculated descriptor data
    """
    m = list()  # list of lists to store translated sequence values
    for l in range(len(seq)):  # translate AA sequence into values
        m.append(scale[str(seq[l])])
    # auto-correlation in defined sequence window
    seqdesc = list()
    for dist in range(window):  # for all correlation distances
        for val in range(len(scale['A'])):  # for all features of the descriptor scale
            valsum = list()
            cntr = 0.
            for pos in range(len(seq)):  # for every position in the sequence
                if (pos + dist) < len(seq):  # check if corr distance is possible at that sequence position
                    cntr += 1  # counter to scale sum
                    valsum.append(m[pos][val] * m[pos + dist][val])
            seqdesc.append(sum(valsum) / cntr)  # append scaled correlation distance values
    return seqdesc


def _one_crosscorr(seq, window, scale):
    """Private function used for calculating cross-correlated descriptors for 1 given sequence, window and an AA scale.
    This function is used by the :py:func:`calculate_crosscorr` method of :py:class:`PeptideDescriptor`.

    :param seq: {str} amino acid sequence to calculate descriptor for
    :param window: {int} correlation-window size
    :param scale: {str} amino acid scale to be used to calculate descriptor
    :return: {numpy.array} calculated descriptor data
    """
    m = list()  # list of lists to store translated sequence values
    for l in range(len(seq)):  # translate AA sequence into values
        m.append(scale[str(seq[l])])
    # auto-correlation in defined sequence window
    seqdesc = list()
    for val in range(len(scale['A'])):  # for all features of the descriptor scale
        for cc in range(len(scale['A'])):  # for every feature cross correlation
            if (val + cc) < len(scale['A']):  # check if corr distance is in range of the num of features
                for dist in range(window):  # for all correlation distances
                    cntr = float()
                    valsum = list()
                    for pos in range(len(seq)):  # for every position in the sequence
                        if (pos + dist) < len(seq):  # check if corr distance is possible at that sequence pos
                            cntr += 1  # counter to scale sum
                            valsum.append(m[pos][val] * m[pos + dist][val + cc])
                    seqdesc.append(sum(valsum) / cntr)  # append scaled correlation distance values
    return seqdesc


def _charge(seq, ph=7.0, amide=False):
    """Calculates charge of a single sequence. Adapted from Bio.SeqUtils.IsoelectricPoint.IsoelectricPoint_chargeR
    function. The method used is first described by Bjellqvist. In the case of amidation, the value for the
    'Cterm' pKa is 15 (and Cterm is added to the pos_pks dictionary.
    The pKa scale is extracted from: http://www.hbcpnetbase.com/ (CRC Handbook of Chemistry and Physics, 96th ed).
    Further references: see the `Biopython <http://biopython.org/>`_ module :mod:`Bio.SeqUtils.IsoelectricPoint`.`

    **pos_pks** = {'Nterm': 9.38, 'K': 10.67, 'R': 12.10, 'H': 6.04}

    **neg_pks** = {'Cterm': 2.15, 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}

    :param ph: {float} pH at which to calculate peptide charge.
    :param amide: {boolean} whether the sequences have an amidated C-terminus.
    :return: {array} descriptor values in the attribute :py:attr:`descriptor
    """
    
    if amide:
        pos_pks = {'Nterm': 9.38, 'K': 10.67, 'R': 12.10, 'H': 6.04}
        neg_pks = {'Cterm': 15., 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}
    else:
        pos_pks = {'Nterm': 9.38, 'K': 10.67, 'R': 12.10, 'H': 6.04}
        neg_pks = {'Cterm': 2.15, 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}
    
    aa_content = count_aa(seq)
    aa_content['Nterm'] = 1.0
    aa_content['Cterm'] = 1.0
    pos_charge = 0.0
    for aa, pK in pos_pks.items():
        c_r = 10 ** (pK - ph)
        partial_charge = c_r / (c_r + 1.0)
        pos_charge += aa_content[aa] * partial_charge
    neg_charge = 0.0
    for aa, pK in neg_pks.items():
        c_r = 10 ** (ph - pK)
        partial_charge = c_r / (c_r + 1.0)
        neg_charge += aa_content[aa] * partial_charge
    return round(pos_charge - neg_charge, 3)


class GlobalDescriptor(object):
    """
    Base class for global, non-amino acid scale dependant descriptors. The following descriptors can be calculated by
    the **methods** linked below:

    - `Sequence Charge      <modlamp.html#modlamp.descriptors.GlobalDescriptor.calculate_charge>`_
    - `Molecular Weight     <modlamp.html#modlamp.descriptors.GlobalDescriptor.calculate_MW>`_
    - `Sequence Length      <modlamp.html#modlamp.descriptors.GlobalDescriptor.length>`_
    - `Isoelectric Point    <modlamp.html#modlamp.descriptors.GlobalDescriptor.isoelectric_point>`_
    - `Charge Density       <modlamp.html#modlamp.descriptors.GlobalDescriptor.charge_density>`_
    - `Hydrophobic Ratio    <modlamp.html#modlamp.descriptors.GlobalDescriptor.hydrophobic_ratio>`_
    - `Aromaticity          <modlamp.html#modlamp.descriptors.GlobalDescriptor.aromaticity>`_
    - `Boman Index          <modlamp.html#modlamp.descriptors.GlobalDescriptor.boman_index>`_
    - `Aliphatic Index      <modlamp.html#modlamp.descriptors.GlobalDescriptor.aliphatic_index>`_
    - `Instability Index    <modlamp.html#modlamp.descriptors.GlobalDescriptor.instability_index>`_

    Most of the methods calculate values with help of the :mod:`Bio.SeqUtils.ProtParam` module of
    `Biopython <http://biopython.org/>`_.
    """

    def __init__(self, seqs):
        """
        :param seqs: a .fasta file with sequences, a list of sequences or a single sequence as string to calculate the
            descriptor values for.
        :return: initialized lists self.sequences, self.names and dictionary self.AA with amino acid scale values
        :Example:

        >>> from modlamp.descriptors import GlobalDescriptor
        >>> desc = GlobalDescriptor('KLAKLAKKLAKLAK')
        >>> desc.sequences
        ['KLAKLAKKLAKLAK']
        >>> desc = GlobalDescriptor('/Path/to/file.fasta')  # load sequences from .fasta file
        >>> desc.sequences
        ['AFDGHLKI','KKLQRSDLLRTK','KKLASCNNIPPR'...]
        """
        d = PeptideDescriptor(seqs, 'eisenberg')
        self.sequences = d.sequences
        self.names = d.names
        self.descriptor = d.descriptor
        self.target = d.target
        self.scaler = None

    def length(self, append=False):
        """
        Method to calculate the length (total AA count) of every sequence in the attribute :py:attr:`sequences`.

        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: array of sequence lengths in the attribute :py:attr:`descriptor`
        :Example:
        
        >>> desc = GlobalDescriptor(['AFDGHLKI','KKLQRSDLLRTK','KKLASCNNIPPR'])
        >>> desc.length()
        >>> desc.descriptor
        array([[ 8], [12], [12]])
        """
        desc = []
        for seq in self.sequences:
            desc.append(len(seq.strip()))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def calculate_MW(self, amide=False, append=False):
        """Method to calculate the molecular weight [g/mol] of every sequence in the attribute :py:attr:`sequences`.

        :param amide: {boolean} whether the sequences are C-terminally amidated (subtracts 0.95 from the MW).
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: array of descriptor values in the attribute :py:attr:`descriptor`
        :Example:
        
        >>> desc = GlobalDescriptor('IAESFKGHIPL')
        >>> desc.calculate_MW(amide=True)
        >>> desc.descriptor
        array([[ 1210.43]])

        .. seealso:: :py:func:`modlamp.core.aa_weights()`

        .. versionchanged:: v2.1.5 amide option added
        """
        desc = []
        weights = aa_weights()
        for seq in self.sequences:
            mw = []
            for aa in seq:  # sum over aa weights
                mw.append(weights[aa])
            desc.append(round(sum(mw) - 18.015 * (len(seq) - 1), 2))  # sum over AA MW and subtract H20 MW for every
            # peptide bond
        desc = np.asarray(desc).reshape(len(desc), 1)
        if amide:  # if sequences are amidated, subtract 0.98 from calculated MW
            desc = [d - 0.98 for d in desc]
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def calculate_charge(self, ph=7.0, amide=False, append=False):
        """Method to overall charge of every sequence in the attribute :py:attr:`sequences`.
        Adapted from Bio.SeqUtils.IsoelectricPoint.IsoelectricPoint_chargeR function.

        The method used is first described by Bjellqvist. In the case of amidation, the value for the 'Cterm' pKa is 15
        (and Cterm is added to the pos_pKs dictionary.
        The pKa scale is extracted from: http://www.hbcpnetbase.com/ (CRC Handbook of Chemistry and Physics, 96th ed).
        Further references: see the `Biopython <http://biopython.org/>`_ module :mod:`Bio.SeqUtils.IsoelectricPoint`.`

        **pos_pKs** = {'Nterm': 9.38, 'K': 10.67, 'R': 12.10, 'H': 6.04}

        **neg_pKs** = {'Cterm': 2.15, 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}

        :param ph: {float} ph at which to calculate peptide charge.
        :param amide: {boolean} whether the sequences have an amidated C-terminus.
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: array of descriptor values in the attribute :py:attr:`descriptor`
        :Example:
        
        >>> desc = GlobalDescriptor('KLAKFGKRSELVALSG')
        >>> desc.calculate_charge(ph=7.4, amide=True)
        >>> desc.descriptor
        array([[ 3.989]])
        """

        desc = []
        for seq in self.sequences:
            desc.append(_charge(seq, ph, amide))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def charge_density(self, ph=7.0, amide=False, append=False):
        """Method to calculate the charge density (charge / MW) of every sequences in the attributes :py:attr:`sequences`

        :param ph: {float} pH at which to calculate peptide charge.
        :param amide: {boolean} whether the sequences have an amidated C-terminus.
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: array of descriptor values in the attribute :py:attr:`descriptor`.
        :Example:
        
        >>> desc = GlobalDescriptor('GNSDLLIEQRTLLASDEF')
        >>> desc.charge_density(ph=6, amide=True)
        >>> desc.descriptor
        array([[-0.00097119]])
        """
        self.calculate_charge(ph, amide)
        charges = self.descriptor
        self.calculate_MW(amide)
        masses = self.descriptor
        desc = charges / masses
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def isoelectric_point(self, amide=False, append=False):
        """
        Method to calculate the isoelectric point of every sequence in the attribute :py:attr:`sequences`.
        The pK scale is extracted from: http://www.hbcpnetbase.com/ (CRC Handbook of Chemistry and Physics, 96th ed).
        The method used is based on the IsoelectricPoint module in `Biopython <http://biopython.org/>`_
        module :mod:`Bio.SeqUtils.ProtParam`.

         **pos_pKs** = {'Nterm': 9.38, 'K': 10.67, 'R': 12.10, 'H': 6.04}

         **neg_pKs** = {'Cterm': 2.15, 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}

        :param amide: {boolean} whether the sequences have an amidated C-terminus.
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: array of descriptor values in the attribute :py:attr:`descriptor`
        :Example:
        
        >>> desc = GlobalDescriptor('KLFDIKFGHIPQRST')
        >>> desc.isoelectric_point()
        >>> desc.descriptor
        array([[ 10.6796875]])
        """

        desc = []
        for seq in self.sequences:

            # Bracket between ph1 and ph2
            ph = 7.0
            charge = _charge(seq, ph, amide)
            if charge > 0.0:
                ph1 = ph
                charge1 = charge
                while charge1 > 0.0:
                    ph = ph1 + 1.0
                    charge = _charge(seq, ph, amide)
                    if charge > 0.0:
                        ph1 = ph
                        charge1 = charge
                    else:
                        ph2 = ph
                        break
            else:
                ph2 = ph
                charge2 = charge
                while charge2 < 0.0:
                    ph = ph2 - 1.0
                    charge = _charge(seq, ph, amide)
                    if charge < 0.0:
                        ph2 = ph
                        charge2 = charge
                    else:
                        ph1 = ph
                        break
            # Bisection
            while ph2 - ph1 > 0.0001 and charge != 0.0:
                ph = (ph1 + ph2) / 2.0
                charge = _charge(seq, ph, amide)
                if charge > 0.0:
                    ph1 = ph
                else:
                    ph2 = ph
            desc.append(ph)
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def instability_index(self, append=False):
        """
        Method to calculate the instability of every sequence in the attribute :py:attr:`sequences`.
        The instability index is a prediction of protein stability based on the amino acid composition.
        ([1] K. Guruprasad, B. V Reddy, M. W. Pandit, Protein Eng. 1990, 4, 155–161.)

        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: array of descriptor values in the attribute :py:attr:`descriptor`
        :Example:
        
        >>> desc = GlobalDescriptor('LLASMNDLLAKRST')
        >>> desc.instability_index()
        >>> desc.descriptor
        array([[ 63.95714286]])
        """
        desc = []
        module_path = dirname(__file__)
        dimv = json.load(open(join(module_path, 'data', 'dimv.json')))
        for seq in self.sequences:
            stabindex = float()
            for i in range(len(seq) - 1):
                stabindex += dimv[seq[i]][seq[i+1]]
            desc.append((10.0 / len(seq)) * stabindex)
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def aromaticity(self, append=False):
        """
        Method to calculate the aromaticity of every sequence in the attribute :py:attr:`sequences`.
        According to Lobry, 1994, it is simply the relative frequency of Phe+Trp+Tyr.

        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: array of descriptor values in the attribute :py:attr:`descriptor`
        :Example:
        
        >>> desc = GlobalDescriptor('GLFYWRFFLQRRFLYWW')
        >>> desc.aromaticity()
        >>> desc.descriptor
        array([[ 0.52941176]])
        """
        desc = []
        for seq in self.sequences:
            f = seq.count('F')
            w = seq.count('W')
            y = seq.count('Y')
            desc.append(float(f + w + y) / len(seq))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def aliphatic_index(self, append=False):
        """
        Method to calculate the aliphatic index of every sequence in the attribute :py:attr:`sequences`.
        According to Ikai, 1980, the aliphatic index is a measure of thermal stability of proteins and is dependant
        on the relative volume occupied by aliphatic amino acids (A,I,L & V).
        ([1] A. Ikai, J. Biochem. 1980, 88, 1895–1898.)

        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: array of descriptor values in the attribute :py:attr:`descriptor`
        :Example:
        
        >>> desc = GlobalDescriptor('KWLKYLKKLAKLVK')
        >>> desc.aliphatic_index()
        >>> desc.descriptor
        array([[ 139.28571429]])
        """
        desc = []
        aa_dict = aa_weights()
        for seq in self.sequences:
            d = {aa: seq.count(aa) for aa in aa_dict.keys()}  # count aa
            d = {k: (float(d[k]) / len(seq)) * 100 for k in d.keys()}  # get mole percent of all AA
            desc.append(d['A'] + 2.9 * d['V'] + 3.9 * (d['I'] + d['L']))  # formula for calculating the AI (Ikai, 1980)
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def boman_index(self, append=False):
        """Method to calculate the boman index of every sequence in the attribute :py:attr:`sequences`.
        According to Boman, 2003, the boman index is a measure for protein-protein interactions and is calculated by
        summing over all amino acid free energy of transfer [kcal/mol] between water and cyclohexane,[2] followed by
        dividing by    sequence length.
        ([1] H. G. Boman, D. Wade, I. a Boman, B. Wåhlin, R. B. Merrifield, *FEBS Lett*. **1989**, *259*, 103–106.
        [2] A. Radzick, R. Wolfenden, *Biochemistry* **1988**, *27*, 1664–1670.)
        
        .. seealso:: :py:func:`modlamp.core.aa_energies()`

        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: array of descriptor values in the attribute :py:attr:`descriptor`
        :Example:
        
        >>> desc = GlobalDescriptor('GLFDIVKKVVGALGSL')
        >>> desc.boman_index()
        >>> desc.descriptor
        array([[-1.011875]])
        """
        d = aa_energies()
        desc = []
        for seq in self.sequences:
            val = []
            for a in seq:
                val.append(d[a])
            desc.append(sum(val) / len(val))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def hydrophobic_ratio(self, append=False):
        """
        Method to calculate the hydrophobic ratio of every sequence in the attribute :py:attr:`sequences`, which is the
        relative frequency of the amino acids **A,C,F,I,L,M & V**.

        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: array of descriptor values in the attribute :py:attr:`descriptor`
        :Example:
        
        >>> desc = GlobalDescriptor('VALLYWRTVLLAIII')
        >>> desc.hydrophobic_ratio()
        >>> desc.descriptor
        array([[ 0.73333333]])
        """
        desc = []
        aa_dict = aa_weights()
        for seq in self.sequences:
            pa = {aa: seq.count(aa) for aa in aa_dict.keys()}  # count aa
            # formula for calculating the AI (Ikai, 1980):
            desc.append((pa['A'] + pa['C'] + pa['F'] + pa['I'] + pa['L'] + pa['M'] + pa['V']) / float(len(seq)))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def feature_scaling(self, stype='standard', fit=True):
        """Method for feature scaling of the calculated descriptor matrix.

        :param stype: {str} **'standard'** or **'minmax'**, type of scaling to be used
        :param fit: {boolean}, defines whether the used scaler is first fitting on the data (True) or
            whether the already fitted scaler in :py:attr:`scaler` should be used to transform (False).
        :return: scaled descriptor values in :py:attr:`self.descriptor`
        :Example:

        >>> desc.descriptor
        array([[0.155],[0.34],[0.16235294],[-0.08842105],[0.116]])
        >>> desc.feature_scaling(stype='minmax',fit=True)
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
            print "Unknown scaler type!\nAvailable: 'standard', 'minmax'"

    def feature_shuffle(self):
        """Method for shuffling features randomly.

        :return: descriptor matrix with shuffled feature columns in the attribute :py:attr:`descriptor`
        :Example:

        >>> desc.descriptor
        array([[0.80685625,167.05234375,39.56818125,-0.26338667,155.16888667,33.48778]])
        >>> desc.feature_shuffle()
        array([[155.16888667,-0.26338667,167.05234375,0.80685625,39.56818125,33.48778]])
        """
        self.descriptor = shuffle(self.descriptor.transpose()).transpose()

    def filter_values(self, values, operator='=='):
        """Method to filter the descriptor matrix in the attribute :py:attr:`descriptor` for a given list of values (same
        size as the number of features in the descriptor matrix!) The operator option tells the method whether to
        filter for values equal, lower, higher ect. to the given values in the **values** array.

        :param values: List/array of values to filter the attribute :py:attr:`descriptor` for
        :param operator: filter criterion, available are all SQL like operators: ``==``, ``<``, ``>``, ``<=``and ``>=``.
        :return: filtered descriptor matrix and updated sequences in the corresponding attributes.
        :Example:
        
        >>> min(desc.descriptor)
        array([ 0.04523998])
        >>> desc.descriptor.shape
        (412, 1)
        >>> desc.filter_values([0.1], operator='>')
        >>> min(desc.descriptor)
        array([ 0.10344828])
        >>> desc.descriptor.shape
        (168, 1)

        .. seealso:: :func:`modlamp.core.filter_values()`
        """
        filter_values(self, values, operator)

    def filter_duplicates(self):
        """
        Method to filter duplicates in the sequences from the class attribute :py:attr:`sequences`

        :return: filtered sequences list in the attribute :py:attr:`sequences`
        :Example:
        
        >>> desc = GlobalDescriptor(['KLLKLLKKLLKLLK', 'KLLKLLKKLLKLLK', 'GLFDIVKKVVGGKKASSERT'])
        >>> desc.filter_duplicates()
        >>> desc.sequences
        ['KLLKLLKKLLKLLK', 'GLFDIVKKVVGGKKASSERT']
        
        .. seealso:: :func:`modlamp.core.filter_sequences()`

        .. versionadded:: v2.2.5
        """
        filter_duplicates(self)

    def keep_natural_aa(self):
        """Method to filter out sequences that do not contain natural amino acids. If the sequence contains a character
        that is not in ['A','C','D,'E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'].

        :return: filtered sequence list in the attribute :py:attr:`sequences`. The other attributes are also filtered
            accordingly.
        :Example:
        
        >>> desc = GlobalDescriptor(['ACDEFGHIKLMNPQRSTVWY', 'JOGHURT', 'ELVISISKING'])
        >>> desc.keep_natural_aa()
        >>> desc.sequences
        ['ACDEFGHIKLMNPQRSTVWY', 'ELVISISKING']

        .. seealso:: :func:`modlamp.core.keep_natural_aa()`

        .. versionadded:: v2.2.5
        """
        keep_natural_aa(self)

    def filter_aa(self, aminoacids):
        """Method to filter sequences and corresponding descriptor values, if the sequences contain any of the given
        amino acids in the argument list **aminoacids**.

        :param aminoacids: List/array of amino acids {str.upper} to filter for
        :return: filtered descriptor matrix and updated sequences and names in the corresponding attributes.
        :Example:
        
        >>> desc = GlobalDescriptor(['KLLKLLKKLLKLLK', 'ACCDFACAD'])
        >>> desc.filter_aa(['C'])
        >>> desc.sequences
        ['KLLKLLKKLLKLLK']
        
        .. seealso:: :func:`modlamp.core.filter_aa()`
        """
        filter_aa(self, aminoacids)

    def filter_sequences(self, sequences):
        """Method to filter out entries for given sequences in *sequences* out of a descriptor instance. All
        corresponding fields of these sequences (*descriptor*, *name*) are deleted as well. The method returns an
        updated descriptor instance.

        :param sequences: {list} sequences to be filtered out of the whole instance, including corresponding data
        :return: updated instance
        :Example:

        >>> desc = GlobalDescriptor(['KLLKLLKKLLKLLK', 'KLLK', 'KLLLKLKLKLL'])
        >>> desc.filter_sequences(['KLLK'])
        >>> desc.sequences
        ['KLLKLLKKLLKLLK', 'KLLLKLKLKLL']

        .. seealso:: :func:`modlamp.core.filter_sequences()`

        .. versionadded:: v2.2.4
        """
        filter_sequences(self, sequences)
    
    def filter_unnatural(self):
        """Method to filter out sequences with unnatural amino acids from :py:attr:`sequences`.
        :return: Filtered sequence list in the attribute :py:attr:`sequences`

        .. seealso:: :func:`modlamp.core.filter_unnatural()`
        """
        filter_unnatural(self)

    def random_selection(self, num):
        """Method to select a random number of sequences (with names and descriptors if present) out of a given
        descriptor instance.

        :param num: {int} number of entries to be randomly selected
        :return: updated instance

        .. seealso:: :func:`modlamp.core.random_selection()`

        .. versionadded:: v2.2.3
        """
        random_selection(self, num)

    def minmax_selection(self, iterations, distmetric='euclidean', randseed=0):
        """Method to select a specified number of sequences out of a given descriptor instance according to the
        minmax algorithm.

        :param iterations: {int} number of sequences to retrieve.
        :param distmetric: {str} distance metric to calculate the distances between the sequences in descriptor space.
            Choose from scipy.spacial.distance (http://docs.scipy.org/doc/scipy/reference/spatial.distance.html).
            E.g. 'euclidean', 'minkowsky'.
        :param randseed: {int} Set a random seed for numpy to pick the first sequence.
        :return: updated instance

        .. seealso:: :func:`modlamp.core.minmax_selection()`

        .. versionadded:: v2.2.6
        """
        minmax_selection(self, iterations, distmetric, randseed)

    def load_descriptordata(self, filename, delimiter=",", targets=False, header=0):
        """Method to load any data file with sequences and descriptor values and save it to a new insatnce of the
        class :class:`modlamp.descriptors.GlobalDescriptor`.

        :param filename: {str} filename of the data file to be loaded
        :param delimiter: {str} column delimiter
        :param targets: {boolean} whether last column in the file contains a target class vector
        :param header: {int} number of header lines to skip in the file
        :return: loaded sequences, descriptor values and targets in the corresponding attributes.
        """
        data = np.genfromtxt(filename, delimiter=delimiter, skip_header=header)
        data = data[:, 1:]  # skip sequences as they are "nan" when read as float
        seqs = np.genfromtxt(filename, delimiter=delimiter, dtype="str")
        seqs = seqs[:, 0]
        if targets:
            self.target = np.array(data[:, -1], dtype='int')
        self.sequences = seqs
        self.descriptor = data

    def save_descriptor(self, filename, delimiter=',', targets=None, header=''):
        """Method to save the descriptor values to a .csv/.txt file

        :param filename: {str} filename of the output file
        :param delimiter: {str} column delimiter
        :param targets: {list} target class vector to be added to descriptor (same length as :py:attr:`sequences`)
        :param header: {str} header to be written at the beginning of the file
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
        np.savetxt(filename, data, delimiter=delimiter, fmt='%s', header=header)

    def save_fasta(self, outputfile, names=False):
        """Method for saving sequences from :py:attr:`sequences` to a FASTA formatted file.

        :param outputfile: {str} filename of the output FASTA file
        :param names: {boolean} whether sequence names from self.names should be saved as sequence identifiers
        :return: list of sequences in self.sequences with corresponding sequence names in the attribute :py:attr:`names`
        """
        save_fasta(self, outputfile, names=names)


class PeptideDescriptor(object):
    """Base class for peptide descriptors. The following **amino acid descriptor scales** are available for descriptor
    calculation:

    - **AASI**           (An amino acid selectivity index scale for helical antimicrobial peptides, *[1] D. Juretić, D. Vukicević, N. Ilić, N. Antcheva, A. Tossi, J. Chem. Inf. Model. 2009, 49, 2873–2882.*)
    - **argos**          (Argos hydrophobicity amino acid scale, *[2] Argos, P., Rao, J. K. M. & Hargrave, P. A., Eur. J. Biochem. 2005, 128, 565–575.*)
    - **bulkiness**      (Amino acid side chain bulkiness scale, *[3] J. M. Zimmerman, N. Eliezer, R. Simha, J. Theor. Biol. 1968, 21, 170–201.*)
    - **charge_physio**  (Amino acid charge at pH 7.0 - Hystidine charge +0.1.)
    - **charge_acidic**  (Amino acid charge at acidic pH - Hystidine charge +1.0.)
    - **cougar**         (modlabs inhouse selection of global peptide descriptors)
    - **eisenberg**      (the Eisenberg hydrophobicity consensus amino acid scale, *[4] D. Eisenberg, R. M. Weiss, T. C. Terwilliger, W. Wilcox, Faraday Symp. Chem. Soc. 1982, 17, 109.*)
    - **Ez**             (potential that assesses energies of insertion of amino acid side chains into lipid bilayers, *[5] A. Senes, D. C. Chadi, P. B. Law, R. F. S. Walters, V. Nanda, W. F. DeGrado, J. Mol. Biol. 2007, 366, 436–448.*)
    - **flexibility**    (amino acid side chain flexibilitiy scale, *[6] R. Bhaskaran, P. K. Ponnuswamy, Int. J. Pept. Protein Res. 1988, 32, 241–255.*)
    - **gravy**          (GRAVY hydrophobicity amino acid scale, *[7] J. Kyte, R. F. Doolittle, J. Mol. Biol. 1982, 157, 105–132.*)
    - **hopp-woods**     (Hopp-Woods amino acid hydrophobicity scale,*[8] T. P. Hopp, K. R. Woods, Proc. Natl. Acad. Sci. 1981, 78, 3824–3828.*)
    - **ISAECI**         (Isotropic Surface Area (ISA) and Electronic Charge Index (ECI) of amino acid side chains, *[9] E. R. Collantes, W. J. Dunn, J. Med. Chem. 1995, 38, 2705–2713.*)
    - **janin**          (Janin hydrophobicity amino acid scale, [10] J. L. Cornette, K. B. Cease, H. Margalit, J. L. Spouge, J. A. Berzofsky, C. DeLisi, J. Mol. Biol. 1987, 195, 659–685.*)
    - **kytedoolittle**  (Kyte & Doolittle hydrophobicity amino acid scale, *[11] J. Kyte, R. F. Doolittle, J. Mol. Biol. 1982, 157, 105–132.*)
    - **levitt_alpha**   (Levitt amino acid alpha-helix propensity scale, extracted from http://web.expasy.org/protscale. *[12] M. Levitt, Biochemistry 1978, 17, 4277-4285.*)
    - **MSS**            (A graph-theoretical index that reflects topological shape and size of amino acid side chains, *[13] C. Raychaudhury, A. Banerjee, P. Bag, S. Roy, J. Chem. Inf. Comput. Sci. 1999, 39, 248–254.*)
    - **MSW**            (Amino acid scale based on a PCA of the molecular surface based WHIM descriptor (MS-WHIM), extended to natural amino acids, *[14] A. Zaliani, E. Gancia, J. Chem. Inf. Comput. Sci 1999, 39, 525–533.*)
    - **pepcats**        (modlabs pharmacophoric feature based PEPCATS scale, *[15] C. P. Koch, A. M. Perna, M. Pillong, N. K. Todoroff, P. Wrede, G. Folkers, J. A. Hiss, G. Schneider, PLoS Comput. Biol. 2013, 9, e1003088.*)
    - **polarity**       (Amino acid polarity scale, *[3] J. M. Zimmerman, N. Eliezer, R. Simha, J. Theor. Biol. 1968, 21, 170–201.*)
    - **PPCALI**         (modlabs inhouse scale derived from a PCA of 143 amino acid property scales, *[15] C. P. Koch, A. M. Perna, M. Pillong, N. K. Todoroff, P. Wrede, G. Folkers, J. A. Hiss, G. Schneider, PLoS Comput. Biol. 2013, 9, e1003088.*)
    - **refractivity**   (Relative amino acid refractivity values, *[16] T. L. McMeekin, M. Wilensky, M. L. Groves, Biochem. Biophys. Res. Commun. 1962, 7, 151–156.*)
    - **t_scale**        (A PCA derived scale based on amino acid side chain properties calculated with 6 different probes of the GRID program, *[17] M. Cocchi, E. Johansson, Quant. Struct. Act. Relationships 1993, 12, 1–8.*)
    - **TM_tend**        (Amino acid transmembrane propensity scale, extracted from http://web.expasy.org/protscale, *[18] Zhao, G., London E. Protein Sci. 2006, 15, 1987-2001.*)
    - **z3**             (The original three dimensional Z-scale, *[17] S. Hellberg, M. Sjöström, B. Skagerberg, S. Wold, J. Med. Chem. 1987, 30, 1126–1135.*)
    - **z5**             (The extended five dimensional Z-scale, *[18] M. Sandberg, L. Eriksson, J. Jonsson, M. Sjöström, S. Wold, J. Med. Chem. 1998, 41, 2481–2491.*)

    Further, amino acid scale independent methods can be calculated with help of the :class:`GlobalDescriptor` class.

    """

    def __init__(self, seqs, scalename='eisenberg'):
        """
        :param seqs: a .fasta file with sequences, a list of sequences or a single sequence as string to calculate the
            descriptor values for.
        :param scalename: {str} name of the amino acid scale (one of the given list above) used to calculate the
            descriptor values
        :return: initialized attributes :py:attr:`sequences`, :py:attr:`names` and dictionary :py:attr:`scale` with
            amino acid scale values of the scale name in :py:attr:`scalename`.
        :Example:

        >>> AMP = PeptideDescriptor('KLLKLLKKLLKLLK','pepcats')
        >>> AMP.sequences
        ['KLLKLLKKLLKLLK']
        >>> seqs = PeptideDescriptor('/Path/to/file.fasta', 'eisenberg')  # load sequences from .fasta file
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
                    for line in f:
                        if line.isupper():
                            self.sequences.append(line.strip())
            else:
                print "Sorry, currently only .fasta or .csv files can be read!"
        else:
            print "'inputfile' does not exist, is not a valid list of sequences or is not a valid sequence string"

        self.scalename, self.scale = load_scale(scalename.lower())
        self.descriptor = np.array([[]])
        self.target = np.array([], dtype='int')
        self.scaler = None
        self.all_moms = list()  # for passing hydrophobic moments to calculate_profile
        self.all_globs = list()  # for passing global  to calculate_profile

    def load_scale(self, scalename):
        """Method to load amino acid values from a given scale

        :param scalename: {str} name of the amino acid scale to be loaded.
        :return: loaded amino acid scale values in a dictionary in the attribute :py:attr:`scale`.

        .. seealso:: :func:`modlamp.core.load_scale()`
        """
        self.scalename, self.scale = load_scale(scalename.lower())

    def read_fasta(self, filename):
        """Method for loading sequences from a FASTA formatted file into the attributes :py:attr:`sequences` and
        :py:attr:`names`. This method is used by the base class :class:`PeptideDescriptor` if the input is a FASTA file.

        :param filename: {str} .fasta file with sequences and headers to read
        :return: list of sequences in self.sequences with corresponding sequence names in self.names
        """
        self.sequences, self.names = read_fasta(filename)

    def save_fasta(self, outputfile, names=False):
        """Method for saving sequences from :py:attr:`sequences` to a FASTA formatted file.

        :param outputfile: {str} filename of the output FASTA file
        :param names: {bool} whether sequence names from self.names should be saved as sequence identifiers
        :return: list of sequences in self.sequences with corresponding sequence names in the attribute :py:attr:`names`
        """
        save_fasta(self, outputfile, names=names)

    def calculate_autocorr(self, window, append=False):
        """Method for auto-correlating the amino acid values for a given descriptor scale

        :param window: {int} correlation window for descriptor calculation in a sliding window approach
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: calculated descriptor numpy.array in self.descriptor
        :Example:

        >>> AMP = PeptideDescriptor('GLFDIVKKVVGALGSL','PPCALI')
        >>> AMP.calculate_autocorr(7)
        >>> AMP.descriptor
        array([[  1.28442339e+00,   1.29025116e+00,   1.03240901e+00, .... ]])
        >>> AMP.descriptor.shape
        (1, 133)

        .. versionchanged:: v.2.3.0
        """
        desc = Parallel(n_jobs=-1)(delayed(_one_autocorr)(seq, window, self.scale) for seq in self.sequences)
        
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def calculate_crosscorr(self, window, append=False):
        """Method for cross-correlating the amino acid values for a given descriptor scale

        :param window: {int} correlation window for descriptor calculation in a sliding window approach
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: calculated descriptor numpy.array in self.descriptor
        :Example:

        >>> AMP = PeptideDescriptor('GLFDIVKKVVGALGSL','pepcats')
        >>> AMP.calculate_crosscorr(7)
        >>> AMP.descriptor
        array([[ 0.6875    ,  0.46666667,  0.42857143,  0.61538462,  0.58333333, ... ]])
        >>> AMP.descriptor.shape
        (1, 147)
        """
        desc = Parallel(n_jobs=-1)(delayed(_one_crosscorr)(seq, window, self.scale) for seq in self.sequences)
        
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def calculate_moment(self, window=1000, angle=100, modality='max', append=False):
        """Method for calculating the maximum or mean moment of the amino acid values for a given descriptor scale and
        window.

        :param window: {int} amino acid window in which to calculate the moment. If the sequence is shorter than the
            window, the length of the sequence is taken. So if the default window of 1000 is chosen, for all sequences
            shorter than 1000, the **global** hydrophobic moment will be calculated. Otherwise, the maximal
            hydrophiobic moment for the chosen window size found in the sequence will be returned.
        :param angle: {int} angle in which to calculate the moment. **100** for alpha helices, **180** for beta sheets.
        :param modality: {'max' or 'mean'} Calculate respectively maximum or mean hydrophobic moment.
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: Calculated descriptor as a numpy.array in self.descriptor and all possible global values in
            :py:attr:`all_moms` (needed for the :py:func:`calculate_profile` method)
        :Example:

        >>> AMP = PeptideDescriptor('GLFDIVKKVVGALGSL','eisenberg')
        >>> AMP.calculate_moment(window=1000, angle=100, modality='max')
        >>> AMP.descriptor
        array([[ 0.48790226]])
        """
        if self.scale['A'] == list:
            print '\n Descriptor moment calculation is only possible for one dimensional descriptors.\n'
            sys.exit()

        desc = list()
        for s, seq in enumerate(self.sequences):
            wdw = min(window, len(seq))  # if sequence is shorter than window, take the whole sequence instead
            mtrx = list()
            for l in range(len(seq)):
                mtrx.append(self.scale[str(seq[l])])

            mwdw = list()
            for i in range(len(mtrx) - wdw + 1):
                mwdw.append(sum(mtrx[i:i + wdw], []))

            mwdw = np.asarray(mwdw)
            rads = angle * (np.pi / 180) * np.asarray(range(wdw))  # calculate actual moment (radial)
            vcos = (mwdw * np.cos(rads)).sum(axis=1)
            vsin = (mwdw * np.sin(rads)).sum(axis=1)
            moms = np.sqrt(vsin * vsin + vcos * vcos) / wdw

            if modality == 'max':  # take window with maximal value
                moment = np.max(moms)
            elif modality == 'mean':  # take average value over all windows
                moment = np.mean(moms)
            else:
                print '\nModality parameter is wrong, please choose between "max" and "mean".\n'
                sys.exit()
            desc.append(moment)
            self.all_moms.append(moms)
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def calculate_global(self, window=1000, modality='max', append=False):
        """Method for calculating a global / window averaging descriptor value of a given AA scale

        :param window: {int} amino acid window in which to calculate the moment. If the sequence is shorter than the
            window, the length of the sequence is taken.
        :param modality: {'max' or 'mean'} Calculate respectively maximum or mean hydrophobic moment.
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: Calculated descriptor as numpy.array in self.descriptor and all possible global values in
            :py:attr:`all_globs` (needed for the :py:func:`calculate_profile` method)
        :Example:

        >>> AMP = PeptideDescriptor('GLFDIVKKVVGALGSL','eisenberg')
        >>> AMP.calculate_global(window=1000, modality='max')
        >>> AMP.descriptor
        array([[ 0.44875]])
        """
        desc = list()
        for n, seq in enumerate(self.sequences):
            wdw = min(window, len(seq))
            mtrx = list()
            for l in range(len(seq)):  # translate AA sequence into values
                mtrx.append(self.scale[str(seq[l])])
            mwdw = list()
            for i in range(len(mtrx) - wdw + 1):
                mwdw.append(sum(mtrx[i:i + wdw], []))  # list of all the values for the different windows
            mwdw = np.asarray(mwdw)
            glob = np.sum(mwdw, axis=1) / wdw
            if modality in ['max', 'mean']:
                if modality == 'max':
                    outglob = np.max(glob)  # returned moment will be the maximum of all windows
                elif modality == 'mean':
                    outglob = np.mean(glob)  # returned moment will be the mean of all windows
                desc.append(outglob)
                self.all_globs.append(glob)

            else:
                print 'Modality parameter is wrong, please choose between "max" and "mean"\n.'

        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def calculate_profile(self, prof_type='uH', window=7, append=False):
        """Method for calculating hydrophobicity or hydrophobic moment profiles for given sequences and fitting for
        slope and intercept. The hydrophobicity scale used is "eisenberg"

        :param prof_type: prof_type of profile, available: 'H' for hydrophobicity or 'uH' for hydrophobic moment
        :param window: {int} size of sliding window used (odd-numbered).
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: Fitted slope and intercept of calculated profile for every given sequence in self.descriptor
        :Example:

        >>> AMP = PeptideDescriptor('KLLKLLKKVVGALG','kytedoolittle')
        >>> AMP.calculate_profile(prof_type='H')
        >>> AMP.descriptor
        array([[ 0.03731293,  0.19246599]])
        """
        if prof_type == 'uH':
            self.calculate_moment(window=window)
            y_vals = self.all_moms
        elif prof_type == 'H':
            self.calculate_global(window=window)
            y_vals = self.all_globs
        else:
            print 'prof_type parameter is unknown, choose "uH" for hydrophobic moment or "H" for hydrophobicity\n.'
            sys.exit()

        desc = list()
        for n, seq in enumerate(self.sequences):
            x_vals = range(len(seq))[((window - 1) / 2):-((window - 1) / 2)]
            if len(seq) <= window:
                slope, intercept, r_value, p_value, std_err = [0, 0, 0, 0, 0]
            else:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals[n])
            desc.append([slope, intercept])

        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

    def count_aa(self, scale='relative', append=False):
        """Method for producing the amino acid distribution for the given sequences as a descriptor

        :param scale: {'absolute' or 'relative'} defines whether counts or frequencies are given for each AA
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
        """
        desc = list()
        scl = 1
        for seq in self.sequences:
            if scale == 'relative':
                scl = len(seq)
            d = {a: (float(seq.count(a)) / scl) for a in self.scale.keys()}
            od = collections.OrderedDict(sorted(d.items()))
            desc.append(od.values())

        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
        else:
            self.descriptor = np.array(desc)

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
            print "Unknown scaler type!\nAvailable: 'standard', 'minmax'"

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
        """Method for shuffling sequence order in self.sequences.

        :return: sequences in :py:attr:`self.sequences` with shuffled order in the list.
        :Example:

        >>> D.sequences
        ['LILRALKGAARALKVA','VKIAKIALKIIKGLG','VGVRLIKGIGRVARGAI','LRGLRGVIRGGKAIVRVGK','GGKLVRLIARIGKGV']
        >>> D.sequence_order_shuffle()
        >>> D.sequences
        ['VGVRLIKGIGRVARGAI','LILRALKGAARALKVA','LRGLRGVIRGGKAIVRVGK','GGKLVRLIARIGKGV','VKIAKIALKIIKGLG']
        """
        self.sequences = shuffle(self.sequences)

    def filter_unnatural(self):
        """Method to filter out sequences with unnatural amino acids from :py:attr:`sequences`.
        :return: Filtered sequence list in the attribute :py:attr:`sequences`

        .. seealso:: :func:`modlamp.core.filter_unnatural()`
        """
        filter_unnatural(self)

    def filter_values(self, values, operator='=='):
        """Method to filter the descriptor matrix in the attribute :py:attr:`descriptor` for a given list of value (same
        size as the number of features in the descriptor matrix!)

        :param values: List of values to filter
        :param operator: filter criterion, available are all SQL like operators: ``==``, ``<``, ``>``, ``<=``and ``>=``.
        :return: filtered descriptor matrix and updated sequences in the corresponding attributes.

        .. seealso:: :func:`modlamp.core.filter_values()`
        """
        filter_values(self, values, operator)

    def filter_duplicates(self):
        """
        Method to filter duplicates in the sequences from the class attribute :py:attr:`sequences`

        :return: filtered sequences list in the attribute :py:attr:`sequences`

        .. seealso:: :func:`modlamp.core.filter_sequences()`

        .. versionadded:: v2.2.5
        """
        filter_duplicates(self)

    def keep_natural_aa(self):
        """Method to filter out sequences that do not contain natural amino acids. If the sequence contains a character
        that is not in ['A','C','D,'E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'].

        :return: filtered sequence list in the attribute :py:attr:`sequences`. The other attributes are also filtered
            accordingly.

        .. seealso:: :func:`modlamp.core.keep_natural_aa()`

        .. versionadded:: v2.2.5
        """
        keep_natural_aa(self)

    def filter_aa(self, aminoacids):
        """Method to filter sequences and corresponding descriptor values, if the sequences contain any of the given
        amino acids in the argument list **aminoacids**.

        :param aminoacids: List/array of amino acids to filter for
        :return: filtered descriptor matrix and updated sequences and names in the corresponding attributes.

        .. seealso:: :func:`modlamp.core.filter_aa()`
        """
        filter_aa(self, aminoacids)

    def filter_sequences(self, sequences):
        """Method to filter out entries for given sequences in *sequences* out of a descriptor instance. All
        corresponding fields of these sequences (*descriptor*, *name*) are deleted as well. The method returns an
        updated descriptor instance.

        :param sequences: list of sequences to be filtered out of the whole instance, including corresponding data
        :return: updated instance

        .. seealso:: :func:`modlamp.core.filter_sequences()`

        .. versionadded:: v2.2.4
        """
        filter_sequences(self, sequences)

    def random_selection(self, num):
        """Method to randomly select a specified number of sequences (with names and descriptors if present) out of a
        given descriptor instance.

        :param num: {int} number of entries to be randomly selected
        :return: updated instance

        .. seealso:: :func:`modlamp.core.random_selection()`

        .. versionadded:: v2.2.3
        """
        random_selection(self, num)

    def minmax_selection(self, iterations, distmetric='euclidean', randseed=0):
        """Method to select a specified number of sequences out of a given descriptor instance according to the
        minmax algorithm.

        :param iterations: {int} Number of sequences to retrieve.
        :param distmetric: Distance metric to calculate the distances between the sequences in descriptor space.
            Choose from scipy.spacial.distance (http://docs.scipy.org/doc/scipy/reference/spatial.distance.html).
            E.g. 'euclidean', 'minkowsky'.
        :param randseed: {int} Set a random seed for numpy to pick the first sequence.
        :return: updated instance

        .. seealso:: :func:`modlamp.core.minmax_selection()`

        .. versionadded:: v2.2.6
        """
        minmax_selection(self, iterations, distmetric, randseed)

    def load_descriptordata(self, filename, delimiter=",", targets=False, header=0):
        """Method to load any data file with sequences and descriptor values and save it to a new insatnce of the
        class :class:`modlamp.descriptors.PeptideDescriptor`.

        .. note::
            The data file should **not** have any headers

        :param filename: filenam of the data file to be loaded
        :param delimiter: column delimiter
        :param targets: {boolean} whether last column in the file contains a target class vector
        :param header: {int} number of header lines to skip in the file
        :return: loaded sequences, descriptor values and targets in the corresponding attributes.
        """
        data = np.genfromtxt(filename, delimiter=delimiter, skip_header=header)
        data = data[:, 1:]  # skip sequences as they are "nan" when read as float
        seqs = np.genfromtxt(filename, delimiter=delimiter, dtype="str")
        seqs = seqs[:, 0]
        if targets:
            self.target = np.array(data[:, -1], dtype='int')
        self.sequences = seqs
        self.descriptor = data

    def save_descriptor(self, filename, delimiter=',', targets=None, header=''):
        """Method to save the descriptor values to a .csv/.txt file

        :param filename: filename of the output file
        :param delimiter: column delimiter
        :param targets: target class vector to be added to descriptor (same length as :py:attr:`sequences`)
        :param header: {str} header to be written at the beginning of the file
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
        np.savetxt(filename, data, delimiter=delimiter, fmt='%s', header=header)

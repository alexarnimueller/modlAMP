# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.descriptors

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates different classes to calculate peptide descriptor values. The following classes are available:

=============================        ============================================================================
Class                                Characteristics
=============================        ============================================================================
:py:class:`GlobalDescriptor`         Global one-dimensional peptide descriptors calculated from the AA sequence.
:py:class:`PeptideDescriptor`        AA scale based global or convoluted descriptors (auto-/cross-correlated).
=============================        ============================================================================

.. seealso:: :class:`modlamp.core.BaseDescriptor` from which the classes in :mod:`modlamp.descriptors` inherit.
"""

import sys

import numpy as np
from scipy import stats
from sklearn.externals.joblib import Parallel, delayed

from modlamp.core import BaseDescriptor, load_scale, count_aas, aa_weights, aa_energies, aa_formulas

__author__ = "Alex Müller, Gisela Gabernet"
__docformat__ = "restructuredtext en"


def _one_autocorr(seq, window, scale):
    """Private function used for calculating auto-correlated descriptors for 1 given sequence, window and an AA scale.
    This function is used by the :py:func:`calculate_autocorr` method of :py:class:`PeptideDescriptor`.

    :param seq: {str} amino acid sequence to calculate descriptor for
    :param window: {int} correlation-window size
    :param scale: {str} amino acid scale to be used to calculate descriptor
    :return: {numpy.array} calculated descriptor data
    """
    try:
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
    except ZeroDivisionError:
        print("ERROR!\nThe chosen correlation window % i is larger than the sequence %s !" % (window, seq))
        

def _one_crosscorr(seq, window, scale):
    """Private function used for calculating cross-correlated descriptors for 1 given sequence, window and an AA scale.
    This function is used by the :py:func:`calculate_crosscorr` method of :py:class:`PeptideDescriptor`.

    :param seq: {str} amino acid sequence to calculate descriptor for
    :param window: {int} correlation-window size
    :param scale: {str} amino acid scale to be used to calculate descriptor
    :return: {numpy.array} calculated descriptor data
    """
    try:
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
    except ZeroDivisionError:
        print("ERROR!\nThe chosen correlation window % i is larger than the sequence %s !" % (window, seq))

def _one_arc(seq, modality, scale):
    """ Privat function used for calculating arc descriptors for one sequence and AA scale. This function is used by
    :py:func:`calculate_arc` method method of :py:class:`PeptideDescriptor`.

    :param seq: {str} amino acid sequence to calculate descriptor for
    :param scale: {str} amino acid scale to be used to calculate descriptor
    :return: {numpy.array} calculated descriptor data
    """
    desc_mat = []
    for aa in seq:
        desc_mat.append(scale[aa])
    desc_mat = np.asarray(desc_mat)

    # Check descriptor dimension
    desc_dim = desc_mat.shape[1]

    # list to store descriptor values for all windows
    allwindows_arc = []

    if len(seq) > 18:
        window = 18
        # calculates number of windows in sequence
        num_windows = len(seq) - window
    else:
        window = len(seq)
        num_windows = 1

    # loop through all windows
    for j in range(num_windows):
        # slices descriptor matrix into current window
        window_mat = desc_mat[j:j + window, :]

        # defines order of amino acids in helical projection
        order = [0, 11, 4, 15, 8, 1, 12, 5, 16, 9, 2, 13, 6, 17, 10, 3, 14, 7]

        # orders window descriptor matrix into helical projection order
        ordered = []
        for pos in order:
            try:
                ordered.append(window_mat[pos, :])
            except:
                # for sequences of len < 18 adding dummy vector with 2s, length of descriptor dimensions
                ordered.append([2] * desc_dim)
        ordered = np.asarray(ordered)

        window_arc = []

        # loop through pharmacophoric features
        for m in range(desc_dim):
            all_arcs = []  # stores all arcs that can be found of a pharmacophoric feature
            arc = 0

            for n in range(18):  # for all positions in helix, regardless of sequence length
                if ordered[n, m] == 0:  # if position does not contain pharmacophoric feature
                    all_arcs.append(arc)  # append previous arc to all arcs list
                    arc = 0  # arc is initialized
                elif ordered[n, m] == 1:  # if position contains pharmacophoric feature(PF), elongate arc by 20°
                    arc += 20
                elif ordered[n, m] == 2:  # if position doesn't contain amino acid:
                    if ordered[n - 1, m] == 1:  # if previous position contained PF add 10°
                        arc += 10
                    elif ordered[n - 1, m] == 0:  # if previous position didn't contain PF don't add anything
                        arc += 0
                    elif ordered[
                                n - 2, m] == 1:  # if previous position is empty then check second previous for PF
                        arc += 10
                    if n == 17:  # if we are at the last position check for position n=0 instead of next position.
                        if ordered[0, m] == 1:  # if it contains PF add 10° extra
                            arc += 10
                    else:  # if next position contains PF add 10° extra
                        if ordered[n + 1, m] == 1:
                            arc += 10
                        elif ordered[n + 1, m] == 0:
                            arc += 0
                        else:  # if next position is empty check for 2nd next position
                            if n == 16:
                                if ordered[0, m] == 1:
                                    arc += 10
                            else:
                                if ordered[n + 2, m] == 1:
                                    arc += 10

            all_arcs.append(arc)
            if not arc == 360:
                arc0 = all_arcs.pop() + all_arcs[0]  # join first and last arc together
                all_arcs = [arc0] + all_arcs[1:]

            window_arc.append(np.max(all_arcs))  # append to window arcs the maximum arc of this PF
        allwindows_arc.append(window_arc)  # append all PF arcs of this window

    allwindows_arc = np.asarray(allwindows_arc)

    if modality == 'max':
        final_arc = np.max(allwindows_arc, axis=0)  # calculate maximum / mean arc along all windows
    elif modality == 'mean':
        final_arc = np.mean(allwindows_arc, axis=0)
    else:
        print 'modality is unknown, please choose between "max" and "mean"\n.'
        sys.exit()
    return final_arc

def _charge(seq, ph=7.0, amide=False):
    """Calculates charge of a single sequence. The method used is first described by Bjellqvist. In the case of
    amidation, the value for the  'Cterm' pKa is 15 (and Cterm is added to the pos_pks dictionary.
    The pKa scale is extracted from: http://www.hbcpnetbase.com/ (CRC Handbook of Chemistry and Physics, 96th ed).

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
    
    aa_content = count_aas(seq, scale='absolute')
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


class GlobalDescriptor(BaseDescriptor):
    """
    Base class for global, non-amino acid scale dependant descriptors. The following descriptors can be calculated by
    the **methods** linked below:

    - `Sequence Length      <modlamp.html#modlamp.descriptors.GlobalDescriptor.length>`_
    - `Molecular Formula    <modlamp.html#modlamp.descriptors.GlobalDescriptor.formula>`_
    - `Molecular Weight     <modlamp.html#modlamp.descriptors.GlobalDescriptor.calculate_MW>`_
    - `Sequence Charge      <modlamp.html#modlamp.descriptors.GlobalDescriptor.calculate_charge>`_
    - `Charge Density       <modlamp.html#modlamp.descriptors.GlobalDescriptor.charge_density>`_
    - `Isoelectric Point    <modlamp.html#modlamp.descriptors.GlobalDescriptor.isoelectric_point>`_
    - `Instability Index    <modlamp.html#modlamp.descriptors.GlobalDescriptor.instability_index>`_
    - `Aromaticity          <modlamp.html#modlamp.descriptors.GlobalDescriptor.aromaticity>`_
    - `Aliphatic Index      <modlamp.html#modlamp.descriptors.GlobalDescriptor.aliphatic_index>`_
    - `Boman Index          <modlamp.html#modlamp.descriptors.GlobalDescriptor.boman_index>`_
    - `Hydrophobic Ratio    <modlamp.html#modlamp.descriptors.GlobalDescriptor.hydrophobic_ratio>`_
    - `all of the above     <modlamp.html#modlamp.descriptors.GlobalDescriptor.calculate_all>`_
    """

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
        array([[ 8.], [12.], [12.]])
        """
        desc = []
        for seq in self.sequences:
            desc.append(float(len(seq.strip())))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
            self.featurenames.append('Length')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['Length']
    
    def formula(self, amide=False, append=False):
        """Method to calculate the molecular formula of every sequence in the attribute :py:attr:`sequences`.
        
        :param amide: {boolean} whether the sequences are C-terminally amidated.
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: array of molecular formulas {str} in the attribute :py:attr:`descriptor`
        :Example:
        
        >>> desc = GlobalDescriptor(['KADSFLSADGHSADFSLDKKLKERL', 'ERTILSDFPQWWFASLDFLNC', 'ACDEFGHIKLMNPQRSTVWY'])
        >>> desc.formula(amide=True)
        >>> for v in desc.descriptor:
        ...     print v[0]
        C122 H197 N35 O39
        C121 H168 N28 O33 S
        C106 H157 N29 O30 S2
        
        .. seealso:: :py:func:`modlamp.core.aa_formulas()`
        
        .. versionadded:: v2.7.6
        """
        desc = []
        formulas = aa_formulas()
        for seq in self.sequences:
            f = {'C': 0, 'H': 0, 'N': 0, 'O': 0, 'S': 0}
            for aa in seq:  # sum over aa weights
                for k in f.keys():
                    f[k] += formulas[aa][k]
            
            # substract H2O for every peptide bond
            f['H'] -= 2 * (len(seq) - 1)
            f['O'] -= (len(seq) - 1)
            
            if amide:  # add C-terminal amide --> replace OH with NH2
                f['O'] -= 1
                f['H'] += 1
                f['N'] += 1
            
            if f['S'] != 0:
                val = 'C%s H%s N%s O%s %s%s' % (f['C'], f['H'], f['N'], f['O'], 'S', f['S'])
            else:
                val = 'C%s H%s N%s O%s' % (f['C'], f['H'], f['N'], f['O'])
                
            desc.append([val])
        
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
            self.featurenames.append('Formula')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['Formula']

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
        if amide:  # if sequences are amidated, subtract 0.98 from calculated MW (OH - NH2)
            desc = [d - 0.98 for d in desc]
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
            self.featurenames.append('MW')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['MW']

    def calculate_charge(self, ph=7.0, amide=False, append=False):
        """Method to overall charge of every sequence in the attribute :py:attr:`sequences`.

        The method used is first described by Bjellqvist. In the case of amidation, the value for the 'Cterm' pKa is 15
        (and Cterm is added to the pos_pKs dictionary.
        The pKa scale is extracted from: http://www.hbcpnetbase.com/ (CRC Handbook of Chemistry and Physics, 96th ed).

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
            desc.append(_charge(seq, ph, amide))  # calculate charge with helper function
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
            self.featurenames.append('Charge')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['Charge']

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
            self.featurenames.append('ChargeDensity')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['ChargeDensity']

    def isoelectric_point(self, amide=False, append=False):
        """
        Method to calculate the isoelectric point of every sequence in the attribute :py:attr:`sequences`.
        The pK scale is extracted from: http://www.hbcpnetbase.com/ (CRC Handbook of Chemistry and Physics, 96th ed).

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
            self.featurenames.append('pI')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['pI']

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
        dimv = load_scale('instability')[1]
        for seq in self.sequences:
            stabindex = float()
            for i in range(len(seq) - 1):
                stabindex += dimv[seq[i]][seq[i+1]]
            desc.append((10.0 / len(seq)) * stabindex)
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
            self.featurenames.append('InstabilityInd')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['InstabilityInd']

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
            self.featurenames.append('Aromaticity')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['Aromaticity']

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
            self.featurenames.append('AliphaticInd')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['AliphaticInd']

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
            self.featurenames.append('BomanInd')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['BomanInd']

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
            self.featurenames.append('HydrophRatio')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['HydrophRatio']

    def calculate_all(self, ph=7.4, amide=True):
        """Method combining all global descriptors and appending them into the feature matrix in the attribute
        :py:attr:`descriptor`.
        
        :param ph: {float} pH at which to calculate peptide charge
        :param amide: {boolean} whether the sequences have an amidated C-terminus.
        :return: array of descriptor values in the attribute :py:attr:`descriptor`
        :Example:
        
        >>> desc = GlobalDescriptor('AFGHFKLKKLFIFGHERT')
        >>> desc.calculate_all(amide=True)
        >>> desc.featurenames
        ['Length', 'MW', 'ChargeDensity', 'pI', 'InstabilityInd', 'Aromaticity', 'AliphaticInd', 'BomanInd', 'HydRatio']
        >>> desc.descriptor
        array([[ 18.,  2.17559000e+03,   1.87167619e-03,   1.16757812e+01, ...  1.10555556e+00,   4.44444444e-01]])
        >>> desc.save_descriptor('/path/to/outputfile.csv')  # save the descriptor data (with feature names header)
        """
        
        # This is a strange way of doing it. However, the append=True option excludes length and charge, no idea why!
        fn = []
        self.length()  # sequence length
        l = self.descriptor
        fn.extend(self.featurenames)
        self.calculate_MW(amide=amide)  # molecular weight
        mw = self.descriptor
        fn.extend(self.featurenames)
        self.calculate_charge(ph=ph, amide=amide)  # net charge
        c = self.descriptor
        fn.extend(self.featurenames)
        self.charge_density(ph=ph, amide=amide)  # charge density
        cd = self.descriptor
        fn.extend(self.featurenames)
        self.isoelectric_point(amide=amide)  # pI
        pi = self.descriptor
        fn.extend(self.featurenames)
        self.instability_index()  # instability index
        si = self.descriptor
        fn.extend(self.featurenames)
        self.aromaticity()  # global aromaticity
        ar = self.descriptor
        fn.extend(self.featurenames)
        self.aliphatic_index()  # aliphatic index
        ai = self.descriptor
        fn.extend(self.featurenames)
        self.boman_index()  # Boman index
        bi = self.descriptor
        fn.extend(self.featurenames)
        self.hydrophobic_ratio()  # Hydrophobic ratio
        hr = self.descriptor
        fn.extend(self.featurenames)
        
        self.descriptor = np.concatenate((l, mw, c, cd, pi, si, ar, ai, bi, hr), axis=1)
        self.featurenames = fn


class PeptideDescriptor(BaseDescriptor):
    """Base class for peptide descriptors. The following **amino acid descriptor scales** are available for descriptor
    calculation:

    - **AASI**           (An amino acid selectivity index scale for helical antimicrobial peptides, *[1] D. Juretić, D. Vukicević, N. Ilić, N. Antcheva, A. Tossi, J. Chem. Inf. Model. 2009, 49, 2873–2882.*)
    - **ABHPRK**         (modlabs inhouse physicochemical feature scale (Acidic, Basic, Hydrophobic, Polar, aRomatic, Kink-inducer)
    - **argos**          (Argos hydrophobicity amino acid scale, *[2] Argos, P., Rao, J. K. M. & Hargrave, P. A., Eur. J. Biochem. 2005, 128, 565–575.*)
    - **bulkiness**      (Amino acid side chain bulkiness scale, *[3] J. M. Zimmerman, N. Eliezer, R. Simha, J. Theor. Biol. 1968, 21, 170–201.*)
    - **charge_phys**    (Amino acid charge at pH 7.0 - Hystidine charge +0.1.)
    - **charge_acid**    (Amino acid charge at acidic pH - Hystidine charge +1.0.)
    - **cougar**         (modlabs inhouse selection of global peptide descriptors)
    - **eisenberg**      (the Eisenberg hydrophobicity consensus amino acid scale, *[4] D. Eisenberg, R. M. Weiss, T. C. Terwilliger, W. Wilcox, Faraday Symp. Chem. Soc. 1982, 17, 109.*)
    - **Ez**             (potential that assesses energies of insertion of amino acid side chains into lipid bilayers, *[5] A. Senes, D. C. Chadi, P. B. Law, R. F. S. Walters, V. Nanda, W. F. DeGrado, J. Mol. Biol. 2007, 366, 436–448.*)
    - **flexibility**    (amino acid side chain flexibilitiy scale, *[6] R. Bhaskaran, P. K. Ponnuswamy, Int. J. Pept. Protein Res. 1988, 32, 241–255.*)
    - **grantham**       (amino acid side chain composition, polarity and molecular volume, *[8] Grantham, R. Science. 185, 862–864 (1974).*)
    - **gravy**          (GRAVY hydrophobicity amino acid scale, *[9] J. Kyte, R. F. Doolittle, J. Mol. Biol. 1982, 157, 105–132.*)
    - **hopp-woods**     (Hopp-Woods amino acid hydrophobicity scale,*[10] T. P. Hopp, K. R. Woods, Proc. Natl. Acad. Sci. 1981, 78, 3824–3828.*)
    - **ISAECI**         (Isotropic Surface Area (ISA) and Electronic Charge Index (ECI) of amino acid side chains, *[11] E. R. Collantes, W. J. Dunn, J. Med. Chem. 1995, 38, 2705–2713.*)
    - **janin**          (Janin hydrophobicity amino acid scale, *[12] J. L. Cornette, K. B. Cease, H. Margalit, J. L. Spouge, J. A. Berzofsky, C. DeLisi, J. Mol. Biol. 1987, 195, 659–685.*)
    - **kytedoolittle**  (Kyte & Doolittle hydrophobicity amino acid scale, *[13] J. Kyte, R. F. Doolittle, J. Mol. Biol. 1982, 157, 105–132.*)
    - **levitt_alpha**   (Levitt amino acid alpha-helix propensity scale, extracted from http://web.expasy.org/protscale. *[14] M. Levitt, Biochemistry 1978, 17, 4277-4285.*)
    - **MSS**            (A graph-theoretical index that reflects topological shape and size of amino acid side chains, *[15] C. Raychaudhury, A. Banerjee, P. Bag, S. Roy, J. Chem. Inf. Comput. Sci. 1999, 39, 248–254.*)
    - **MSW**            (Amino acid scale based on a PCA of the molecular surface based WHIM descriptor (MS-WHIM), extended to natural amino acids, *[16] A. Zaliani, E. Gancia, J. Chem. Inf. Comput. Sci 1999, 39, 525–533.*)
    - **pepArc**         (modlabs pharmacophoric feature scale, dimensions are: hydrophobicity, polarity, positive charge, negative charge, proline.)
    - **pepcats**        (modlabs pharmacophoric feature based PEPCATS scale, *[17] C. P. Koch, A. M. Perna, M. Pillong, N. K. Todoroff, P. Wrede, G. Folkers, J. A. Hiss, G. Schneider, PLoS Comput. Biol. 2013, 9, e1003088.*)
    - **polarity**       (Amino acid polarity scale, *[18] J. M. Zimmerman, N. Eliezer, R. Simha, J. Theor. Biol. 1968, 21, 170–201.*)
    - **PPCALI**         (modlabs inhouse scale derived from a PCA of 143 amino acid property scales, *[19] C. P. Koch, A. M. Perna, M. Pillong, N. K. Todoroff, P. Wrede, G. Folkers, J. A. Hiss, G. Schneider, PLoS Comput. Biol. 2013, 9, e1003088.*)
    - **refractivity**   (Relative amino acid refractivity values, *[20] T. L. McMeekin, M. Wilensky, M. L. Groves, Biochem. Biophys. Res. Commun. 1962, 7, 151–156.*)
    - **t_scale**        (A PCA derived scale based on amino acid side chain properties calculated with 6 different probes of the GRID program, *[21] M. Cocchi, E. Johansson, Quant. Struct. Act. Relationships 1993, 12, 1–8.*)
    - **TM_tend**        (Amino acid transmembrane propensity scale, extracted from http://web.expasy.org/protscale, *[22] Zhao, G., London E. Protein Sci. 2006, 15, 1987-2001.*)
    - **z3**             (The original three dimensional Z-scale, *[23] S. Hellberg, M. Sjöström, B. Skagerberg, S. Wold, J. Med. Chem. 1987, 30, 1126–1135.*)
    - **z5**             (The extended five dimensional Z-scale, *[24] M. Sandberg, L. Eriksson, J. Jonsson, M. Sjöström, S. Wold, J. Med. Chem. 1998, 41, 2481–2491.*)

    Further, amino acid scale independent methods can be calculated with help of the :class:`GlobalDescriptor` class.

    """

    def __init__(self, seqs, scalename='Eisenberg'):
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
        super(PeptideDescriptor, self).__init__(seqs)
        self.scalename, self.scale = load_scale(scalename.lower())
        self.all_moms = list()  # for passing hydrophobic moments to calculate_profile
        self.all_globs = list()  # for passing global  to calculate_profile

    def load_scale(self, scalename):
        """Method to load amino acid values from a given scale

        :param scalename: {str} name of the amino acid scale to be loaded.
        :return: loaded amino acid scale values in a dictionary in the attribute :py:attr:`scale`.

        .. seealso:: :func:`modlamp.core.load_scale()`
        """
        self.scalename, self.scale = load_scale(scalename.lower())

    def calculate_autocorr(self, window, append=False):
        """Method for auto-correlating the amino acid values for a given descriptor scale

        :param window: {int} correlation window for descriptor calculation in a sliding window approach
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: calculated descriptor numpy.array in the attribute :py:attr:`descriptor`.
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
        :return: calculated descriptor numpy.array in the attribute :py:attr:`descriptor`.
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
        :param modality: {'all', 'max' or 'mean'} Calculate respectively maximum or mean hydrophobic moment. If all,
            moments for all windows are returned.
        :param append: {boolean} whether the produced descriptor values should be appended to the existing ones in the
            attribute :py:attr:`descriptor`.
        :return: Calculated descriptor as a numpy.array in the attribute :py:attr:`descriptor` and all possible global
            values in :py:attr:`all_moms` (needed for the :py:func:`calculate_profile` method)
        :Example:

        >>> AMP = PeptideDescriptor('GLFDIVKKVVGALGSL', 'eisenberg')
        >>> AMP.calculate_moment()
        >>> AMP.descriptor
        array([[ 0.48790226]])
        """
        if self.scale['A'] == list:
            print('\n Descriptor moment calculation is only possible for one dimensional descriptors.\n')
    
        else:
            desc = []
            for seq in self.sequences:
                wdw = min(window, len(seq))  # if sequence is shorter than window, take the whole sequence instead
                mtrx = []
                mwdw = []
                
                for aa in range(len(seq)):
                    mtrx.append(self.scale[str(seq[aa])])
                    
                for i in range(len(mtrx) - wdw + 1):
                    mwdw.append(sum(mtrx[i:i + wdw], []))
    
                mwdw = np.asarray(mwdw)
                rads = angle * (np.pi / 180) * np.asarray(range(wdw))  # calculate actual moment (radial)
                vcos = (mwdw * np.cos(rads)).sum(axis=1)
                vsin = (mwdw * np.sin(rads)).sum(axis=1)
                moms = np.sqrt(vsin ** 2 + vcos ** 2) / wdw
    
                if modality == 'max':  # take window with maximal value
                    moment = np.max(moms)
                elif modality == 'mean':  # take average value over all windows
                    moment = np.mean(moms)
                elif modality == 'all':
                    moment = moms
                else:
                    print('\nERROR!\nModality parameter is wrong, please choose between "all", "max" and "mean".\n')
                    return
                desc.append(moment)
                self.all_moms.append(moms)
                
            desc = np.asarray(desc).reshape(len(desc), 1)  # final descriptor array
            
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
        :return: Calculated descriptor as a numpy.array in the attribute :py:attr:`descriptor` and all possible global
            values in :py:attr:`all_globs` (needed for the :py:func:`calculate_profile` method)
        :Example:

        >>> AMP = PeptideDescriptor('GLFDIVKKVVGALGSL','eisenberg')
        >>> AMP.calculate_global(window=1000, modality='max')
        >>> AMP.descriptor
        array([[ 0.44875]])
        """
        desc = list()
        for n, seq in enumerate(self.sequences):
            wdw = min(window, len(seq))  # if sequence is shorter than window, take the whole sequence instead
            mtrx = []
            mwdw = []
            
            for l in range(len(seq)):  # translate AA sequence into values
                mtrx.append(self.scale[str(seq[l])])

            for i in range(len(mtrx) - wdw + 1):
                mwdw.append(sum(mtrx[i:i + wdw], []))  # list of all the values for the different windows
                
            mwdw = np.asarray(mwdw)
            glob = np.sum(mwdw, axis=1) / wdw
            outglob = float()
            
            if modality in ['max', 'mean']:
                if modality == 'max':
                    outglob = np.max(glob)  # returned moment will be the maximum of all windows
                elif modality == 'mean':
                    outglob = np.mean(glob)  # returned moment will be the mean of all windows
                else:
                    print('Modality parameter is wrong, please choose between "max" and "mean"\n.')
                    return
            desc.append(outglob)
            self.all_globs.append(glob)
        
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
        :return: Fitted slope and intercept of calculated profile for every given sequence in the attribute
            :py:attr:`descriptor`.
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
            print('prof_type parameter is unknown, choose "uH" for hydrophobic moment or "H" for hydrophobicity\n.')
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

    def calculate_arc(self, modality="max", append=False):
        """ Method for calculating property arcs as seen in the helical wheel plot. Use for binary amino acid scales only.
        
        :param modality: modality of the arc to calculate, to choose between "max" and "mean".
        :param append: if true, append to current descriptor stored in the descriptor attribute.
        :return: calculated descriptor as numpy.array in the descriptor attribute.

        :Example:

        >>> arc = PeptideDescriptor("KLLKLLKKLLKLLK", scalename="peparc")
        >>> arc.calculate_arc(modality="max", append=False)
        >>> arc.descriptor
        array([[200, 160, 160,   0,   0]])
        """
        desc = Parallel(n_jobs=-1)(delayed(_one_arc)(seq, modality, self.scale) for seq in self.sequences)

        # Converts each of the amino acids to descriptor vector
        for seq in self.sequences:

            # desc_mat = []
            # for aa in seq:
            #     desc_mat.append(self.scale[aa])
            # desc_mat = np.asarray(desc_mat)
			#
            # # Check descriptor dimension
            # desc_dim = desc_mat.shape[1]
			#
            # # list to store descriptor values for all windows
            # allwindows_arc = []
			#
            # if len(seq) > 18:
            #     window = 18
            #     # calculates number of windows in sequence
            #     num_windows = len(seq) - window
            # else:
            #     window = len(seq)
            #     num_windows = 1
			#
            # # loop through all windows
            # for j in range(num_windows):
            #     # slices descriptor matrix into current window
            #     window_mat = desc_mat[j:j + window, :]
			#
            #     # defines order of amino acids in helical projection
            #     order = [0, 11, 4, 15, 8, 1, 12, 5, 16, 9, 2, 13, 6, 17, 10, 3, 14, 7]
			#
            #     # orders window descriptor matrix into helical projection order
            #     ordered = []
            #     for pos in order:
            #         try:
            #             ordered.append(window_mat[pos, :])
            #         except:
            #             # for sequences of len < 18 adding dummy vector with 2s, length of descriptor dimensions
            #             ordered.append([2] * desc_dim)
            #     ordered = np.asarray(ordered)
			#
            #     window_arc = []
			#
            #     # loop through pharmacophoric features
            #     for m in range(desc_dim):
            #         all_arcs = []  # stores all arcs that can be found of a pharmacophoric feature
            #         arc = 0
			#
            #         for n in range(18):  # for all positions in helix, regardless of sequence length
            #             if ordered[n, m] == 0:  # if position does not contain pharmacophoric feature
            #                 all_arcs.append(arc)  # append previous arc to all arcs list
            #                 arc = 0  # arc is initialized
            #             elif ordered[n, m] == 1:  # if position contains pharmacophoric feature(PF), elongate arc by 20°
            #                 arc += 20
            #             elif ordered[n, m] == 2:  # if position doesn't contain amino acid:
            #                 if ordered[n - 1, m] == 1:  # if previous position contained PF add 10°
            #                     arc += 10
            #                 elif ordered[n - 1, m] == 0:  # if previous position didn't contain PF don't add anything
            #                     arc += 0
            #                 elif ordered[
            #                             n - 2, m] == 1:  # if previous position is empty then check second previous for PF
            #                     arc += 10
            #                 if n == 17:  # if we are at the last position check for position n=0 instead of next position.
            #                     if ordered[0, m] == 1:  # if it contains PF add 10° extra
            #                         arc += 10
            #                 else:  # if next position contains PF add 10° extra
            #                     if ordered[n + 1, m] == 1:
            #                         arc += 10
            #                     elif ordered[n + 1, m] == 0:
            #                         arc += 0
            #                     else:  # if next position is empty check for 2nd next position
            #                         if n == 16:
            #                             if ordered[0, m] == 1:
            #                                 arc += 10
            #                         else:
            #                             if ordered[n + 2, m] == 1:
            #                                 arc += 10
			#
            #         all_arcs.append(arc)
            #         if not arc == 360:
            #             arc0 = all_arcs.pop() + all_arcs[0]  # join first and last arc together
            #             all_arcs = [arc0] + all_arcs[1:]
			#
            #         window_arc.append(np.max(all_arcs))  # append to window arcs the maximum arc of this PF
            #     allwindows_arc.append(window_arc)  # append all PF arcs of this window
			#
            # allwindows_arc = np.asarray(allwindows_arc)
			#
            # if modality == 'max':
            #     final_arc = np.max(allwindows_arc, axis=0)  # calculate maximum / mean arc along all windows
            # elif modality == 'mean':
            #     final_arc = np.mean(allwindows_arc, axis=0)
            # else:
            #     print 'modality is unknown, please choose between "max" and "mean"\n.'
            #     sys.exit()

            if append:
                self.descriptor = np.hstack((self.descriptor, np.array(desc)))
            else:
                self.descriptor = np.array(desc)









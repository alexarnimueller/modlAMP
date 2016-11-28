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

import json
import sys
from os.path import dirname, join

import numpy as np
from scipy import stats
from sklearn.externals.joblib import Parallel, delayed

from core import BaseDescriptor, load_scale, count_aa, aa_weights, aa_energies

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


class GlobalDescriptor(BaseDescriptor):
    """
    Base class for global, non-amino acid scale dependant descriptors. The following descriptors can be calculated by
    the **methods** linked below:

    - `Sequence Length      <modlamp.html#modlamp.descriptors.GlobalDescriptor.length>`_
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
        array([[ 8], [12], [12]])
        """
        desc = []
        for seq in self.sequences:
            desc.append(len(seq.strip()))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptor = np.hstack((self.descriptor, np.array(desc)))
            self.featurenames.append('Length')
        else:
            self.descriptor = np.array(desc)
            self.featurenames = ['Length']

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
            desc.append(_charge(seq, ph, amide))
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
        
        >>> d = GlobalDescriptor('AFGHFKLKKLFIFGHERT')
        >>> d.calculate_all(amide=true)
        >>> d.featurenames
        ['MW', 'ChargeDensity', 'pI', 'InstabilityInd', 'Aromaticity', 'AliphaticInd', 'BomanInd', 'HydrophRatio']
        >>> d.descriptor
        array([[  2.17559000e+03,   1.87167619e-03,   1.16757812e+01, ...  1.10555556e+00,   4.44444444e-01]])
        """
        self.length()  # sequence length
        self.calculate_MW(amide=amide, append=True)  # molecular weight
        self.calculate_charge(ph=ph, amide=amide, append=True)  # net charge
        self.charge_density(ph=ph, amide=amide, append=True)  # charge density
        self.isoelectric_point(amide=amide, append=True)  # pI
        self.instability_index(append=True)  # instability index
        self.aromaticity(append=True)  # global aromaticity
        self.aliphatic_index(append=True)  # aliphatic index
        self.boman_index(append=True)  # Boman index
        self.hydrophobic_ratio(append=True)  # Hydrophobic ratio


class PeptideDescriptor(BaseDescriptor):
    """Base class for peptide descriptors. The following **amino acid descriptor scales** are available for descriptor
    calculation:

    - **AASI**           (An amino acid selectivity index scale for helical antimicrobial peptides, *[1] D. Juretić, D. Vukicević, N. Ilić, N. Antcheva, A. Tossi, J. Chem. Inf. Model. 2009, 49, 2873–2882.*)
    - **ABHPRK**          (modlabs inhouse physicochemical feature scale (Acidic, Basic, Hydrophobic, Polar, aRomatic, Kink-inducer)
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

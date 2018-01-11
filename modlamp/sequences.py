# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.sequences

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates different classes to generate peptide sequences with different characteristics from scratch.
The following classes are available:

============================        ===============================================================================
Class                               Characteristics
============================        ===============================================================================
:py:class:`Random`                  Generates random sequences with a specified amino acid distribution.
:py:class:`Helices`                 Generates presumed amphipathic helical sequences with a hydrophobic moment.
:py:class:`Kinked`                  Generates presumed amphipathic helices with a kink (Pro residue).
:py:class:`Oblique`                 Generates presumed oblique oriented sequences in presence of libid membranes.
:py:class:`Centrosymmetric`         Generates centrosymmetric sequences with a symmetry axis.
:py:class:`AmphipathicArc`          Generates presumed amphipathic helices with controlled hydrophobic arc size.
:py:class:`HelicesACP`              Generates sequences with the amino acid probabiliy of helical ACPs.
:py:class:`MixedLibrary`            Generates a mixed library of sequences of most other classes.
:py:class:`Hepahelices`             Generates presumed amphipathic helices with a heparin-binding-domain.
:py:class:`AMPngrams`               Generates sequences from most frequent ngrams in the APD3.
============================        ===============================================================================

All classes share the same method :py:func:`generate_sequences()` for sequence generation, potentially enabling
automation or looping over different classes.

.. note:: During the process of sequence generation, duplicates are only removed for the :py:class:`MixedLibrary`
    class. To remove duplicates, call the class methods :py:func:`self.filter_duplicates()`.
    
.. seealso:: :class:`modlamp.core.BaseSequence` from which all classes in this module inherit.
"""

from numpy import random
from itertools import cycle

import numpy as np

from modlamp.core import BaseSequence, ngrams_apd

__author__ = "Alex Müller, Gisela Gabernet"
__docformat__ = "restructuredtext en"


class Random(BaseSequence):
    """Class for random peptide sequences.

    This class incorporates methods for generating peptide random peptide sequences of defined length.
    The amino acid probabilities can be chosen from different probabilities:

    - **rand**: equal probabilities for all amino acids
    - **AMP**: amino acid probabilities taken from the antimicrobial peptide database `APD3 <http://aps.unmc.edu/AP/statistic/statistic.php>`_, March 17, 2016, containing 2674 sequences.
    - **AMPnoCM**: same amino acid probabilities as **AMP** but lacking Cys and Met (for synthesizability)
    - **ACP**: amino acid probabilities taken from 339 linear peptides in CancerPPD database `CancerPPD  <http://http://crdd.osdd.net/raghava/cancerppd/>`
    - **randnoCM**: equal probabilities for all amino acids, except 0.0 for both Cys and Met (for synthesizability)

    The probability values for all natural AA can be found in the following table:

    ===  ====    ======    =========    ==========  ==========
    AA   rand    AMP       AMPnoCM      randnoCM    ACP
    ===  ====    ======    =========    ==========  ==========
    A    0.05    0.0766    0.0812275    0.05555555  0.14526966
    C    0.05    0.071     0.0          0.0         0.
    D    0.05    0.026     0.0306275    0.05555555  0.00690031
    E    0.05    0.0264    0.0310275    0.05555555  0.00780824
    F    0.05    0.0405    0.0451275    0.05555555  0.06991102
    G    0.05    0.1172    0.1218275    0.05555555  0.04957327
    H    0.05    0.021     0.0256275    0.05555555  0.01725077
    I    0.05    0.061     0.0656275    0.05555555  0.05647358
    K    0.05    0.0958    0.1004275    0.05555555  0.27637552
    L    0.05    0.0838    0.0884275    0.05555555  0.17759216
    M    0.05    0.0123    0.0          0.0         0.00998729
    N    0.05    0.0386    0.0432275    0.05555555  0.00798983
    P    0.05    0.0463    0.0509275    0.05555555  0.01307427
    Q    0.05    0.0251    0.0297275    0.05555555  0.00381333
    R    0.05    0.0545    0.0591275    0.05555555  0.02941711
    S    0.05    0.0613    0.0659275    0.05555555  0.02651171
    T    0.05    0.0455    0.0501275    0.05555555  0.01543490
    V    0.05    0.0572    0.0618275    0.05555555  0.04013074
    W    0.05    0.0155    0.0201275    0.05555555  0.04067550
    Y    0.05    0.0244    0.0290275    0.05555555  0.00581079
    ===  ====    ======    =========    ==========  ==========

    """

    def generate_sequences(self, proba='rand'):
        """Method to actually generate the sequences.

        :param proba: {str or list} AA probability to be used to generate sequences. Available from str: AMP, AMPnoCM, 
            rand, randnoCM. You can also provide your own list of porbabilities as a list (in AA order, length 20, sum to 1)
        :return: A list of random AMP sequences with defined AA probabilities
        :Example:

        >>> b = Random(6, 5, 20)
        >>> b.generate_sequences(proba='AMP')
        >>> b.sequences
        ['CYGALWHIFV','NIVRHHAPSTVIK','LCPNPILGIV','TAVVRGKESLTP','GTGSVCKNSCRGRFGIIAF','VIIGPSYGDAEYA']
        """
        self.clean()
        if proba == 'AMPnoCM':
            self.prob = self.prob_AMPnoCM
        elif proba == 'AMP':
            self.prob = self.prob_AMP
        elif proba == 'randnoCM':
            self.prob = self.prob_randnoCM
        elif proba == 'ACP':
            self.prob = self.prob_ACP
        elif isinstance(proba, list) and len(proba) == 20:
            self.prob = proba
        # else just keep self.prob which is defined as equal probabilities for all AA

        for s in range(self.seqnum):
            seq = []
            for l in range(random.choice(range(self.lenmin, self.lenmax + 1))):
                seq.append(np.random.choice(self.AAs, replace=True, p=self.prob))  # weighed random selection of AA
            self.sequences.append(''.join(seq))


class Helices(BaseSequence):
    """Class for peptide sequences probable to form helices.
    This class incorporates methods for generating presumed amphipathic alpha-helical peptide sequences.
    These sequences are generated by placing basic residues along the sequence with distance 3-4 AA to each other.
    The remaining empty spots are filled up by hydrophobic AAs.
    """
    
    def generate_sequences(self):
        """Method to generate amphipathic helical sequences with class features defined in :class:`Helices()`

        :return: In the attribute :py:attr:`sequences`: a list of sequences with presumed amphipathic helical structure.
        :Example:

        >>> h = Helices(5, 7, 21)
        >>> h.generate_sequences()
        >>> h.sequences
        ['KGIKVILKLAKAGVKAVRL','IILKVGKV','IAKAGRAIIK','LKILKVVGKGIRLIVRIIKAL','KAGKLVAKGAKVAAKAIKI']
        """
        self.clean()
        for s in range(self.seqnum):  # for the number of sequences to generate
            seq = ['X'] * random.choice(range(self.lenmin, self.lenmax + 1))
            basepos = random.choice(range(4))  # select spot for first basic residue from 0 to 3
            seq[basepos] = random.choice(self.AA_basic)  # place first basic residue
            gap = cycle([3, 4]).next  # gap cycle of 3 & 4 --> 3,4,3,4,3,4...
            g = gap()
            while g + basepos < len(
                    seq):  # place more basic residues 3-4 positions further (changing between distance 3 and 4)
                basepos += g
                seq[basepos] = random.choice(self.AA_basic)  # place more basic residues
                g = gap()  # next gap
            
            for p in range(len(seq)):
                while seq[p] == 'X':  # fill up remaining spots with hydrophobic AAs
                    seq[p] = random.choice(self.AA_hyd)
            
            self.sequences.append(''.join(seq))


class Kinked(BaseSequence):
    """Class for peptide sequences probable to form helices with a kink.

    This class incorporates methods for presumed kinked amphipathic alpha-helical peptide sequences:
    Sequences are generated by placing basic residues along the sequence with distance 3-4 AA to each other.
    The remaining spots are filled up by hydrophobic AAs. Then, a basic residue is replaced by proline, presumably
    leading to a kink in the hydrophobic face of the amphipathic helices.
    """
    
    def generate_sequences(self):
        """Method to actually generate the presumed kinked sequences with features defined in the class instances.

        :return: sequence list with strings stored in the attribute :py:attr:`sequences`
        :Example:

        >>> k = Kinked(8, 7, 28)
        >>> k.generate_sequences()
        >>> k.sequences
        ['IILRLHPIG','ARGAKVAIKAIRGIAPGGRVVAKVVKVG','GGKVGRGVAFLVRIILK','KAVKALAKGAPVILCVAKVI', ...]
        """
        self.clean()
        for s in range(self.seqnum):  # for the number of sequences to generate
            poslist = []  # used to
            seq = ['X'] * random.choice(range(self.lenmin, self.lenmax + 1))
            basepos = random.choice(range(4))  # select spot for first basic residue from 0 to 3
            seq[basepos] = random.choice(self.AA_basic)  # place first basic residue
            poslist.append(basepos)
            gap = cycle([3, 4]).next  # gap cycle of 3 & 4 --> 3,4,3,4,3,4...
            g = gap()
            while g + basepos < len(
                    seq):  # place more basic residues 3-4 positions further (changing between distance 3 and 4)
                basepos += g
                seq[basepos] = random.choice(self.AA_basic)  # place more basic residues
                g = gap()  # next gap
                poslist.append(basepos)
            
            for p in range(len(seq)):
                while seq[p] == 'X':  # fill up remaining spots with hydrophobic AAs
                    seq[p] = random.choice(self.AA_hyd)
            
            # place proline around the middle of the sequence
            propos = poslist[len(poslist) / 2]
            seq[propos] = 'P'
            
            self.sequences.append(''.join(seq))


class Oblique(BaseSequence):
    """Class for oblique sequences with a so called linear hydrophobicity gradient.

    This class incorporates methods for generating peptide sequences with a linear hydrophobicity gradient, meaning that
    these sequences have a hydrophobic tail. This feature gives rise to the hypothesis that they orient themselves
    tilted/oblique in membrane environment.
    """
    
    def generate_sequences(self):
        """Method to generate the possible oblique sequences.

        :return: A list of sequences in the attribute :py:attr:`sequences`.
        :Example:

        >>> o = Oblique(4, 10, 30)
        >>> o.generate_sequences()
        >>> o.sequences
        ['GLLKVIRIAAKVLKVAVLVGIIAI','AIGKAGRLALKVIKVVIKVALILLAAVA','KILRAAARVIKGGIKAIVIL','VRLVKAIGKLLRIILRLARLAVGGILA']
        """
        self.clean()
        for s in range(self.seqnum):  # for the number of sequences to generate
            seq = ['X'] * random.choice(range(self.lenmin, self.lenmax + 1))
            basepos = random.choice(range(4))  # select spot for first basic residue from 0 to 3
            seq[basepos] = random.choice(self.AA_basic)  # place first basic residue
            gap = cycle([3, 4]).next  # gap cycle of 3 & 4 --> 3,4,3,4,3,4...
            g = gap()
            while g + basepos < len(
                    seq):  # place more basic residues 3-4 positions further (changing between distance 3 and 4)
                basepos += g
                seq[basepos] = random.choice(self.AA_basic)  # place more basic residues
                g = gap()  # next gap
            
            for p in range(len(seq)):
                while seq[p] == 'X':  # fill up remaining spots with hydrophobic AAs
                    seq[p] = random.choice(self.AA_hyd)
            
            for e in range(1, len(
                    seq) / 3):  # transform last 3rd of sequence into hydrophobic ones --> hydrophobicity gradient = oblique
                seq[-e] = random.choice(self.AA_hyd)
            
            self.sequences.append(''.join(seq))


class Centrosymmetric(BaseSequence):
    """Class for peptide sequences produced out of 7 AA centro-symmetric blocks yielding peptides of length
    14 or 21 AA (2*7 or 3*7).

    This class incorporates methods to generate special peptide sequences with an overall presumed
    hydrophobic moment. Sequences are generated by centro-symmetric blocks of seven amino acids. Two or three blocks
    are added to build a final sequence of length 14 or 21 amino acids length. If the option ``symmetric`` is used,
    two or three identical blocks are concatenated. If the the option ``asymmetric`` is used, two or three different
    blocks are concatenated.
    """
    
    def generate_sequences(self, symmetry='asymmetric'):
        """Generate overall symmetric or asymmetric sequences out of two or three blocks of centro-symmetric blocks of 7
        amino acids. The resulting sequence presumably has a large hydrophobic moment.

        :param symmetry: {str} type of centrosymmetric sequences. ``symmetric``: builds sequences out of only one
            block, ``asymmetric``: builds sequences out of different blocks
        :return: In the attribute :py:attr:`sequences`: centro-symmetric peptide sequences of the form [h,+,h,a,h,+,h]
            with h = hydrophobic AA, + = basic AA, a = anchor AA (F,Y,W,(P)), sequence length is 14 or 21 AA
        :Example:

        >>> s = Centrosymmetric(5)
        >>> s.generate_sequences(symmetry='symmetric')
        >>> s.sequences
        ['ARIFIRAARIFIRA','GRIYIRGGRIYIRGGRIYIRG','IRGFGRIIRGFGRIIRGFGRI','GKAYAKGGKAYAKG','AKGYGKAAKGYGKAAKGYGKA']
        """
        if symmetry == 'symmetric':
            self.clean()
            for s in range(self.seqnum):  # iterate over number of sequences to generate
                n = random.choice(range(2, 4))  # number of sequence blocks to take (2 or 3)
                seq = ['X'] * 7  # template sequence AA list with length 7
                for a in range(7):  # generate symmetric sequence block of 7 AA with an anchor in the middle
                    if a == 0:
                        seq[0] = random.choice(self.AA_hyd)
                        seq[6] = seq[0]
                    elif a == 1:
                        seq[1] = random.choice(self.AA_basic)
                        seq[5] = seq[1]
                    elif a == 2:
                        seq[2] = random.choice(self.AA_hyd)
                        seq[4] = seq[2]
                    elif a == 3:
                        seq[3] = random.choice(self.AA_aroma)
                    else:
                        continue
                self.sequences.append(''.join(seq) * n)
        
        elif symmetry == 'asymmetric':
            self.clean()
            for s in range(self.seqnum):  # iterate over number of sequences to generate
                n = random.choice(range(2, 4))  # number of sequence blocks to take (2 or 3)
                seq = ['X'] * 7  # template sequence AA list with length 7
                blocks = []
                for c in range(n):
                    for a in range(7):  # generate symmetric sequence block of 7 AA with an anchor in the middle
                        if a == 0:
                            seq[0] = random.choice(self.AA_hyd)
                            seq[6] = seq[0]
                        elif a == 1:
                            seq[1] = random.choice(self.AA_basic)
                            seq[5] = seq[1]
                        elif a == 2:
                            seq[2] = random.choice(self.AA_hyd)
                            seq[4] = seq[2]
                        elif a == 3:
                            seq[3] = random.choice(self.AA_aroma)
                        else:
                            continue
                    blocks.append(''.join(seq))
                self.sequences.append(''.join(blocks))
        
        else:
            raise AttributeError('Unknown symmetry option given! Choose from [symmetric, asymmetric].')
    

class AmphipathicArc(BaseSequence):
    """Class for generating positively-charged amphipathic peptide sequences based on an alpha-helix pattern with
    different arc sizes.

    The probability values for the Hydrophobic and Polar positions of the helix can be found in the following table:

    ===   ====   =====
    AA    Hydr   Polar
    ===   ====   =====
    A     0.00   0.05
    C     0.00   0.00
    D     0.00   0.05
    E     0.00   0.05
    F     0.16   0.00
    G     0.00   0.05
    H     0.00   0.05
    I     0.16   0.00
    K     0.00   0.25
    L     0.16   0.00
    M     0.00   0.00
    N     0.00   0.05
    P     0.00   0.05
    Q     0.00   0.05
    R     0.00   0.25
    S     0.00   0.05
    T     0.00   0.05
    V     0.16   0.00
    W     0.16   0.00
    Y     0.16   0.05
    ===   ====   =====

    """
    
    def generate_sequences(self, arcsize=180):
        """Method to generate the possible amphipathic helices with defined hydrophobic arc sizes (option
        ``arcsize``), with mixed arc sizes (option ``arcsize=None``)

        :param arcsize: {int} to choose among 100, 140, 180, 220, 260, or choose `mixed` to generate a mixture.
        :return: A list of sequences in the attribute :py:attr:`sequences`.
        :Example:

        >>> from modlamp.sequences import AmphipathicArc
        >>> amphi_hel = AmphipathicArc(4, 10, 25)
        >>> amphi_hel.generate_sequences(100)
        >>> amphi_hel.sequences
        ['YLYANLRQE', 'GVKPRIK', 'RWKKKVKDSVKDFEKRFKDIEKRIQRKLA', 'KIKEQLRNSVSGWHRN']
        """
        self.clean()
        
        if arcsize == 'mixed':  # generate mixed arc sizes
            idx = [[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                   [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                   [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                   [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                   [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]]
                
        elif arcsize == 100:
            idx = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
        elif arcsize == 140:
            idx = [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
        elif arcsize == 180:
            idx = [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0]
        elif arcsize == 220:
            idx = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0]
        elif arcsize == 260:
            idx = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
        else:
            raise AttributeError("Arc size unknown, choose among: 80, 120, 160, 200, 240 or None (= mixed).")

        idxcycle = cycle(idx).next
        idx = idxcycle()
        for s in range(self.seqnum):
            seq = []
            icycle = cycle(idx).next  # jumping from one probability to next one in idx array
            i = icycle()
            for n in range(random.choice(range(self.lenmin, self.lenmax + 1))):
                seq.append(random.choice(self.AAs, p=self.prob_amphihel[i]))
                i = icycle()
            self.sequences.append(''.join(seq))
    
    def make_H_gradient(self):
        """Method to mutate the generated sequences to have a hydrophobic gradient by substituting the last third of
        the sequence amino acids to hydrophobic.

        :return: A list of sequences in :py:attr:`sequences`
        :Example:

        >>> amphi_grad = AmphipathicArc(10, 7, 30)
        >>> amphi_grad.generate_sequences(180)
        >>> amphi_grad.make_H_gradient()
        >>> amphi_grad.sequences
        """
        for s in range(len(self.sequences)):
            seq = list(self.sequences[s])
            for aa in range(1, len(seq) / 3 + 1):
                seq[-aa] = random.choice(self.AAs, p=self.prob_amphihel[1])
            self.sequences[s] = ''.join(seq)


class HelicesACP(BaseSequence):
    """Class for peptides sequences with the amino acid probability of alpha-helical ACPs.

    This class incorporates methods for generating presumed alpha-helical peptides with the amino acid probability
    distribution of alpha-helical ACPs. For each of the positions in the helix (1-18) the amino acid distribution among
    62 anuran and hymenopteran alpha-helical ACPs was computed and is used to design the new sequences
    (G. Gabernet, A. T. Müller, J. A. Hiss, G. Schneider, *Med. Chem. Commun.* **2016**, 7, 2232–2245.).

    .. image:: ../docs/static/helixplot_legend.png
        :height: 300px
    """
    
    def generate_sequences(self):
        """Method to generate the sequences with the mentioned amino acid probabilities.

        :return: A list of potentially helical peptides with the amino acid distribution of ACP helical peptides
            according to the position in the helix wheel.

        :Example:

        >>> from modlamp.sequences import HelicesACP
        >>> helACP = HelicesACP(4, 7, 18)
        >>> helACP.generate_sequences()
        >>> helACP.sequences
        ['FLFDVAKKVAGTALT', 'GLGIILGAGG', 'GLRIKLGVWAKKA', 'GFWGFIKTI']
        """
        self.clean()
        for s in range(self.seqnum):
            seq = []
            for l in range(np.random.choice(range(self.lenmin, self.lenmax + 1))):
                l = l - 18 * (l / 18)  # for helices >18aa, the probabilities start from the beginning again
                seq.append(np.random.choice(self.AAs, p=self.prob_ACPhel[:, l]))
            self.sequences.append(''.join(seq))


class MixedLibrary(BaseSequence):
    """Class for holding a virtual peptide library.

    This class :class:`MixedLibrary` incorporates methods to generate a virtual peptide library composed out of different
    sub-libraries. The available library subtypes are all from the classes :class:`Centrosymmetric`, :class:`Helices`,
    :class:`Kinked`, :class:`Oblique` or :class:`Random`.
    """
    
    def __init__(self, seqnum, centrosymmetric=1, centroasymmetric=1, helix=1, kinked=1, oblique=1, rand=1, randAMP=1,
                 randAMPnoCM=1):
        """initializing method of the class :class:`MixedLibrary`. Except from **number**, all other parameters are
        ratios of sequences of the given sequence class.

        :param seqnum: {int} number of sequences to be generated
        :param centrosymmetric: ratio of symmetric centrosymmetric sequences in the library
        :param centroasymmetric: ratio of asymmetric centrosymmetric sequences in the library
        :param helix: ratio of amphipathic helical sequences in the library
        :param kinked: ratio of kinked amphipathic helical sequences in the library
        :param oblique: ratio of oblique oriented amphipathic helical sequences in the library
        :param rand: ratio of random sequneces in the library
        :param randAMP: ratio of random sequences with APD2 amino acid distribution in the library
        :param randAMPnoCM: ratio of random sequences with APD2 amino acid distribution without Cys and Met in the library

        .. warning::
            If duplicate sequences are created, these are removed during the creation process. It is therefore quite
            probable that you will not get the exact size of library that you entered as the parameter **number**. If you
            generate a small library, it can also happen that the size is bigger than expected, because ratios are rounded.
        """
        super(MixedLibrary, self).__init__(seqnum)  # inherit methods and some attributes from BaseSequence
        self.libsize = int(seqnum)
        norm = float(sum((centrosymmetric, centroasymmetric, helix, kinked, oblique, rand, randAMP, randAMPnoCM)))
        self.ratios = {'sym': float(centrosymmetric) / norm, 'asy': float(centroasymmetric) / norm,
                       'hel': float(helix) / norm, 'knk': float(kinked) / norm, 'obl': float(oblique) / norm,
                       'ran': float(rand) / norm, 'AMP': float(randAMP) / norm, 'nCM': float(randAMPnoCM) / norm}
        self.nums = {'sym': int(round(float(self.libsize) * self.ratios['sym'], ndigits=0)),
                     'asy': int(round(float(self.libsize) * self.ratios['asy'], ndigits=0)),
                     'hel': int(round(float(self.libsize) * self.ratios['hel'], ndigits=0)),
                     'knk': int(round(float(self.libsize) * self.ratios['knk'], ndigits=0)),
                     'obl': int(round(float(self.libsize) * self.ratios['obl'], ndigits=0)),
                     'ran': int(round(float(self.libsize) * self.ratios['ran'], ndigits=0)),
                     'AMP': int(round(float(self.libsize) * self.ratios['AMP'], ndigits=0)),
                     'nCM': int(round(float(self.libsize) * self.ratios['nCM'], ndigits=0))}
    
    def generate_sequences(self):
        """This method generates a virtual sequence library with the subtype ratios initialized in class :class:`MixedLibrary()`.
        All sequences are between 7 and 28 amino acids in length.

        :return: a virtual library of sequences in the attribute :py:attr:`sequences`, the sub-library class names in
            :py:attr:`names`, the number of sequences generated for each class in :py:attr:`nums` and the library size in
            :py:attr:`libsize`.
        :Example:

        >>> lib = MixedLibrary(10000, centrosymmetric=5, centroasymmetric=5, helix=3, kinked=3, oblique=2, rand=10,
        randAMP=10,randAMPnoCM=5)
        >>> lib.generate_sequences()
        >>> lib.libsize  # as duplicates were present, the library does not have the size that was sepecified
        9126
        >>> lib.sequences
        ['RHTHVAGSWYGKMPPSPQTL','MRIKLRKIPCILAC','DGINKEVKDSYGVFLK','LRLYLRLGRVWVRG','GKLFLKGGKLFLKGGKLFLKG',...]
        >>> lib.nums
        {'AMP': 2326,
        'asy': 1163,
        'hel': 698,
        'knk': 698,
        'nCM': 1163,
        'obl': 465,
        'ran': 2326,
        'sym': 1163}
        """
        Cs = Centrosymmetric(self.nums['sym'])
        Cs.generate_sequences(symmetry='symmetric')
        Ca = Centrosymmetric(self.nums['asy'])
        Ca.generate_sequences(symmetry='asymmetric')
        H = Helices(7, 28, self.nums['hel'])
        H.generate_sequences()
        K = Kinked(7, 28, self.nums['knk'])
        K.generate_sequences()
        O = Oblique(7, 28, self.nums['obl'])
        O.generate_sequences()
        R = Random(7, 28, self.nums['ran'])
        R.generate_sequences('rand')
        Ra = Random(7, 28, self.nums['AMP'])
        Ra.generate_sequences('AMP')
        Rc = Random(7, 28, self.nums['nCM'])
        Rc.generate_sequences('AMPnoCM')
        
        # TODO: update libnums according to real numbers
        
        sequences = Cs.sequences + Ca.sequences + H.sequences + K.sequences + O.sequences + R.sequences \
                    + Ra.sequences + Rc.sequences
        names = ['sym'] * self.nums['sym'] + ['asy'] * self.nums['asy'] + ['hel'] * self.nums['hel'] + \
                ['knk'] * self.nums['knk'] + ['obl'] * self.nums['obl'] + ['ran'] * self.nums['ran'] + \
                ['AMP'] * self.nums['AMP'] + ['nCM'] * self.nums['nCM']
        # combining sequence and name to remove duplicates
        comb = []
        for i, s in enumerate(sequences):
            comb.append(s + '_' + names[i])
        comb = set(comb)
        # remove duplicates
        for c in comb:
            self.sequences.append(c.split('_')[0])
            self.names.append(c.split('_')[1])
        # update libsize and nums
        self.libsize = len(self.sequences)
        self.nums = {k: self.names.count(k) for k in self.nums.keys()}  # update the number of sequences for every class
    
    def prune_library(self, newsize):
        """Method to cut down a library to the given new size.

        :param newsize: new desired size of the mixed library
        :return: adapted library with corresponding attributes (sequences, names, libsize, nums)
        """
        self.names = self.names[:newsize]
        self.sequences = self.sequences[:newsize]
        self.libsize = len(self.sequences)
        self.nums = {k: self.names.count(k) for k in self.nums.keys()}  # update the number of sequences for every class


class Hepahelices(BaseSequence):
    """Class for peptide sequences probable to form helices and include a heparin-binding-domain.

    This class is used to construct presumed amphipathic helices that include a heparin-binding-domain (HBD)
    probable to bind heparin. The HBD sequence for alpha-helices usually has the following form: **XBBBXXBX**
    (B: basic AA; X: hydrophobic, uncharged AA, with mainly Ser & Gly).

    :Reference: Munoz, E. M. & Linhardt, R. J. Heparin-Binding Domains in Vascular Biology. *Arterioscler. Thromb.
        Vasc. Biol.* **24**, 1549–1557 (2004).

    .. versionadded:: v2.3.1
    """
    
    def generate_sequences(self):
        """Method to generate helical sequences with class features defined in :class:`Hepahelices()`

        :return: In the attribute :py:attr:`sequences`: a list of sequences including a heparin-binding-domain.
        :Example:

        >>> h = Hepahelices(10, 8,21)  # minimal length: 8, maximal length: 50
        >>> h.generate_sequences()
        >>> h.sequences
        ['GRLARSLKRKLNRLVRGGGRLVRGGG', 'IRSIRRRLSKLARSLGRGARSLGRG', 'RAVKRKVNKLLKGAAKVLKGAAKVLKGAAK', ... ]
        """
        self.clean()
        for s in range(self.seqnum):  # for the number of sequences to generate
            # generate heparin binding domain with the from HBBBHPBH (H: hydrophobic, B: basic, P: polar)
            hbd = [random.choice(self.AA_hyd)] + [random.choice(self.AA_basic)] + [random.choice(self.AA_basic)] + \
                  [random.choice(self.AA_basic)] + [random.choice(self.AA_hyd)] + [random.choice(self.AA_polar)] + \
                  [random.choice(self.AA_basic)] + [random.choice(self.AA_hyd)]
            # generate amphipathic block to add in front of HBD
            bef = [random.choice(self.AA_hyd)] + [random.choice(['A', 'G'])] + [random.choice(self.AA_basic)] + \
                  [random.choice(self.AA_hyd)] + [random.choice(self.AA_hyd)] + [random.choice(self.AA_basic)] + \
                  [random.choice(['A', 'G', 'S', 'T'])]
            # generate amphipathic block to add after HBD
            aft = [random.choice(self.AA_hyd)] + [random.choice(self.AA_basic)] + \
                  [random.choice(['A', 'G', 'S', 'T'])] + [random.choice(self.AA_hyd)] + \
                  [random.choice(['A', 'G'])] + [random.choice(self.AA_basic)] + [random.choice(self.AA_hyd)]
            l = random.choice(range(self.lenmin, self.lenmax + 1))  # total sequence length
            try:
                r = l - 8  # remaining empty positions in sequence
                b = random.choice(r)  # positions before HBD
                a = r - b  # positions after HBD
                seq = 3 * bef + hbd + 3 * aft
                seq = seq[21 - b: 29 + a]
            except ValueError:  # if sequence length is 8, take HBD as whole sequence
                seq = hbd
            
            self.sequences.append(''.join(seq))


class AMPngrams(BaseSequence):
    """Class for sequence generation from the most prominent ngrams (2, 3, 4grams) found in all natural AMP
    sequences extracted from the `APD3 <http://aps.unmc.edu/AP/>`_ (version August 2016 with 2727 sequences).
    For all 2, 3 and 4grams, all possible ngrams were generated from all sequences and the top 50 most frequent
    assembled into a list. Finally, leading and tailing spaces were striped and duplicates as well as ngrams containing
    spaces were removed.
    
    .. versionadded:: v2.4.1
    
    .. seealso:: :py:func:`modlamp.core.ngrams_apd()`
    """
    
    def __init__(self, seqnum, n_min=3, n_max=11):
        """
        :param seqnum: {int} number of sequences to be generated
        :param n_min: {int} minimum number of ngrams to take for sequence assembly
        :param n_max: {int} maximum number of ngrams to take for sequence assembly
        :Example:
        
        >>> s = AMPngrams(10)
        >>> s.generate_sequences()
        >>> s.sequences
        ['LAKSLGAGKYGGGKA', 'KAALESCVGGGGC', 'GCSGKAAAAAVG', 'GAASCKPCGEAKGLKVCY', 'IGGGCKITGESCVAGLWCGESTCGCSG', ...]
        >>> s.ngrams
        array(['AGK', 'CKI', 'RR', 'YGGG', 'LSGL', 'RG', 'YGGY', 'PRP', 'LGGG', ...]
        """
        super(AMPngrams, self).__init__(seqnum)  # inherit from BaseSequences and combine with n_min & n_max
        self.ngrams = ngrams_apd()
        self.n_min = n_min
        self.n_max = n_max
    
    def generate_sequences(self):
        """Method to generate sequences out of APD3 ngrams stored in :py:attr:`ngrams`.
        
        :return: list of sequences in :py:attr:`sequences`
        """
        for _ in range(self.seqnum):
            size = np.random.randint(self.n_min, self.n_max)  # number of ngrams to choose from list to build sequence
            # build sequence from a random selection of ngrams
            self.sequences.append(''.join(self.ngrams[np.random.randint(0, self.ngrams.shape[0], size=size)]))

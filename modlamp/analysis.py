# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.analysis

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module can be used for diverse analysis of given peptide libraries.
"""
import numpy as np
from modlamp.core import count_aa


class Global(object):
    """
    Base class for amino acid sequence library analysis
    """
    def __init__(self, library):
        if type(library) == np.ndarray:
            self.library = library
        else:
            self.library = np.array(library)

        # check if library consists of sub-libraries
        if len(self.library.shape) > 1:
            self.libshape = self.library.shape[0]
        else:
            self.libshape = 1

        self.aafreq = np.zeros((self.libshape, 20), dtype='float64')
            
    def calc_aa_freq(self):
        """Method to get the frequency of every amino acid in the library. If the library consists of sub-libraries,
        the frequencies of these are calculated independently.
        
        :return: {numpy.ndarray} amino acid frequencies in the attribute :py:attr:`aafreq`. The values are oredered
            alphabetically.
        :Example:
        
        >>> g = Global(sequences)
        >>> g.calc_aa_freq()
        >>> g.aafreq
            array([[ 0.08250071,  0.        ,  0.02083928,  0.0159863 ,  0.1464459 ,
                     0.04795889,  0.06622895,  0.0262632 ,  0.12988867,  0.        ,
                     0.09192121,  0.03111619,  0.01712818,  0.04852983,  0.05937768,
                     0.07079646,  0.04396232,  0.0225521 ,  0.05994862,  0.01855552]])
        """
        for l in range(self.libshape):
            if self.libshape == 1:
                concatseq = ''.join(self.library)
            else:
                concatseq = ''.join(self.library[l])
            d_aa = count_aa(concatseq)
            self.aafreq[l] = [v / float(len(concatseq)) for v in d_aa.values()]

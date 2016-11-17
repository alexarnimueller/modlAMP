# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.analysis

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module can be used for diverse analysis of given peptide libraries.
"""
import matplotlib.pyplot as plt
import numpy as np

from modlamp.core import count_aa
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor


class GlobalAnalysis(object):
    """
    Base class for amino acid sequence library analysis
    
    .. versionadded:: 2.6.0
    """
    
    def __init__(self, library):
        if type(library) == np.ndarray:
            self.library = library
        else:
            self.library = np.array(library)
        
        # reshape library to 2D array if without sub-libraries
        if len(self.library.shape) == 1:
            self.library = self.library.reshape((1, -1))
        
        self.aafreq = np.zeros((self.library.shape[0], 20), dtype='float64')
        self.H = np.zeros(self.library.shape, dtype='float64')
        self.uH = np.zeros(self.library.shape, dtype='float64')
        self.charge = np.zeros(self.library.shape, dtype='float64')
        self.len = np.zeros(self.library.shape, dtype='float64')
    
    def calc_aa_freq(self, plot=True):
        """Method to get the frequency of every amino acid in the library. If the library consists of sub-libraries,
        the frequencies of these are calculated independently.
        
        :param plot: {bool} whether the amino acid frequencies should be plotted in a histogram.
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
        
        .. image:: ../docs/static/aa_anal.png
            :height: 300px
        """
        for l in range(self.library.shape[0]):
            concatseq = ''.join(self.library[l])
            d_aa = count_aa(concatseq)
            self.aafreq[l] = [v / float(len(concatseq)) for v in d_aa.values()]
            
            if plot:
                fig, ax = plt.subplots()
                
                for a in range(20):
                    plt.bar(a - 0.45, self.aafreq[l, a], 0.9, color='#83AF9B')
                
                plt.xlim([-0.75, 19.75])
                plt.ylim([0, max(self.aafreq[l, :]) + 0.05])
                plt.xticks(range(20), d_aa.keys(), fontweight='bold')
                plt.ylabel('Amino Acid Frequency', fontweight='bold')
                plt.title('Amino Acid Distribution', fontsize=16, fontweight='bold')
                
                # only left and bottom axes, no box
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                
                plt.show()
    
    def calc_H(self):
        """Method for calculating global hydrophobicity (Eisenberg scale) of all sequences in the library.
        
        :return: {numpy.ndarray} Eisenberg hydrophobicities in the attribute :py:attr:`H`.
        
        .. seealso:: modlamp.descriptors.PeptideDescriptor.calculate_global()
        """
        for l in range(self.library.shape[0]):
            d = PeptideDescriptor(self.library[l].tolist(), 'eisenberg')
            d.calculate_global()
            self.H[l] = d.descriptor[:, 0]
    
    def calc_uH(self, window=1000, angle=100, modality='max'):
        """Method for calculating hydrophobic moments (Eisenberg scale) for all sequences in the library.
        
        :param window: {int} amino acid window in which to calculate the moment. If the sequence is shorter than the
            window, the length of the sequence is taken. So if the default window of 1000 is chosen, for all sequences
            shorter than 1000, the **global** hydrophobic moment will be calculated. Otherwise, the maximal
            hydrophiobic moment for the chosen window size found in the sequence will be returned.
        :param angle: {int} angle in which to calculate the moment. **100** for alpha helices, **180** for beta sheets.
        :param modality: {'max' or 'mean'} calculate respectively maximum or mean hydrophobic moment.
        :return: {numpy.ndarray} calculated hydrophobic moments in the attribute :py:attr:`uH`.
        
        .. seealso:: modlamp.descriptors.PeptideDescriptor.calculate_moment()
        """
        for l in range(self.library.shape[0]):
            d = PeptideDescriptor(self.library[l].tolist(), 'eisenberg')
            d.calculate_moment(window=window, angle=angle, modality=modality)
            self.uH[l] = d.descriptor[:, 0]
    
    def calc_charge(self, ph=7.0, amide=True):
        """Method to calculate the total molecular charge at a given pH for all sequences in the library.
        
        :param ph: {float} ph at which to calculate the peptide charge.
        :param amide: {boolean} whether the sequences have an amidated C-terminus (-> charge += 1).
        :return: {numpy.ndarray} calculated charges in the attribute :py:attr:`charge`.
        """
        for l in range(self.library.shape[0]):
            d = GlobalDescriptor(self.library[l].tolist())
            d.calculate_charge(ph=ph, amide=amide)
            self.charge[l] = d.descriptor[:, 0]
    
    def calc_len(self):
        """Method to get the sequence length of all sequences in the library.
        
        :return: {numpy.ndarray} sequence lengths in the attribute :py:attr:`len`.
        """
        for l in range(self.library.shape[0]):
            d = GlobalDescriptor(self.library[l].tolist())
            d.length()
            self.len[l] = d.descriptor[:, 0]
    
    def plot_summary(self):
        """Method to generate a visual summary of different characteristics of the given library. The class methods
        are used with their standard options.
    
        :return: visual summary (plot) of the library characteristics.
        """
        # calculate all global properties
        self.calc_len()
        self.calc_aa_freq(plot=False)
        self.calc_charge()
        self.calc_H()
        self.calc_uH()
        
        # plot
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
        ((ax1, ax2), (ax3, ax4)) = axes
        plt.suptitle('Global Summary', fontweight='bold', fontsize=16.)
        
        # length histogram
        ax1.hist(self.len.T, len(range(int(np.max(self.len) - np.min(self.len)))),
                 normed=1, facecolor='blue', alpha=0.5, align='right', rwidth=0.9)
        ax1.set_xlabel('Sequence Length', fontweight='bold', fontsize=14.)
        ax1.set_ylabel('Fraction', fontweight='bold', fontsize=14.)
        ax1.set_xlim(np.min(self.len) - .5, np.max(self.len) + 1.5)
        
        # aa histogram
        d_aa = count_aa('')
        for a in range(20):
            ax2.bar(a - 0.45, self.aafreq[0, a], 0.9, color='#83AF9B')
        ax2.set_xlim([-0.75, 19.75])
        ax2.set_ylim([0, max(self.aafreq[0, :]) + 0.05])
        ax2.set_xticks(range(20))
        ax2.set_xticklabels(d_aa.keys(), fontweight='bold')
        ax1.set_ylabel('Fraction', fontweight='bold', fontsize=14.)
        ax2.set_xlabel('Amino Acids', fontweight='bold', fontsize=14.)
        
        # hydophobicity violin plot
        ax3.violinplot(self.H, widths=0.33, showmeans=True, showmedians=False)
        ax3.set_xticks([])  # xticks off
        ax3.set_ylabel('Global Hydrophobicity', fontweight='bold', fontsize=14.)
        
        # hydrophobic moment violin plot
        ax4.violinplot(self.uH, widths=0.33, showmeans=True, showmedians=False)
        ax4.set_xticks([])  # xticks off
        ax4.set_ylabel('Global Hydrophobic Moment', fontweight='bold', fontsize=14.)
        
        # only left and bottom axes, no box
        for a in [ax1, ax2, ax3, ax4]:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.xaxis.set_ticks_position('bottom')
            a.yaxis.set_ticks_position('left')
        
        plt.show()

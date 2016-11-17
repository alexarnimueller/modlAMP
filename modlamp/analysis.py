# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.analysis

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module can be used for diverse analysis of given peptide libraries.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from modlamp.core import count_aa
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor


class GlobalAnalysis(object):
    """
    Base class for amino acid sequence library analysis
    
    .. versionadded:: 2.6.0
    """
    
    def __init__(self, library):
        """
        :param library: {numpy.ndarray} sequence library, if 2D, the rows are considered as sub-libraries.
        """
        
        if type(library) == np.ndarray:
            self.library = library
            # reshape library to 2D array if without sub-libraries
            if len(self.library.shape) == 1:
                self.library = self.library.reshape((1, -1))
    
            self.aafreq = np.zeros((self.library.shape[0], 20), dtype='float64')
            self.H = np.zeros(self.library.shape, dtype='float64')
            self.uH = np.zeros(self.library.shape, dtype='float64')
            self.charge = np.zeros(self.library.shape, dtype='float64')
            self.len = np.zeros(self.library.shape, dtype='float64')
        else:
            raise TypeError('Input library must be of type numpy.ndarray!')
    
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
        :Example:
        
        >>> g = GlobalAnalysis(np.array([seqs1, seqs2, seqs3])  # seqs being lists of sequences
        >>> g.plot_summary()
        
        .. image:: ../docs/static/summary.png
            :height: 500px
        """
        # calculate all global properties
        self.calc_len()
        self.calc_aa_freq(plot=False)
        self.calc_charge()
        self.calc_H()
        self.calc_uH()
        
        # plot settings
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
        ((ax1, ax2), (ax3, ax4)) = axes
        plt.suptitle('Summary', fontweight='bold', fontsize=16.)
        labels = ['Lib ' + str(x + 1) for x in range(self.library.shape[0])]
        colors = ['#4E395D', '#8EBE94', '#DC5B3E', '#827085', '#CCFC8E', '#9CC4E4']
        num = len(labels)

        for a in [ax1, ax2, ax3, ax4]:
            # only left and bottom axes, no box
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.xaxis.set_ticks_position('bottom')
            a.yaxis.set_ticks_position('left')
        
        # 1 length histogram
        ax1.hist(self.len.T, len(range(int(np.max(self.len) - np.min(self.len)))),
                 normed=1, alpha=0.7, align='left', rwidth=0.9, histtype='bar', label=labels, color=colors[:num])
        ax1.set_xlabel('Sequence Length', fontweight='bold', fontsize=14.)
        ax1.set_ylabel('Fraction', fontweight='bold', fontsize=14.)
        ax1.set_xlim(np.min(self.len) - 1.5, np.max(self.len) + .5)
        ax1.legend()
        
        # 2 aa bar plot
        d_aa = count_aa('')
        hands = [mpatches.Patch(label=labels[i], linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.7)
                 for i in range(len(labels))]
        w = 0.9
        offsets = np.arange(start=-(w / 2), step=(w / num), stop=(w / 2))
        for i, l in enumerate(self.aafreq):
            for a in range(20):
                ax2.bar(a - offsets[i] - (0.5 * w / num), l[a], w / num, color=colors[i], alpha=0.7)
        ax2.set_xlim([-0.75, 19.75])
        ax2.set_ylim([0, max(self.aafreq[0, :]) + 0.05])
        ax2.set_xticks(range(20))
        ax2.set_xticklabels(d_aa.keys(), fontweight='bold')
        ax2.set_ylabel('Fraction', fontweight='bold', fontsize=14.)
        ax2.set_xlabel('Amino Acids', fontweight='bold', fontsize=14.)
        ax2.legend(handles=hands, labels=labels)
        
        # 3 hydophobicity violin plot
        for i, l in enumerate(self.H):
            vplot = ax3.violinplot(l, positions=[i + 1], widths=0.5, showmeans=True, showmedians=False)
            vplot['cbars'].set_edgecolor('black')
            vplot['cmins'].set_edgecolor('black')
            vplot['cmeans'].set_edgecolor('black')
            vplot['cmaxes'].set_edgecolor('black')
            vplot['cmeans'].set_linestyle('--')
            for pc in vplot['bodies']:
                pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
                pc.set_alpha(0.7)
                pc.set_label(labels[i])
        ax3.set_xticks([x + 1 for x in range(len(labels))])
        ax3.set_xticklabels(labels, fontweight='bold')
        ax3.set_ylabel('Global Hydrophobicity', fontweight='bold', fontsize=14.)
        
        # 4 hydrophobic moment violin plot
        for i, l in enumerate(self.uH):
            vplot = ax4.violinplot(l, positions=[i + 1], widths=0.5, showmeans=True, showmedians=False)
            vplot['cbars'].set_edgecolor('black')
            vplot['cmins'].set_edgecolor('black')
            vplot['cmeans'].set_edgecolor('black')
            vplot['cmaxes'].set_edgecolor('black')
            vplot['cmeans'].set_linestyle('--')
            for pc in vplot['bodies']:
                pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
                pc.set_alpha(0.7)
                pc.set_label(labels[i])
        ax4.set_xticks([x + 1 for x in range(len(labels))])
        ax4.set_xticklabels(labels, fontweight='bold')
        ax4.set_ylabel('Global Hydrophobic Moment', fontweight='bold', fontsize=14.)
        
        plt.show()

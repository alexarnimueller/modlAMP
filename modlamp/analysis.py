# -*- coding: utf-8 -*-
"""
.. currentmodule:: modlamp.analysis

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module can be used for diverse analysis of given peptide libraries.
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from modlamp.core import count_aas
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

__author__ = "Alex Müller, Gisela Gabernet"
__docformat__ = "restructuredtext en"


class GlobalAnalysis(object):
    """
    Base class for amino acid sequence library analysis
    
    .. versionadded:: 2.6.0
    """
    
    def __init__(self, library, names=None):
        """
        :param library: {list, numpy.ndarray, pandas.DataFrame} sequence library, if 2D, the rows are considered as
            sub-libraries.
        :param names: {list} list of library names to plot as labels and legend
        :Example:
        
        >>> g = GlobalAnalysis(['GLFDIVKKVVGALG', 'KLLKLLKKLLKLLK', ...], names=['Library1'])
        """
        self.shapes = list()  # find out about library structure and check if libraries are of different sizes
        if isinstance(library[0], list):
            for i, l in enumerate(library):
                self.shapes.append(len(l))
                self.library = np.array(library, dtype='object')
        else:
            if type(library) == np.ndarray:
                self.library = library
            elif type(library) == pd.core.frame.DataFrame:
                if library.shape[0] > library.shape[1]:  # if each library is a column
                    self.library = library.values.T
                    if not names:
                        self.libnames = library.columns.values.tolist()  # take library names from column headers
                else:  # if each library is a row
                    self.library = library.values
                    if not names:
                        self.libnames = library.index.values.tolist()  # take library names from row headers
            else:
                self.library = np.array(library)
        
        if names:
            self.libnames = names
        
        # reshape library to 2D array if without sub-libraries
        if len(self.library.shape) == 1:
            self.library = self.library.reshape((1, -1))
            if not names:
                names = ['Lib1']
        
        if not names:
            self.libnames = ['Lib ' + str(x + 1) for x in range(self.library.shape[0])]
        
        self.aafreq = np.zeros((self.library.shape[0], 20), dtype='float64')
        self.H = list()
        self.uH = list()
        self.charge = list()
        self.len = list()
        self.AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    def calc_aa_freq(self, plot=True, color='#83AF9B', filename=None):
        """Method to get the frequency of every amino acid in the library. If the library consists of sub-libraries,
        the frequencies of these are calculated independently.
        
        :param plot: {bool} whether the amino acid frequencies should be plotted in a histogram.
        :param color: {str} color of the plot
        :param filename: {str} filename to save the plot to, if None, the plot is shown
        :return: {numpy.ndarray} amino acid frequencies in the attribute :py:attr:`aafreq`. The values are oredered
            alphabetically.
        :Example:
        
        >>> g = GlobalAnalysis(sequences)  # sequences being a list / array of amino acid sequences
        >>> g.calc_aa_freq()
        >>> g.aafreq
            array([[ 0.08250071,  0.        ,  0.02083928,  0.0159863 ,  0.1464459 ,
                     0.04795889,  0.06622895,  0.0262632 ,  0.12988867,  0.        ,
                     0.09192121,  0.03111619,  0.01712818,  0.04852983,  0.05937768,
                     0.07079646,  0.04396232,  0.0225521 ,  0.05994862,  0.01855552]])
        
        .. image:: ../docs/static/AA_dist.png
            :height: 300px
        """
        for l in range(self.library.shape[0]):
            concatseq = ''.join(self.library[l])
            d_aa = count_aas(concatseq)
            self.aafreq[l] = [d_aa[a] for a in self.AAs]
            if plot:
                fig, ax = plt.subplots()
                
                for a in range(20):
                    plt.bar(a, self.aafreq[l, a], 0.9, color=color)
                
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

                if filename:
                    plt.savefig(filename)
                else:
                    plt.show()
    
    def calc_H(self, scale='eisenberg'):
        """Method for calculating global hydrophobicity (Eisenberg scale) of all sequences in the library.
        
        :param scale: {str} hydrophobicity scale to use. For available scales,
            see :class:`modlamp.descriptors.PeptideDescriptor`.
        :return: {numpy.ndarray} Eisenberg hydrophobicities in the attribute :py:attr:`H`.
        
        .. seealso:: :func:`modlamp.descriptors.PeptideDescriptor.calculate_global()`
        """
        for l in range(self.library.shape[0]):
            d = PeptideDescriptor(self.library[l], scale)
            d.calculate_global()
            self.H.append(d.descriptor[:, 0])
    
    def calc_uH(self, window=1000, angle=100, modality='max'):
        """Method for calculating hydrophobic moments (Eisenberg scale) for all sequences in the library.
        
        :param window: {int} amino acid window in which to calculate the moment. If the sequence is shorter than the
            window, the length of the sequence is taken. So if the default window of 1000 is chosen, for all sequences
            shorter than 1000, the **global** hydrophobic moment will be calculated. Otherwise, the maximal
            hydrophiobic moment for the chosen window size found in the sequence will be returned.
        :param angle: {int} angle in which to calculate the moment. **100** for alpha helices, **180** for beta sheets.
        :param modality: {'max' or 'mean'} calculate respectively maximum or mean hydrophobic moment.
        :return: {numpy.ndarray} calculated hydrophobic moments in the attribute :py:attr:`uH`.
        
        .. seealso:: :func:`modlamp.descriptors.PeptideDescriptor.calculate_moment()`
        """
        for l in range(self.library.shape[0]):
            d = PeptideDescriptor(self.library[l], 'eisenberg')
            d.calculate_moment(window=window, angle=angle, modality=modality)
            self.uH.append(d.descriptor[:, 0])
    
    def calc_charge(self, ph=7.0, amide=True):
        """Method to calculate the total molecular charge at a given pH for all sequences in the library.
        
        :param ph: {float} ph at which to calculate the peptide charge.
        :param amide: {boolean} whether the sequences have an amidated C-terminus (-> charge += 1).
        :return: {numpy.ndarray} calculated charges in the attribute :py:attr:`charge`.
        """
        for l in range(self.library.shape[0]):
            d = GlobalDescriptor(self.library[l])
            d.calculate_charge(ph=ph, amide=amide)
            self.charge.append(d.descriptor[:, 0])
    
    def calc_len(self):
        """Method to get the sequence length of all sequences in the library.
        
        :return: {numpy.ndarray} sequence lengths in the attribute :py:attr:`len`.
        """
        for l in range(self.library.shape[0]):
            d = GlobalDescriptor(self.library[l])
            d.length()
            self.len.append(d.descriptor[:, 0])
    
    def plot_summary(self, filename=None, colors=None, plot=True):
        """Method to generate a visual summary of different characteristics of the given library. The class methods
        are used with their standard options.
    
        :param filename: {str} path to save the generated plot to.
        :param colors: {str / list} color or list of colors to use for plotting. e.g. '#4E395D', 'red', 'k'
        :param plot: {boolean} whether the plot should be created or just the features are calculated
        :return: visual summary (plot) of the library characteristics (if ``plot=True``).
        :Example:
        
        >>> g = GlobalAnalysis([seqs1, seqs2, seqs3])  # seqs being lists / arrays of sequences
        >>> g.plot_summary()
        
        .. image:: ../docs/static/summary.png
            :height: 600px
        """
        # calculate all global properties
        self.calc_len()
        self.calc_aa_freq(plot=False)
        self.calc_charge(ph=7.4, amide=True)
        self.calc_H()
        self.calc_uH()
        
        if plot:
            
            # plot settings
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 15))
            ((ax2, ax5, ax1), (ax3, ax4, ax6)) = axes
            plt.suptitle('Summary', fontweight='bold', fontsize=16.)
            labels = self.libnames
            if not colors:
                colors = ['#FA6900', '#69D2E7', '#542437', '#53777A', '#CCFC8E', '#9CC4E4']
            num = len(labels)
            
            for a in [ax1, ax2, ax3, ax4, ax5, ax6]:
                # only left and bottom axes, no box
                a.spines['right'].set_visible(False)
                a.spines['top'].set_visible(False)
                a.xaxis.set_ticks_position('bottom')
                a.yaxis.set_ticks_position('left')
            
            # 1 length box plot
            box = ax1.boxplot(self.len, notch=1, vert=1, patch_artist=True)
            plt.setp(box['whiskers'], color='black')
            plt.setp(box['medians'], linestyle='-', linewidth=1.5, color='black')
            for p, patch in enumerate(box['boxes']):
                patch.set(facecolor=colors[p], edgecolor='black', alpha=0.8)
            ax1.set_ylabel('Sequence Length', fontweight='bold', fontsize=14.)
            ax1.set_xticks([x + 1 for x in range(len(labels))])
            ax1.set_xticklabels(labels, fontweight='bold')
            
            # 2 AA bar plot
            d_aa = count_aas('')
            hands = [mpatches.Patch(label=labels[i], facecolor=colors[i], alpha=0.8) for i in range(len(labels))]
            w = .9 / num  # bar width
            offsets = np.arange(start=-w, step=w, stop=num * w)  # bar offsets if many libs
            for i, l in enumerate(self.aafreq):
                for a in range(20):
                    ax2.bar(a - offsets[i], l[a], w, color=colors[i], alpha=0.8)
            ax2.set_xlim([-1., 20.])
            ax2.set_ylim([0, 1.05 * np.max(self.aafreq)])
            ax2.set_xticks(range(20))
            ax2.set_xticklabels(d_aa.keys(), fontweight='bold')
            ax2.set_ylabel('Fraction', fontweight='bold', fontsize=14.)
            ax2.set_xlabel('Amino Acids', fontweight='bold', fontsize=14.)
            ax2.legend(handles=hands, labels=labels)
            
            # 3 hydophobicity violin plot
            for i, l in enumerate(self.H):
                vplot = ax3.violinplot(l, positions=[i + 1], widths=0.5, showmeans=True, showmedians=False)
                # crappy adaptions of violin dictionary elements
                vplot['cbars'].set_edgecolor('black')
                vplot['cmins'].set_edgecolor('black')
                vplot['cmeans'].set_edgecolor('black')
                vplot['cmaxes'].set_edgecolor('black')
                vplot['cmeans'].set_linestyle('--')
                for pc in vplot['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.8)
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
                # crappy adaptions of violin dictionary elements
                vplot['cbars'].set_edgecolor('black')
                vplot['cmins'].set_edgecolor('black')
                vplot['cmeans'].set_edgecolor('black')
                vplot['cmaxes'].set_edgecolor('black')
                vplot['cmeans'].set_linestyle('--')
                for pc in vplot['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.8)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.5)
                    pc.set_alpha(0.7)
                    pc.set_label(labels[i])
            ax4.set_xticks([x + 1 for x in range(len(labels))])
            ax4.set_xticklabels(labels, fontweight='bold')
            ax4.set_ylabel('Global Hydrophobic Moment', fontweight='bold', fontsize=14.)
            
            # 5 charge histogram
            if self.shapes:  # if the library consists of different sized sub libraries
                bwidth = 1. / len(self.shapes)
                for i, c in enumerate(self.charge):
                    counts, bins = np.histogram(c, range=[-5, 20], bins=25, normed=True)
                    ax5.bar(bins[1:] + i * bwidth, counts, bwidth, color=colors[i], label=labels[i], alpha=0.8)
                    # ax5.hist(c, bins, alpha=0.7, align=alignments[i], rwidth=0.95 / len(self.shapes), histtype='bar',
                    #         normed=1, label=labels[i], color=colors[i])
            else:
                ax5.hist(self.charge, 25, normed=1, alpha=0.8, align='left', rwidth=0.95, histtype='bar', label=labels,
                         color=colors[:num])
            ax5.set_xlabel('Global Charge', fontweight='bold', fontsize=14.)
            ax5.set_ylabel('Fraction', fontweight='bold', fontsize=14.)
            ax5.set_xlim(-6, 21)
            ax5.text(0.95, 0.8, b'amide: $true$', verticalalignment='center', horizontalalignment='right',
                     transform=ax5.transAxes, fontsize=15)
            ax5.text(0.95, 0.75, b'pH:  $7.4$', verticalalignment='center', horizontalalignment='right',
                     transform=ax5.transAxes, fontsize=15)
            ax5.legend()
            
            # 6 3D plot
            ax6.spines['left'].set_visible(False)
            ax6.spines['bottom'].set_visible(False)
            ax6.set_xticks([])
            ax6.set_yticks([])
            ax6 = fig.add_subplot(2, 3, 6, projection='3d')
            for i, l in enumerate(range(num)):
                xt = self.H[l]  # find all values in x for the given target
                yt = self.charge[l]  # find all values in y for the given target
                zt = self.uH[l]  # find all values in y for the given target
                ax6.scatter(xt, yt, zt, c=colors[l], alpha=.8, s=25, label=labels[i])
            
            ax6.set_xlabel('H', fontweight='bold', fontsize=14.)
            ax6.set_ylabel('Charge', fontweight='bold', fontsize=14.)
            ax6.set_zlabel('uH', fontweight='bold', fontsize=14.)
            data_c = [item for sublist in self.charge for item in sublist]  # flatten charge data into one list
            data_H = [item for sublist in self.H for item in sublist]  # flatten H data into one list
            data_uH = [item for sublist in self.uH for item in sublist]  # flatten uH data into one list
            ax6.set_xlim([np.min(data_H), np.max(data_H)])
            ax6.set_ylim([np.min(data_c), np.max(data_c)])
            ax6.set_zlim([np.min(data_uH), np.max(data_uH)])
            ax6.legend(loc='best')
            
            if filename:
                plt.savefig(filename, dpi=200)
            else:
                plt.show()

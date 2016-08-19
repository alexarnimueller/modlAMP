# -*- coding: utf-8 -*-
"""
.. module:: modlamp.wetlab

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to load raw data files from wetlab experiments and calculate different
characteristics.

=============================        ============================================================================
Function                             Data
=============================        ============================================================================
:py:func:`load_AMPvsTMset`           Antimicrobial peptides versus trans-membrane sequences
:py:func:`load_helicalAMPset`        Helical antimicrobial peptides versus other helical peptides
:py:func:`load_ACPvsNeg`             Helical anticancer peptides versus other mixed sequences
:py:func:`load_AMPvsUniProt`         AMPs from the *APD3* versus other peptides from *UniProt*
=============================        ============================================================================
"""

from os import listdir, makedirs
from os.path import join, exists, splitext

import matplotlib.pyplot as plt
import numpy as np

from modlamp.descriptors import GlobalDescriptor

__author__ = "modlab"
__docformat__ = "restructuredtext en"


class CD:
    """
    Class to handle circular dichroism data files and calculate several ellipticity and helicity values.
    The class can only handle data files of the *Applied Photophysics* type.
    
    For explanations of different units used in CD spectroscopy,
    visit https://www.photophysics.com/resources/7-cd-units-conversions.
    """
    
    def __init__(self, directory, wmin, wmax, amide=True, pathlen=0.1):
        """Init method for class CD.
        
        :param directory: {str} directory containing all data files to be read. Files need a **.csv** ending
        :param wmin: {int} smalles wavelength measured
        :param wmax: {int} highest wavelength measured
        :param amide: {bool} specifies whether the sequences have amidated C-termini
        :param pathlen: {float} cuvette path length in cm
        """
        
        # read filenames from directory
        files = listdir(directory)
        self.filenames = [filename for filename in files if filename.endswith('.csv')]  # get all .csv files in dir
        
        # initialize attributes
        self.names = list()
        self.sequences = list()
        self.conc_umol = list()
        self.conc_mgml = list()
        self.mw = list()
        self.meanres_mw = list()
        self.solvent = list()
        self.circular_dichroism = list()
        self.molar_ellipticity = list()
        self.meanres_ellipticity = list()
        self.directory = directory
        self.wmin = wmin
        self.wmax = wmax
        self.amide = amide
        self.pathlen = pathlen
        
        self._read_header()  # call the _read_header function to fill up all attributes
        
    def _read_header(self):
        """Priveat method called by ``__init__`` to read all file headers into the class attributes and calculate
        sequence dependant values.
        
        :return: headers in class attributes.
        """
        
        d = GlobalDescriptor('X')  # template
        
        # loop through all files in the directory
        for i, file in enumerate(self.filenames):
            with open(join(self.directory, file)) as f:  # read first 4 lines as header, rest as data
                head = [next(f) for _ in range(4)]
                data = [next(f) for _ in range(4, (self.wmax - self.wmin) + 5)]

            # read headers into class attributes
            name = head[0].split('\r\n')[0]
            self.names.append(name)
            sequence = head[1].split('\r\n')[0]
            self.sequences.append(sequence)
            umol = float(head[2].split('\r\n')[0])
            self.conc_umol.append(umol)
            self.solvent.append(head[3].split('\r\n')[0])
            
            # read CD data
            wlengths = [int(line.split(',')[0]) for line in data]  # get rid of s***** line ends
            ellipts = [float(line.split(',')[1].split('\r\n')[0]) for line in data]
            self.circular_dichroism.append(np.array(zip(wlengths, ellipts)))
            
            # calculate MW and transform concentration to mg/ml
            d.sequences = [sequence]
            d.calculate_MW(amide=self.amide)
            self.mw.append(d.descriptor[0][0])
            self.conc_mgml.append(self.mw[i] * umol / 10**6)
            self.meanres_mw.append(self.mw[i] / (len(sequence) - 1))  # mean residue molecular weight (MW / n-1)
            
    def calc_molar_ellipticity(self):
        """Method to calculate the molar ellipticity for all loaded data in :py:attr:`circular_dichroism` according
        to the following formula:
        
        *molar ellipticity = (theta x MW x 100) / (conc x pathlength)*

        :return: {numpy array} molar ellipticity in :py:attr:`molar_ellipticity`
        """

        for i, data in enumerate(self.circular_dichroism):
            # calculate molar ellipticity: (theta * MW * 100) / (conc * pathlength); and concat. with wavelengths
            mol_ellipt = np.array(zip(data[:, 0], (data[:, 1] * self.mw[i] * 100) / (self.conc_mgml[i] * self.pathlen)))
            self.molar_ellipticity.append(mol_ellipt)

    def calc_meanres_ellipticity(self):
        """Method to calculate the mean residue ellipticity for all loaded data in :py:attr:`circular_dichroism`
        according to the following formula:
        
        *mean residue molar ellipticity = (theta x mean residue weight) / (10 x conc x pathlength)*
        
        *mean residue weight = MW / (n_residues - 1)*
        
        :return: {numpy array} molar ellipticity in :py:attr:`meanres_ellipticity`
        """
        
        for i, data in enumerate(self.circular_dichroism):
            # calculate molar ellipticity: (theta * mrw) / (10 * conc * pathlength); and concat. with wavelengths
            mol_ellipt = np.array(zip(data[:, 0], (data[:, 1] * self.meanres_mw[i] / (10 * self.conc_mgml[i] *
                                                                                      self.pathlen))))
            self.meanres_ellipticity.append(mol_ellipt)

    def plot(self, data='mean residue ellipticity'):
        """Method to generate CD plots for all read data in the initial directory.
        
        :param data: {str} which data should be plotted (``mean residue ellipticity``, ``molar ellipticity`` or
        ``circular dichroism``)
        :return: .pdf plots saved to the directory containing the read files.
        """
        
        # check if output folder exists already, else create one
        if not exists(join(self.directory, 'PDF')):
            makedirs(join(self.directory, 'PDF'))
        
        # check data option
        if data in ['mean residue ellipticity', 'molar ellipticity', 'circular dichroism']:
            # loop through all data
            for i, f in enumerate(self.filenames):
                if data == 'mean residue ellipticity':
                    d = self.meanres_ellipticity[i][:, 1] / 1000
                    y_label = r"$[\Theta] \ast 10^-3 (deg \ast cm^2 \ast dmol^-1)$"
                elif data == 'molar ellipticity':
                    d = self.molar_ellipticity[i][:, 1]
                    y_label = r"$[\Theta] (deg \ast cm^2 \ast dmol^-1)$"
                else:
                    d = self.circular_dichroism[i][:, 1]
                    y_label = r"$\Delta A \ast 32.986$"
                
                w = self.circular_dichroism[i][:, 0]  # wavelengths
                
                if self.solvent[i] == 'T':  # color
                    col = 'r'
                else:
                    col = 'b'
                
                # plotting
                fig, ax = plt.subplots()
                line = ax.plot(w, d)
                plt.setp(line, color=col, linewidth=2.0)
                ax.set_xlabel('Wavelength (nm)', fontsize=16)
                ax.set_ylabel(y_label, fontsize=16)
                plt.title(splitext(f)[0], fontsize=18, fontweight='bold')
                img_name = splitext(f)[0] + '.pdf'
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                
                plt.savefig(join(self.directory, 'PDF', img_name), dpi=150)
                
        else:
            print("Wrong data option given!\nAvailable:")
            print("['mean residue ellipticity', 'molar ellipticity', 'circular dichroism']")

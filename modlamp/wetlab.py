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

import numpy as np
from os import listdir
from os.path import join

from modlamp.descriptors import GlobalDescriptor


__author__ = "modlab"
__docformat__ = "restructuredtext en"


class CD:
    """
    Class to handle circular dichroism data files and calculate several ellipticity and helicity values.
    The class can only handle data files of the *Applied Photophysics* type.
    """
    
    def __init__(self, directory, wmin, wmax, amide=True):
        """Init method for class CD.
        
        :param directory: {str} directory containing all data files to be read. Files need a **.csv** ending
        :param wmin: {int} smalles wavelength measured
        :param wmax: {int} highest wavelength measured
        :param amide: {bool} specifies whether the sequences have amidated C-termini
        """
        
        # initialize attributes
        self.filenames = list()
        self.names = list()
        self.sequences = list()
        self.conc_umol = list()
        self.conc_mgml = list()
        self.meanres_mw = list()
        self.solvent = list()
        self.circular_dichroism = np.empty((1, 2))
        self.molar_ellipticity = np.empty((1, 2))
        self.meanres_ellipticity = np.empty((1, 2))
        self.directory = directory
        self.wmin = wmin
        self.wmax = wmax
        self.amide = amide
        
        files = listdir(directory)
        self.filenames = [filename for filename in files if filename.endswith('.csv')]  # get all .csv files in dir
        
    def read_header(self):
        """Method to read all file headers into the class attributes and calculate sequence dependant values.
        
        :return: headers in class attributes.
        """
        
        d = GlobalDescriptor('X')  # template
        
        # loop through all files in the directory
        for file in self.filenames:
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
            self.circular_dichroism = np.array(zip(wlengths, ellipts))
            
            # calculate MW and transform concentration to mg/ml
            d.sequences = [sequence]
            d.calculate_MW(amide=self.amide)
            self.conc_mgml.append(d.descriptor[0] * umol / 10**6)
            self.meanres_mw.append(d.descriptor[0] / (len(sequence) - 1))  # mean residue molecular weight (MW / n-1)
            
    
#    def calc_molar_ellipticity(self):
#        """Method to calculate the molar ellipticity for all loaded data in :py:attr:`circular_dichroism`.
#
#        :return: {numpy array} molar ellipticity in :py:attr:`molar_ellipticity`
#        """
#
#        for data in self.circular_dichroism:
            
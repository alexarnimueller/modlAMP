#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to calculate different peptide descriptors for a given sequences.fasta file and save them to two files.
"""

from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

pd = PeptideDescriptor('/Users/modlab/Desktop/sequences.fasta', 'eisenberg')
gd = GlobalDescriptor('/Users/modlab/Desktop/sequences.fasta')

# Peptide Descriptor (AA scales) Calculations
pd.calculate_global()  # calculate global Eisenberg hydrophobicity
pd.calculate_moment(append=True)  # calculate Eisenberg hydrophobic moment

pd.load_scale('gravy')  # load GRAVY scale
pd.calculate_global(append=True)  # calculate global GRAVY hydrophobicity
pd.calculate_moment(append=True)  # calculate GRAVY hydrophobic moment

pd.load_scale('z3')  # load old Z scale
pd.calculate_autocorr(1, append=True)  # calculate global Z scale (=window1 autocorrelation)

# save descriptor data to .csv file
col_names1 = 'ID,Sequence,H_Eisenberg,uH_Eisenberg,H_GRAVY,uH_GRAVY,Z3_1,Z3_2,Z3_3'
pd.save_descriptor('/Users/modlab/Desktop/descriptors_1.csv', header=col_names1)

# Global Descriptor Calculations
gd.length()
gd.boman_index(append=True)
gd.aromaticity(append=True)
gd.aliphatic_index(append=True)
gd.instability_index(append=True)
gd.calculate_charge(ph=7.4, amide=False, append=True)
gd.calculate_MW(amide=False, append=True)

# save descriptor data to .csv file
col_names2 = 'ID,Sequence,Length,BomanIndex,Aromaticity,AliphaticIndex,InstabilityIndex,Charge,MW'
gd.save_descriptor('/Users/modlab/Desktop/descriptors_2.csv', header=col_names2)

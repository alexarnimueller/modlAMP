#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to calculate different peptide descriptors for a given sequences.fasta file and save them to two files.
"""

from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

# Load sequence file into descriptor object
pepdesc = PeptideDescriptor('/path/to/sequences.fasta', 'Eisenberg')  # use Eisenberg consensus scale
globdesc = GlobalDescriptor('/path/to/sequences.fasta')

# --------------- Peptide Descriptor (AA scales) Calculations ---------------
pepdesc.calculate_global()  # calculate global Eisenberg hydrophobicity
pepdesc.calculate_moment(append=True)  # calculate Eisenberg hydrophobic moment

# load other AA scales
pepdesc.load_scale('gravy')  # load GRAVY scale
pepdesc.calculate_global(append=True)  # calculate global GRAVY hydrophobicity
pepdesc.calculate_moment(append=True)  # calculate GRAVY hydrophobic moment
pepdesc.load_scale('z3')  # load old Z scale
pepdesc.calculate_autocorr(1, append=True)  # calculate global Z scale (=window1 autocorrelation)

# save descriptor data to .csv file
col_names1 = 'ID,Sequence,H_Eisenberg,uH_Eisenberg,H_GRAVY,uH_GRAVY,Z3_1,Z3_2,Z3_3'
pepdesc.save_descriptor('/path/to/descriptors1.csv', header=col_names1)

# --------------- Global Descriptor Calculations ---------------
globdesc.length()  # sequence length
globdesc.boman_index(append=True)  # Boman index
globdesc.aromaticity(append=True)  # global aromaticity
globdesc.aliphatic_index(append=True)  # aliphatic index
globdesc.instability_index(append=True)  # instability index
globdesc.calculate_charge(ph=7.4, amide=False, append=True)  # net charge
globdesc.calculate_MW(amide=False, append=True)  # molecular weight

# save descriptor data to .csv file
col_names2 = 'ID,Sequence,Length,BomanIndex,Aromaticity,AliphaticIndex,InstabilityIndex,Charge,MW'
globdesc.save_descriptor('/path/to/descriptors2.csv', header=col_names2)

"""
.. module:: oblique

.. moduleauthor:: modlab Alex Mueller <alex.mueller@pharma.ethz.ch>

This module incorporates methods for generating peptide sequences with a linear hydrophobicity gradient, meaning that
these sequences have a hydrophobic tail. This feature gives rise to the hypothesis that they orient themselves
tilted/oblique in membrane environment.
"""

import random
from core.templates import mutate_AA, aminoacids, template, clean, save_fasta, filter_unnatural
from itertools import cycle

__author__ = 'modlab'

class Oblique(object):
	"""
	Base class for oblique sequences with a so called linear hydrophobicity gradient.
	"""
	def __init__(self,lenmin,lenmax,seqnum):
		'''
		:param lenmin: minimal sequence length
		:param lenmax: maximal sequence length
		:param seqnum: number of sequences to generate
		:return: defined self variables
		'''
		aminoacids(self)
		template(self,lenmin,lenmax,seqnum)

	def generate_oblique(self):
		"""
		Method to generate the possible oblique sequences.
		:return: A list of sequences in self.sequences
		:Example:

		>>> O = Oblique(10,30,4)
		>>> O.generate_oblique()
		>>> O.sequences
		['GLLKVIRIAAKVLKVAVLVGIIAI','AIGKAGRLALKVIKVVIKVALILLAAVA','KILRAAARVIKGGIKAIVIL','VRLVKAIGKLLRIILRLARLAVGGILA']
		"""
		clean(self)
		for s in range(self.seqnum): #for the number of sequences to generate
			seq = ['X'] * random.choice(range(self.lenmin,self.lenmax + 1))
			basepos = random.choice(range(4)) #select spot for first basic residue from 0 to 3
			seq[basepos] = random.choice(self.AA_basic) #place first basic residue
			gap = cycle([3,4]).next #gap cycle of 3 & 4 --> 3,4,3,4,3,4...
			g = gap()
			while g+basepos < len(seq): #place more basic residues 3-4 positions further (changing between distance 3 and 4)
				basepos += g
				seq[basepos] = random.choice(self.AA_basic) #place more basic residues
				g = gap() #next gap

			for p in range(len(seq)):
				while seq[p] == 'X': #fill up remaining spots with hydrophobic AAs
					seq[p] = random.choice(self.AA_hyd)

			for e in range(1,len(seq)/3): # transform last 3rd of sequence into hydrophobic ones --> hydrophobicity gradient = oblique
				seq[-e] = random.choice(self.AA_hyd)

			self.sequences.append(''.join(seq))


	def mutate_AA(self,nr,prob):
		"""
		Method to mutate with **prob** probability a **nr** of positions per sequence randomly.

		:param nr: number of mutations to perform per sequence
		:param prob: probability of mutating a sequence
		:return: In *self.sequences*: mutated sequences
		:Example:

		>>> H.sequences
		['IAKAGRAIIK']
		>>> H.mutate_AA(3,1)
		>>> H.sequences
		['NAKAGRAWIK']
		"""
		mutate_AA(self,nr,prob)


	def save_fasta(self,filename):
		"""
		Method for saving sequences in the instance self.sequences to a file in FASTA format.

		:param filename: output filename (ending .fasta)
		:return: a FASTA formatted file containing the generated sequences
		"""
		save_fasta(self,filename)


	def filter_unnatrual(self):
		"""
		Method to filter out sequences with unnatural amino acids from :py:attr:`self.sequences` as well as duplicates.
		:return: Filtered sequence list in :py:attr:`self.sequences`
		"""
		filter_unnatural(self)
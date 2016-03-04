"""
.. module:: random_seq

.. moduleauthor:: modlab Alex Mueller <alex.mueller@pharma.ethz.ch>

This module incorporates methods for generating peptide random peptide sequences of defined length.
The amino acid probabilities can be chosen from different probabilities:

- **rand**: equal probabilities for all amino acids
- **AMP**: amino acid probabilities taken from the antimicrobial peptide database `APD2 <http://aps.unmc.edu/AP/main.php>`_.
- **AMPnoCM**: same amino acid probabilities as **AMP** but lacking Cys and Met (for synthesizability)

"""

import random
import numpy as np
import os
from core import mutate_AA, aminoacids, template, clean

__author__ = 'modlab'

class RandomSeqs:
	'''
	Class for random peptide sequences
	'''

	def __init__(self,lenmin,lenmax,seqnum):
		'''
		:param lenmin: minimal sequence length
		:param lenmax: maximal sequence length
		:param seqnum: number of sequences to generate
		:return: defined variables
		'''
		aminoacids(self)
		template(self,lenmin,lenmax,seqnum)


	def generate_sequences(self,proba='rand'):
		'''
		:param proba: AA probability to be used to generate sequences. Available: AMP, AMPnoCM, rand
		:return: A list of random AMP sequences with defined AA probabilities
		:Example:

		>>> R = RandomSeqs(5,20,6)
		>>> R.generate_sequences(proba='AMP')
		>>> R.sequences
		['CYGALWHIFV','NIVRHHAPSTVIK','LCPNPILGIV','TAVVRGKESLTP','GTGSVCKNSCRGRFGIIAF','VIIGPSYGDAEYA']
		'''
		clean(self)
		self.prob = self.prob_rand # default probability = rand
		if proba == 'AMPnoCM':
			self.prob = self.prob_AMPnoCM
		elif proba == 'AMP':
			self.prob = self.prob_AMP

		for s in range(self.seqnum):
			self.seq = []
			for l in range(random.choice(range(self.lenmin, self.lenmax+1))):
				self.seq.append(np.random.choice(self.AAs,p=self.prob)) #weighed random selection of amino acid, probabilities = prob
			self.sequences.append(''.join(self.seq))


	def save_fasta(self,filename):
		'''
		:param filename: output filename in which the sequences are safed in fasta format.
		:return: a fasta file containing the generated sequences
		'''
		if os.path.exists(filename):
			os.remove(filename) #remove outputfile, it it exists
		o = open(filename, 'a')

		for n in range(len(self.sequences)):
			print >> o, '>Seq_' + str(n)
			print >> o, self.sequences[n]
		o.close()


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
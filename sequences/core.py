"""
.. module:: core

.. moduleauthor:: modlab Alex Mueller <alex.mueller@pharma.ethz.ch>

Core helper functions for all sequence modules.
"""

import numpy as np
import os
import difflib
import random
import re

random.seed(8)

def save_fasta(self,filename):
	'''
	Method for saving sequences in the instance self.sequences to a file in FASTA format.

	:param filename: output filename (ending .fasta)
	:return: a FASTA formatted file containing the generated sequences
	'''
	if os.path.exists(filename):
		os.remove(filename) #remove outputfile, it it exists
	with open(filename,'w') as o:
		for n, seq in enumerate(self.sequences):
			print >> o, '>Seq_' + str(n)
			print >> o, seq


def mutate_AA(self,nr,prob):
	'''
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
	'''
	for s in range(len(self.sequences)):
		mutate = np.random.choice([1,0],1,p=[prob,1-prob]) #mutate: yes or no? probability = given mutation probability
		if mutate == 1:
			seq = list(self.sequences[s])
			cnt = 0
			while cnt < nr: #mutate "nr" AA
				seq[random.choice(range(len(seq)))] = random.choice(self.AAs)
				cnt += 1
			self.sequences[s] = ''.join(seq)


def aminoacids(self):
	"""
	Method used by all classes in :mod:`sequences` to generate templates for all needed instances.

	:return: all needed instances of the classes in this package
	"""
	self.sequences = []
	# AA classes
	self.AA_hyd = ['G','A','L','I','V']
	self.AA_basic = ['K','R']
	self.AA_anchor = ['W','Y','F']
	# AA labels
	self.AAs = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
	# AA probability from the APD2 database
	self.prob_AMP = [0.07693,0.05413,0.03873,0.02583,0.07133,0.0253,0.02633,0.11772,0.02093,0.06032,0.08423,0.09622,0.01233,0.04062,0.04613,0.06122,0.04493,0.01532,0.02423,0.05722]
	# AA probability from the APD2 database without Cys and Met (synthesis reasons)
	self.prob_AMPnoCM = [0.083953554,0.05907196,0.042265971,0.028188227,0,0.027609839,0.028733876,0.128467599,0.022840867,0.065827095,0.09192003,0.105004693,0,0.044328524,0.050341576,0.066809263,0.049032019,0.016718685,0.02644215,0.062444071]
	# equal AA probabilities
	self.prob_rand = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]


def template(self,lenmin,lenmax,seqnum):
	"""
	Method used by different classes in :mod:`sequences` to generate length and number templates for all needed instances.

	:param lenmin: minimal length of the generated sequences
	:param lenmax: maximal length of the generated sequences
	:param seqnum: number of sequences to generate
	:return: all needed instances (involving numbers and lengths) of the classes in this package
	"""
	self.lenmin = int(lenmin)
	self.lenmax = int(lenmax)
	self.seqnum = int(seqnum)


def clean(self):
	"""
	Method to clean the attributes **sequences**, **names** and **descriptor**.

	:return:
	"""
	self.names = []
	self.sequences = []
	self.descriptor = []


def filter_similarity(self, threshold=0.8):
	"""
	Method to filter out peptide sequences above a given similarity threshold in a list of given sequences. The list
	is first shuffled and all duplicates are removed. Then, the function iterates through the list only keeps sequences
	that have a lower similarity to the other list members than the given threshold.

	:param lst: list of sequences to be filtered for internal similarity
	:param threshold: similarity threshold over which one of similar members of **seq_list** gets kicked out
	:return: filtered list of sequences
	"""
	self.sequences = [x for x in set(self.sequences)] # remove duplicates
	random.shuffle(self.sequences)

	lst = []

	for s in self.sequences:
		for l in self.sequences:
			seq=difflib.SequenceMatcher(None,s,l)
			if seq.ratio() < threshold: # and seq.ratio() != 1.: #if higher than threshold but not self
				lst.append(s)

	self.sequences = [x for x in set(lst)] # return unique list of filtered sequences


def filter_unnatural(self):
	"""
	Method to filter out sequences with non-proteinogenic amino acids [B,J,O,U,X,Z]. Dublicates are removed as well.

	:param seq_list: List of sequences to be filtered.
	:return: Filtered list.
	"""
	seq_list = [x for x in set(self.sequences)] # remove duplicates
	pattern = re.compile('|'.join(['B','J','O','U','X','Z']))

	lst = []

	for s in seq_list:
		if not pattern.search(s):
			lst.append(s)

	self.sequences = lst
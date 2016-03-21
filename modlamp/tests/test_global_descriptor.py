import os
import sys
import unittest
from modlamp.descriptors import GlobalDescriptor

sys.path.insert(0, os.path.abspath('..'))

__author__ = 'modlab'


class TestGlobalDescriptor(unittest.TestCase):

	G = GlobalDescriptor(['GLFDIVKKVVGALG'])

	def test_charge(self):
		self.G.calculate_charge()
		self.assertAlmostEqual(self.G.descriptor[0], 1.00099999)

	def test_isoelectric(self):
		self.G.isoelectric_point()
		self.assertAlmostEqual(self.G.descriptor[0], 8.59100341)

	def test_molweight(self):
		self.G.calculate_MW()
		self.assertAlmostEqual(self.G.descriptor[0], 1415.71869999)

	def test_length(self):
		self.G.length()
		self.assertEqual(self.G.descriptor[0], 14)

	def test_charge_density(self):
		self.G.charge_density()
		self.assertAlmostEqual(self.G.descriptor[0], 0.00070706)

	def test_instability_index(self):
		self.G.instability_index()
		self.assertAlmostEqual(self.G.descriptor[0], -8.21428571)

	def test_aliphatic_index(self):
		self.G.aliphatic_index()
		self.assertAlmostEqual(self.G.descriptor[0], 152.85714286)

	def test_boman_index(self):
		self.G.boman_index()
		self.assertAlmostEqual(self.G.descriptor[0], -1.04785714)

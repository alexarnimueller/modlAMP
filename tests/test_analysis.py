import unittest

import numpy as np
import pandas as pd
from modlamp.analysis import GlobalAnalysis
from modlamp.core import read_fasta
from os.path import dirname, join


class TestAnalysis(unittest.TestCase):
    fname = join(dirname(__file__), 'files/plots/testplot.png')
    sequences, names = read_fasta(join(dirname(__file__), 'files/lib.fasta'))
    a = GlobalAnalysis(sequences)
    s1 = sequences[:10]
    s2 = sequences[10:20]
    b = GlobalAnalysis([s1, s2])
    sequences = np.array(sequences)
    c = GlobalAnalysis(sequences)
    sequences = pd.DataFrame(sequences)
    d = GlobalAnalysis(sequences)
    sequences = sequences.T
    e = GlobalAnalysis(sequences, names=['Lib1'])

    def test_input(self):
        self.assertEqual(self.a.library[0, 4], 'ALHAHASF')
        self.assertEqual(self.b.library[0, 4], 'ALHAHASF')
        self.assertEqual(self.c.library[0, 4], 'ALHAHASF')
        self.assertEqual(self.d.library[0, 4], 'ALHAHASF')
        self.assertEqual(self.e.library[0, 4], 'ALHAHASF')
    
    def test_libshape(self):
        self.assertEqual(self.a.library.shape[1], 246)
        
    def test_aa_freq(self):
        self.a.calc_aa_freq(plot=False)
        self.assertAlmostEqual(self.a.aafreq[0, 0], 0.08250071)
    
    def test_H(self):
        self.a.calc_H()
        self.assertAlmostEqual(self.a.H[0][1], 2.54615385e-01)
    
    def test_uH(self):
        self.a.calc_uH()
        self.assertAlmostEqual(self.a.uH[0][5], 0.6569639743)

    def test_charge(self):
        self.a.calc_charge()
        self.assertAlmostEqual(self.a.charge[0][2], 5.995, places=3)

    def test_len(self):
        self.a.calc_len()
        self.assertEqual(self.a.len[0][2], 24.)

    def test_summary(self):
        self.e.plot_summary(plot=False, filename=self.fname)


if __name__ == '__main__':
    unittest.main()

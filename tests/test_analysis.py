import unittest

from modlamp.analysis import GlobalAnalysis
from modlamp.core import read_fasta
from os.path import dirname, join


class TestAnalysis(unittest.TestCase):
    sequences, _ = read_fasta(join(dirname(__file__), 'files/lib.fasta'))
    a = GlobalAnalysis([sequences])
    
    def test_libshape(self):
        self.assertEqual(self.a.library.shape[1], 246)
        
    def test_aa_freq(self):
        self.a.calc_aa_freq(plot=False)
        self.assertAlmostEqual(self.a.aafreq[0, 0], 0.08250071)
    
    def test_H(self):
        self.a.calc_H()
        self.assertAlmostEquals(self.a.H[0][1], 2.54615385e-01)
    
    def test_uH(self):
        self.a.calc_uH()
        self.assertAlmostEqual(self.a.uH[0][5], 0.6569639743)

    def test_charge(self):
        self.a.calc_charge()
        self.assertAlmostEqual(self.a.charge[0][2], 5.995, places=3)

    def test_len(self):
        self.a.calc_len()
        self.assertEqual(self.a.len[0][2], 24.)

if __name__ == '__main__':
    unittest.main()

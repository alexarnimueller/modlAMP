import unittest
from ..analysis import *
from ..core import read_fasta
from os.path import abspath, join


class TestAnalysis(unittest.TestCase):
    sequences, names = read_fasta(join(abspath('.'), 'modlamp/tests/files/lib.fasta'))
    
    def test_there(self):
        self.assertEqual(self.sequences, self.sequences)
        

if __name__ == '__main__':
    unittest.main()

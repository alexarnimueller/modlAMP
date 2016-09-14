import unittest
from numpy.random import randint
from ..sequences import AMPngrams


class TestCentrosymmetric(unittest.TestCase):
    S = AMPngrams(10, n_min=2, n_max=10)
    S.generate_sequences()

    def test_seqnum(self):
        self.assertEqual(len(self.S.sequences), 10)
    
    def test_ngrams_in_seq(self):
        seq_str = ''.join(self.S.sequences)
        ngram_str = ''.join(self.S.ngrams.tolist())
        pos = randint(0, len(seq_str) -2)
        self.assertTrue(seq_str[pos:pos+2] in ngram_str)

if __name__ == '__main__':
    unittest.main()

import unittest
from modlamp.sequences import AmphipathicArc


class TestAmphipathicArc(unittest.TestCase):
    S = AmphipathicArc(10, 8, 27)
    S.generate_sequences(arcsize=100)
    
    def test_seq_num(self):
        self.assertEqual(len(self.S.sequences), 10)
    
    def test_seq_len(self):
        for seq in self.S.sequences:
            self.assertIn(len(seq), range(8, 28))
    
    def test_seq_arc(self):
        for seq in self.S.sequences:
            self.assertTrue(any(s in seq[0] for s in ('A', 'D', 'E', 'G', 'H', 'K', 'N', 'P', 'Q', 'R', 'S', 'T', 'Y')))
            self.assertTrue(any(s in seq[1] for s in ('F', 'I', 'L', 'V', 'W', 'Y')))
    
    def test_make_H(self):
        apa = AmphipathicArc(10, 8, 27)
        apa.generate_sequences(arcsize=100)
        apa.make_H_gradient()
        for seq in apa.sequences:
            for a in range(1, int(len(seq) / 3 + 1)):
                self.assertTrue(seq[-a] in ('F', 'I', 'L', 'V', 'W', 'Y'))


if __name__ == '__main__':
    unittest.main()

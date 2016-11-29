import unittest

from modlamp.sequences import Oblique


class TestOblique(unittest.TestCase):
    S = Oblique(10, 10, 30)
    S.generate_sequences()
    
    def test_seq_num(self):
        self.assertEqual(len(self.S.sequences), 10)
    
    def test_seq_len(self):
        for seq in self.S.sequences:
            self.assertIn(len(seq), range(10, 31))
    
    def test_gradient(self):
        for seq in self.S.sequences:
            for a in range(1, len(seq) / 3):
                self.assertIn(seq[-a], self.S.AA_hyd)


if __name__ == '__main__':
    unittest.main()

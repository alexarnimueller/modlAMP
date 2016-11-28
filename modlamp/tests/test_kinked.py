import unittest

from ..sequences import Kinked


class TestKink(unittest.TestCase):
    K = Kinked(1, 10, 40)
    K.generate_kinked()
    
    def test_kink_length(self):
        self.assertIn(len(self.K.sequences[0]), range(10, 41))
    
    def test_first_placement(self):
        self.assertTrue(any(s in self.K.sequences[0][:4] for s in ('K', 'R', 'P')))


if __name__ == '__main__':
    unittest.main()

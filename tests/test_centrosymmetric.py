# -*- coding: utf-8 -*-
import unittest

from modlamp.sequences import Centrosymmetric

# enough draws that a per-sequence failure probability of ~0.7 % is caught essentially every run:
# the pre-fix implementation repeated a block in about 1 in 150 sequences
REPLICATES = 2000


def blocks(seq):
    """Split a generated sequence into its seven-residue blocks."""
    return [seq[i : i + 7] for i in range(0, len(seq), 7)]


class TestCentrosymmetric(unittest.TestCase):
    S = Centrosymmetric(REPLICATES)
    S.generate_sequences(symmetry="symmetric")

    def test_block_symmetry(self):
        # every block is a palindrome of the form [h, +, h, a, h, +, h]
        for seq in self.S.sequences:
            for b in blocks(seq):
                self.assertEqual(b[0], b[6])
                self.assertEqual(b[1], b[5])
                self.assertEqual(b[2], b[4])

    def test_whole_symmetry(self):
        # symmetric mode repeats one single block
        for seq in self.S.sequences:
            self.assertEqual(len(set(blocks(seq))), 1)

    def test_length(self):
        for seq in self.S.sequences:
            self.assertIn(len(seq), (14, 21))


class TestCentroAsymmetric(unittest.TestCase):
    AS = Centrosymmetric(REPLICATES)
    AS.generate_sequences(symmetry="asymmetric")

    def test_blocks_always_differ(self):
        # the class documents asymmetric mode as concatenating *different* blocks. Drawing each
        # block independently is not enough: only 150 distinct blocks exist, so collisions happened
        # in ~0.7 % of sequences and asymmetric mode silently returned a symmetric sequence.
        for seq in self.AS.sequences:
            b = blocks(seq)
            self.assertEqual(len(set(b)), len(b), "repeated block in asymmetric sequence %s" % seq)

    def test_block_symmetry(self):
        # asymmetric across blocks, still centro-symmetric within each block
        for seq in self.AS.sequences:
            for b in blocks(seq):
                self.assertEqual(b[0], b[6])
                self.assertEqual(b[1], b[5])
                self.assertEqual(b[2], b[4])

    def test_length(self):
        for seq in self.AS.sequences:
            self.assertIn(len(seq), (14, 21))


class TestCentrosymmetricArguments(unittest.TestCase):
    def test_unknown_symmetry_raises(self):
        s = Centrosymmetric(1)
        self.assertRaises(AttributeError, s.generate_sequences, symmetry="palindromic")


if __name__ == "__main__":
    unittest.main()

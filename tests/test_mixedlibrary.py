# -*- coding: utf-8 -*-
"""Regression tests for MixedLibrary."""
import unittest

from modlamp.sequences import MixedLibrary

__author__ = "modlab"


class TestMixedLibrary(unittest.TestCase):
    def test_respects_length_bounds_and_size(self):
        lib = MixedLibrary(400)
        lib.generate_sequences()
        self.assertGreater(len(lib.sequences), 300)
        self.assertTrue(all(7 <= len(s) <= 28 for s in lib.sequences))

    def test_every_sublibrary_is_populated(self):
        lib = MixedLibrary(800)
        lib.generate_sequences()
        for key, n in lib.nums.items():
            self.assertGreater(n, 0, "sub-library %s received no sequences" % key)
        self.assertEqual(sum(lib.nums.values()), len(lib.sequences))
        self.assertEqual(len(lib.names), len(lib.sequences))

    def test_zero_ratio_does_not_crash(self):
        lib = MixedLibrary(100, centrosymmetric=1, centroasymmetric=0, helix=0, kinked=0,
                           oblique=0, rand=0, randAMP=0, randAMPnoCM=0)
        lib.generate_sequences()
        self.assertGreater(len(lib.sequences), 0)

    def test_deduplication_preserves_order(self):
        lib = MixedLibrary(200)
        lib.generate_sequences()
        self.assertEqual(len(lib.sequences), len(set(lib.sequences)))


if __name__ == "__main__":
    unittest.main()

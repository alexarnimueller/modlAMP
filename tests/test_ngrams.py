# -*- coding: utf-8 -*-
"""Regression tests for n-gram counting."""
import unittest

from modlamp.core import count_ngrams

__author__ = "modlab"


class TestNgramCounts(unittest.TestCase):
    def test_overlapping_occurrences_are_counted(self):
        self.assertEqual(count_ngrams("AAAA", 2)["AA"], 3)
        self.assertEqual(count_ngrams("KKKKK", 3)["KKK"], 3)

    def test_non_repeating_sequence_is_unchanged(self):
        self.assertEqual(count_ngrams("ACDEF", 2), {"AC": 1, "CD": 1, "DE": 1, "EF": 1})

    def test_total_matches_number_of_windows(self):
        seq = "GLLDFLSLAALSLDKLVKKGALS"
        self.assertEqual(sum(count_ngrams(seq, 3).values()), len(seq) - 2)


if __name__ == "__main__":
    unittest.main()

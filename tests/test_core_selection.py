# -*- coding: utf-8 -*-
"""Regression tests for sequence selection and de-duplication."""
import unittest

from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

__author__ = "modlab"

SEQS = ["GLFDIVKKVVGALGSL", "KLLKLLKKLLKLLK", "ACDEFGHIK", "RRWWRRWRR", "GGGGGGGG", "AAKKAAKK"]


class TestMinMaxSelection(unittest.TestCase):
    def _instance(self):
        d = PeptideDescriptor(SEQS, "eisenberg")
        d.calculate_global()
        return d

    def test_returns_requested_number_of_real_sequences(self):
        d = self._instance()
        d.minmax_selection(3)
        self.assertEqual(len(d.sequences), 3)
        self.assertEqual(len(set(d.sequences)), 3)
        for s in d.sequences:
            self.assertIn(s, SEQS)

    def test_descriptor_rows_follow_the_selection(self):
        d = self._instance()
        d.minmax_selection(3)
        self.assertEqual(d.descriptor.shape[0], 3)

    def test_is_reproducible_for_a_given_seed(self):
        out = []
        for _ in range(2):
            d = self._instance()
            d.minmax_selection(3, seed=7)
            out.append(d.sequences)
        self.assertEqual(out[0], out[1])

    def test_handles_duplicate_descriptor_rows(self):
        d = PeptideDescriptor(["AAAA", "AAAA", "KKKK", "LLLL"], "eisenberg")
        d.calculate_global()
        d.minmax_selection(2)
        self.assertEqual(len(d.sequences), 2)


class TestFilterDuplicates(unittest.TestCase):
    def test_works_after_descriptor_calculation(self):
        d = GlobalDescriptor(["KLLKLLKKLLKLLK", "KLLKLLKKLLKLLK", "GLFDIVKKVVGALG"])
        d.calculate_charge()
        d.filter_duplicates()
        self.assertEqual(d.sequences, ["KLLKLLKKLLKLLK", "GLFDIVKKVVGALG"])
        self.assertEqual(d.descriptor.shape[0], 2)

    def test_works_on_a_multi_column_descriptor(self):
        d = GlobalDescriptor(["KLLKLLKKLLKLLK", "KLLKLLKKLLKLLK", "GLFDIVKKVVGALG"])
        d.calculate_all()
        cols = d.descriptor.shape[1]
        d.filter_duplicates()
        self.assertEqual(d.descriptor.shape, (2, cols))

    def test_works_without_descriptors(self):
        d = GlobalDescriptor(["AAAA", "AAAA", "CCCC"])
        d.filter_duplicates()
        self.assertEqual(d.sequences, ["AAAA", "CCCC"])


if __name__ == "__main__":
    unittest.main()

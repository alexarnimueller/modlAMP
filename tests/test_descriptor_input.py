# -*- coding: utf-8 -*-
"""Regression tests for BaseDescriptor input handling."""
import unittest
from os.path import dirname, join

import numpy as np

from modlamp.descriptors import GlobalDescriptor

__author__ = "modlab"


class TestDescriptorInput(unittest.TestCase):
    def test_lowercase_sequences_are_accepted(self):
        d = GlobalDescriptor(["glfdivkkvvgalg"])
        self.assertEqual(d.sequences, ["GLFDIVKKVVGALG"])

    def test_single_lowercase_string(self):
        d = GlobalDescriptor("glfdivkkvvgalg")
        self.assertEqual(d.sequences, ["GLFDIVKKVVGALG"])

    def test_uppercase_list_is_unchanged(self):
        d = GlobalDescriptor(["GLFDIVKKVVGALG", "KKLLKKLLKK"])
        self.assertEqual(d.sequences, ["GLFDIVKKVVGALG", "KKLLKKLLKK"])

    def test_ndarray_input(self):
        d = GlobalDescriptor(np.array(["GLFDIVKKVVGALG", "KKLLKKLLKK"]))
        self.assertEqual(len(d.sequences), 2)

    def test_fasta_file_still_loads(self):
        d = GlobalDescriptor(join(dirname(__file__), "files", "lib.fasta"))
        self.assertGreater(len(d.sequences), 0)
        self.assertEqual(len(d.sequences), len(d.names))

    def test_empty_input_raises(self):
        self.assertRaises(ValueError, GlobalDescriptor, [])

    def test_nonexistent_path_raises(self):
        self.assertRaises(ValueError, GlobalDescriptor, "/no/such/file.fasta")

    def test_unsupported_file_type_raises(self):
        self.assertRaises(ValueError, GlobalDescriptor, join(dirname(__file__), "..", "README.rst"))


if __name__ == "__main__":
    unittest.main()

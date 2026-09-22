# -*- coding: utf-8 -*-
"""Regression tests for the FASTA reader."""
import os
import tempfile
import unittest

from modlamp.core import read_fasta

__author__ = "modlab"


class TestReadFasta(unittest.TestCase):
    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".fasta")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_repeated_last_line_is_not_duplicated(self):
        p = self._write(">s1\nKLLKLLKKLLKLLK\n>s2\nGLFDIVKKVVGALG\n>s3\nKLLKLLKKLLKLLK\n")
        seqs, names = read_fasta(p)
        self.assertEqual(seqs, ["KLLKLLKKLLKLLK", "GLFDIVKKVVGALG", "KLLKLLKKLLKLLK"])
        self.assertEqual(names, ["s1", "s2", "s3"])

    def test_sequences_and_names_stay_aligned(self):
        p = self._write(">a description one\nAAAA\n>b description two\nCCCC\n")
        seqs, names = read_fasta(p)
        self.assertEqual(len(seqs), len(names))
        self.assertEqual(names, ["a", "b"])

    def test_tab_separated_header(self):
        p = self._write(">a\tdescription\nAAAA\n")
        _, names = read_fasta(p)
        self.assertEqual(names, ["a"])

    def test_wrapped_sequence(self):
        p = self._write(">s1\nKLLKLLK\nGLFDIVK\n>s2\nAAAA\n")
        seqs, _ = read_fasta(p)
        self.assertEqual(seqs, ["KLLKLLKGLFDIVK", "AAAA"])

    def test_blank_lines_and_missing_trailing_newline(self):
        p = self._write(">s1\nKLLKLLK\n\n>s2\nAAAA")
        seqs, _ = read_fasta(p)
        self.assertEqual(seqs, ["KLLKLLK", "AAAA"])


if __name__ == "__main__":
    unittest.main()

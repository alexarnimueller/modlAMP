# -*- coding: utf-8 -*-
"""The database module must import without a MySQL driver installed."""
import unittest

__author__ = "modlab"


class TestDatabaseImport(unittest.TestCase):
    def test_module_imports_without_mysql(self):
        import modlamp.database as db

        self.assertTrue(hasattr(db, "query_apd"))
        self.assertTrue(hasattr(db, "query_camp"))
        self.assertTrue(hasattr(db, "query_database"))

    def test_version_is_single_sourced(self):
        import modlamp
        from modlamp.version import __version__

        self.assertEqual(modlamp.__version__, __version__)


if __name__ == "__main__":
    unittest.main()

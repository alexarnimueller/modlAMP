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

    def test_version_is_consistent(self):
        import re
        from os.path import dirname, join

        import modlamp

        setup_py = open(join(dirname(modlamp.__file__), "..", "setup.py")).read()
        m = re.search(r'version="([^"]+)"', setup_py)
        if m:  # only checkable from a source checkout
            self.assertEqual(m.group(1), modlamp.__version__)


if __name__ == "__main__":
    unittest.main()

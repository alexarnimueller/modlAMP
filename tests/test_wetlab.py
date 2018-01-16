import unittest
from os.path import dirname, join

from modlamp.wetlab import CD


class TestCD(unittest.TestCase):
    cd = CD(join(dirname(__file__), 'files'), 180, 260, amide=True)
    
    def test_init(self):
        self.assertIsNotNone(self.cd.filenames)
        self.assertIsInstance(self.cd.filenames[0], basestring)
    
    def test_read_header(self):
        self.assertEqual(self.cd.sequences[1], 'KLLKLLKKLVGALG')
        self.assertEqual(self.cd.sequences[0], 'GLFDIVKKVLKLLK')
        self.assertEqual(self.cd.conc_umol[1], 33.)
        self.assertAlmostEqual(self.cd.conc_mgml[1], 0.04926938, 5)
        self.assertAlmostEqual(self.cd.meanres_mw[1], 114.84538, 4)
    
    def test_molar_ellipticity(self):
        self.cd.calc_molar_ellipticity()
        self.assertAlmostEqual(self.cd.molar_ellipticity[1][0, 1], -1172.7787878787878, 5)
    
    def test_meanres_ellipticity(self):
        self.cd.calc_meanres_ellipticity()
        self.assertAlmostEqual(self.cd.meanres_ellipticity[1][38, 1], -1990.3473193473196, 5)
    
    def test_helicity(self):
        self.cd.calc_meanres_ellipticity()
        self.cd.helicity()
        self.assertEqual(float(self.cd.helicity_values.iloc[0]['Helicity']), 79.68)


if __name__ == '__main__':
    unittest.main()

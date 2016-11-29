# -*- coding: utf-8 -*-

from setuptools import setup

with open('README') as f:
    readme = f.read()

with open('LICENSE') as f:
    lic = f.read()

setup(
        name='modlAMP',
        version='2.6.0',
        description='modlabs peptide package with for in silico peptide QSAR studies',
        long_description=readme,
        author='modlab, Alex Müller, Gisela Gabernet',
        author_email='alex.mueller@pharma.ethz.ch',
        url='https://www.cadd.ethz.ch/software/modlamp.html',
        license=lic,
        keywords="antimicrobial peptide descriptor sequences QSAR machine learning design",
        packages=['modlamp', 'tests'],
        package_data={'modlamp': ['data/*.csv', 'data/*.fasta']},
        scripts=['bin/example_modlamp.py'],
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'License :: OSI Approved :: BSD-3',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7'],
        install_requires=['setuptools>=20.2.2',
                          'nose>=1.3.7',
                          'sphinx>=1.3.5',
                          'numpy>=1.10.4',
                          'scipy>=0.17.0',
                          'biopython>=1.66',
                          'matplotlib>=1.5.1',
                          'scikit-learn>=0.17.1',
                          'pandas>=0.18.1',
                          'requests>=2.11.1',
                          'lxml>=3.6.4']
)

# -*- coding: utf-8 -*-

from setuptools import setup


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='modlAMP',
    version='2.1.0',
    description='modlabs peptide package with descriptors, sequences, ML etc. for peptide QSAR studies',
    long_description=readme,
    author='modlab, Alex Müller',
    author_email='alex.mueller@pharma.ethz.ch',
    url='https://www.cadd.ethz.ch',
    license=license,
    keywords = "antimicrobial peptide descriptor sequences QSAR machine learning",
    packages=['modlamp', 'modlamp.tests'],
    package_data = {'modlamp': ['data/*.csv','data/*.fasta']},
    scripts = ['bin/*.py'],
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',],
)
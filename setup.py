# -*- coding: utf-8 -*-

from setuptools import setup
from pip.req import parse_requirements

# parse the requirements from the requirements file
install_reqs = parse_requirements('requirements.txt', session='hack')
reqs = [str(ir.req) for ir in install_reqs]

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    lic = f.read()

setup(name='modlamp',
      version='2.6.0',
      description='modlabs peptide package with for in silico peptide QSAR studies',
      long_description=readme,
      author='modlab, Alex Müller, Gisela Gabernet',
      author_email='alex.mueller@pharma.ethz.ch',
      url='https://gitlab.com/arni/modlAMP',
      download_url='',
      license=lic,
      keywords="antimicrobial peptide descriptor sequences QSAR machine learning design",
      packages=['modlamp'],
      package_data={'modlamp': ['data/*.csv', 'data/*.fasta']},
      scripts=['bin/example_modlamp.py', 'bin/example_descriptors.py'],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7'],
      install_requires=install_reqs
      )

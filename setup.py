# -*- coding: utf-8 -*-

from setuptools import setup


with open('README.rst', 'r') as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    reqs = f.read().split('\n')

setup(name='modlamp',
      version='4.3.0',  # also change version in version.py
      description='python package for in silico peptide design and QSAR studies',
      long_description=readme,
      author='Alex Müller, Gisela Gabernet',
      author_email='alexarnimueller@protonmail.com',
      url='http://modlamp.org',
      license='BSD-3',
      keywords="antimicrobial anticancer peptide descriptor sequences QSAR machine learning design",
      packages=['modlamp'],
      package_data={'modlamp': ['data/*.csv', 'data/*.fasta']},
      scripts=['bin/example_modlamp.py', 'bin/example_descriptors.py'],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6'],
      install_requires=reqs
      )


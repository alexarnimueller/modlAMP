README
======

**modlAMP**

This is a python package that is designed for working with peptides, proteins or any amino acid sequence of natural amino acids. 
It incorporates several modules, like descriptor calculation (module **descriptors**) or sequence generation (module **sequences**).
For basic instructions how to use the package, see Usage_.


Installation
************

For the installation to work, ``pip`` needs to be installed. If you're not sure whether you already have pip, type
``pip --version``. If you don't have pip installed, install it via ``easy_install pip``.

When pip is installed, run the following command when located in the modlAMP package directory::

    sudo make
    sudo make build
    make doc

Usage
*****

After installation, you should be able to import and use the different modules like shown below:

>>> from sequences.centrosymmetric import CentroSequences
>>> from descriptors.peptide_descriptor import PeptideDescriptor
>>> S = CentroSequences(5)
>>> S.generate_asymmetric() # generate 5 asymmetric centrosymmetric sequences
>>> S.sequences
['VRVFVRVVKLYLKVVKGFGKV','ARAFARAVRGYGRV','VKIWIKVGRAFARG','LRIWIRLIKAYAKI','VKAYAKVVRLWLRV']

>>> D = PeptideDescriptor(S.sequences,'PPCALI') # describe the sequences by the auto-correlated PPCALI descriptor scale
>>> D.calculate_autocorr(7)
>>> D.save_descriptor('/Users/name/Desktop/descriptor.csv',delimiter=',')


Package Structure
*****************

The package structure looks as follows::

	__init__.py
	README.rst
	LICENSE
	setup.py
	requirements.txt

	descriptors/
		__init__.py
		peptide_descriptor.py
		scales/

	sequences/
		__init__.py
		centrosymmetric.py
		helix.py
		kinked.py
		oblique.py
		random_seq.py
	
	machinelearning/
		__init__.py

	plot/
		__init__.py

	tests/
		test_peptide_descriptor.py
		test_centrosymmetric.py
		test_helix.py
		test_kinked.py
		test_oblique.py
		test_random.py
		files/
	
	docs/
		_build/
			html/
				index.html


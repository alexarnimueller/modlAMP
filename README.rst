README
======

**modlamp**

This is a python package that is designed for working with peptides, proteins or any amino acid sequence of natural amino acids. 
It incorporates several modules, like descriptor calculation (module **descriptors**) or sequence generation (module **sequences**).
For basic instructions how to use the package, see Usage_ or this `example script <examplescript.html>`_.

.. warning::
    You are advised to install `Anaconda <https://www.continuum.io/downloads>`_ python package manager before
    installing modlAMP. It will take care of all necessairy requirements and versions.
        

.. note::
    If you are reading this on Gitlab, several links like the *example script* will not work. Please clone the
    repository to your local machine and consider the documentation in ``modlAMP/docs/build/html/index.html``.
    Use ``git clone git@gitlab.ethz.ch:CADD/modlAMP.git`` to clone modlAMP to your current working directory.


Installation
************

For the installation to work, ``pip`` needs to be installed. If you're not sure whether you already have pip, type
``pip --version``. If you don't have pip installed, install it via ``sudo easy_install pip``.

When pip is installed, run the following command when located in the modlAMP package directory::

    make
    sudo make install
    make doc

Usage
*****

For a detailed description of all modules see the documentation in ``modlAMP/docs/build/html/index.html``.

Importing modules
-----------------

After installation, you should be able to import and use the different modules like shown below:

>>> from modlamp.sequences import Centrosymmetric
>>> from modlamp.descriptors import PeptideDescriptor
>>> from modlamp.database import query_database


Generating Sequences
--------------------

The following example shows how to generate a library of 1000 sequences out of all available sequence generation methods:

>>> from modlamp.sequences import MixedLibrary
>>> Lib = MixedLibrary(1000)
>>> Lib.generate_library()
>>> Lib.sequences[:10]
['VIVRVLKIAA','VGAKALRGIGPVVK','QTGKAKIKLVKLRAGPYANGKLF','RLIKGALKLVRIVGPGLRVIVRGAR','DGQTNRFCGI','ILRVGKLAAKV',...]

These commands generated a mixed peptide library comprising of 1000 sequences.

.. note::
    If duplicates are present in :py:attr:`self.sequences`, these are removed during generation. Therefore it is possible
    that less than the specified sequences are obtained.

The module :mod:`sequences` incorporates different sequence generation classes. For documentation thereof, consider the
docs for `sequences <modlamp.html#module-modlamp.sequences>`_.


Calculating Descriptor Values
-----------------------------

Now, different descriptor values can be calculated for the generated sequences: (see `Generating Sequences`_)

>>> from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
>>> P = PeptideDescriptor(Lib.sequences,'eisenberg')
>>> P.calculate_moment()
>>> P.descriptor[:10]
array([[ 0.60138255],[ 0.61232763],[ 0.01474009],[ 0.72333858],[ 0.20390763],[ 0.68818279],...]
>>> G = GlobalDescriptor(Lib.sequences)
>>> G.isoelectric_point()
>>> G.descriptor[:10]
array([ 10.09735107,   8.75006104,  12.30743408,  11.26385498, ...]
>>> pepCATS = PeptideDescriptor('sequence/file/to/be/loaded.fasta', 'pepcats')
>>> pepCATS.calculate_crosscorr(7)
>>> pepCATS.descriptor
array([[ 0.6875    ,  0.46666667,  0.42857143,  0.61538462,  0.58333333,

We calculated the global hydrophobic moments from the Eisenberg hydrophobicity scale and the isoelectric points.
Many more descriptors can be calculated, from global descriptors to concoluted / correlated descriptors from different
amino acid scales. For further information consider the docs for `descriptors <modlamp.html#module-modlamp.descriptors>`_.


Plotting Features
-----------------

We can now plot these values as a boxplot, for example the hydrophobic moment:

>>> from modlamp.plot import plot_feature
>>> plot_feature(P.descriptor,y_label='uH Eisenberg')

.. image:: static/uH_Eisenberg.png
    :height: 300px

We can additionally compare these descriptor values to known AMP sequences. For that, we import sequences from the APD3, which
are stored in the FASTA formatted file ``APD3.fasta``:

>>> APD = PeptideDescriptor('/Path/to/file/APD3.fasta', 'eisenberg')
>>> APD.calculate_moment()

Now lets compare the values by plotting:

>>> plot_feature((P.descriptor, APD.descriptor), y_label='uH Eisenberg', x_tick_labels=['Library', 'APD3'])

.. image:: static/uH_APD3.png
    :height: 300px

It is also possible to plot 2 or 3 different features in a scatter plot:

:Example: **2D Scatter Plot**

>>> from modlamp.plot import plot_2_features
>>> A = PeptideDescriptor('/Path/to/file/class1&2.fasta', 'eisenberg')
>>> A.calculate_moment()
>>> B = GlobalDescriptor('/Path/to/file/class1&2.fasta')
>>> B.isoelectric_point()
>>> target = [1] * (len(A.sequences) / 2) + [2] * (len(A.sequences) / 2)
>>> plot_2_features(A.descriptor, B.descriptor, x_label='uH', y_label='pI', targets=target)

.. image:: static/2D_scatter.png
    :height: 300px

:Example: **3D Scatter Plot**

>>> from modlamp.plot import plot_3_features
>>> C = GlobalDescriptor('/Path/to/file/APD3.fasta')
>>> C.length()
>>> plot_3_features(A.descriptor, B.descriptor, C.descriptor, x_label='uH', y_label='pI', z_label='length')

.. image:: static/3D_scatter.png
    :height: 300px

Further plotting methods like **helical wheel plots** are available. See the documentation for the
`plot <modlamp.html#module-modlamp.plot>`_ module.


Database Connection
-------------------

Peptides from the two most prominent AMP databases `APD <http://aps.unmc.edu/AP/>`_ and `CAMP <http://camp.bicnirrh
.res.in/>`_ can be directly scraped with the :mod:`modlamp.database` module.

For downloading a set of sequences from the **APD** database, first get the IDs of the sequences you want to query
from the APD website. Then proceed as follows:

>>> query_apd([15, 16, 17, 18, 19, 20])  # download sequences with IDs 15
['GLFDIVKKVVGALGSL','GLFDIVKKVVGAIGSL','GLFDIVKKVVGTLAGL','GLFDIVKKVVGAFGSL','GLFDIAKKVIGVIGSL','GLFDIVKKIAGHIAGSI']

The same holds true for the **CAMP** database:

>>> query_camp([2705, 2706])
['GLFDIVKKVVGALGSL','GLFDIVKKVVGTLAGL']

modlamp also hosts a module for connecting to the modlab internal peptide database on our local server.
Peptide sequences included in any table in the peptides database can be downloaded directly in python.

.. warning::
    This module only works in the modlab intranet at ETH Zurich

For querying sequences from a given table, the sequences must be stored in a column called "sequences" in the mysql
table. The query then works as follows:

>>> from modlamp.database import query_database
>>> query_database('modlab_experiments', ['sequence'])
Password: >? ***********
Connecting to MySQL database...
connection established!
['ILDSSWQRTFLLS','IKLLHIF','ACFDDGLFRIIKFLLASDRFFT', ...]


Loading Prepared Datasets
-------------------------

For AMP QSAR models, different options exist of choosing negative / inactive peptide examples. We assembled several
datasets for classification tasks, that can be read by the :mod:`modlamp.datasets` module.

:Example: **Helical AMPs vs. random all helical peptides**

>>> from modlamp.datasets import load_helicalAMPset
>>> data = load_helicalAMPset()
>>> data.keys()
['target_names', 'target', 'feature_names', 'sequences']

The variable ``data`` holds **four different keys, which can also be called as its attributes**. The available
attributes for :py:func:`load_helicalAMPset()` are :py:attr:`target_names` (target names), :py:attr:`target` (the
target class vector), :py:attr:`feature_names` (the name of the data features, here: 'Sequence') and
:py:attr:`sequences` (the loaded sequences).

:Example:

>>> data.target_names
array(['HEL', 'AMP'], dtype='|S3')
>>> data.sequences[:5]
['FDQAQTEIQATMEEN', 'DVDAALHYLARLVEAG', 'RCPLVIDYLIDLATRS', 'NPATLMMFFK', 'NLEDSIQILRTD']


Analysing Wetlab Circular Dichroism Data
----------------------------------------

The modlule :mod:`modlamp.wetlab` includes the class :py:class:`modlamp.wetlab.CD` to analyse raw circular dichroism
data from wetlab experiments. The following example shows how to load a raw datafile and calculate secondary
structure contents:

>>> cd = CD('/path/to/your/folder', 185, 260)  # load all files in a specified folder
>>> cd.names  # peptide names read from the file headers
['Pep 10', 'Pep 10', 'Pep 11', 'Pep 11', ... ]
>>> cd.calc_meanres_ellipticity()  # calculate the mean residue ellipticity values
>>> cd.meanres_ellipticity
array([[   260.        ,   -266.95804196],
       [   259.        ,   -338.13286713],
       [   258.        ,   -387.25174825], ...])
>>> cd.helicity(temperature=24., k=3.492185008, induction=True)  # calculate helical content
>>> cd.helicity_values
               Name     Solvent  Helicity  Induction
            0  Aurein       T    100.0     3.823
            1  Aurein       W    26.16     0.000
            2  Klak         T    76.38     3.048
            3  Klak         W    25.06     0.000 ...

.. seealso:: :py:func:`modlamp.wetlab.CD.helicity()`

The read and calculated values can finally be plotted as follows:

>>> cd.plot(data='mean residue ellipticity', combine=True)

.. image:: static/cd1.png
    :height: 300px
.. image:: static/cd2.png
    :height: 300px
.. image:: static/cd3.png
    :height: 300px


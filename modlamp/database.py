"""
.. module:: modlamp.database

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to connect to the modlab internal peptide database server and query sequences from
the different database tables.
"""

from os.path import dirname
from os.path import join
import csv
import numpy as np
from getpass import getpass
import mysql.connector
from mysql.connector import Error

__author__ = 'modlab'


def _read_db_config(host='gsdelta641.ethz.ch',database='peptides',user='modlab',password=None):
	"""
	Read database configuration and return a dictionary object.
	This function generally does not need to be used as it is called by the function :func:`query_sequences` directly.

	:param host: The host name of your server hosting the database.
	:param database: Name of the database.
	:param user: Username
	:return: a dictionary of database parameters
	"""
	if password == None:
		password = getpass()

	db = {	'host' : host,
			'database' : database,
			'user' : user,
			'password' : password
			}

	return db


def _connect():
	"""
	Connect to a given MySQL database (in config.ini file).
	This function is called by the function :func:`query_sequences`.

	:return: a mysql.connector connection object
	"""
	config = _read_db_config()

	try:
		print('Connecting to MySQL database...')
		conn = mysql.connector.connect(**config)
		if conn.is_connected():
			print('connection established!')
		else:
			print('connection failed!')

	except mysql.connector.Error as err:
		print(err)

	finally:
		if conn != None:
			return conn


def query_sequences(table='modlab_experiments'):
	"""
	This function extracts all sequences stored in a table column "sequence" and returns them as a list.

	:param table: the mysql database table to be queried
	:return: a list of sequences as strings
	:Example:

	>>> query_sequences(table='modlab_peptides')
	Password: *********
	Connecting to MySQL database...
	connection established!
	['YGGFL','WGKFFAGVKKLTKAILGEIA','WGKFFAGVKKLTKAILGEIA',....]
	"""
	try:
		conn = _connect()
		cursor = conn.cursor()
		cursor.execute('SELECT sequence FROM ' + table)

		row = cursor.fetchone()
		rows = []
		while row is not None:
			rows.append(row[0].encode('utf-8','ignore').strip()) # encode from unicode to utf-8 string
			row = cursor.fetchone()

		return rows

	except Error as e:
		print(e)


class Bunch(dict):
	"""Container object for datasets

	Dictionary-like object that exposes its keys as attributes. Taken from the ``sklearn`` package.

	:Example:

	>>> b = Bunch(a=1, b=2)
	>>> b['b']
	2
	>>> b.b
	2
	>>> b.a = 3
	>>> b['a']
	3
	>>> b.c = 6
	>>> b['c']
	6
	"""

	def __init__(self, **kwargs):
		dict.__init__(self, kwargs)

	def __setattr__(self, key, value):
		self[key] = value

	def __getattr__(self, key):
		try:
			return self[key]
		except KeyError:
			raise AttributeError(key)

	def __setstate__(self, state):
		# Bunch pickles generated with scikit-learn 0.16.* have an non
		# empty __dict__. This causes a surprising behaviour when
		# loading these pickles scikit-learn 0.17: reading bunch.key
		# uses __dict__ but assigning to bunch.key use __setattr__ and
		# only changes bunch['key']. More details can be found at:
		# https://github.com/scikit-learn/scikit-learn/issues/6196.
		# Overriding __setstate__ to be a noop has the effect of
		# ignoring the pickled __dict__
		pass


def load_AMPvsTM():
	"""
	Function to load a dataset consisting of AMP sequences and transmembrane regions of proteins.

	:return: Bunch, a dictionary-like object, the interesting attributes are: 'data', the data to learn, 'target', the
	classification labels, 'target_names', the meaning of the labels and 'feature_names', the meaning of the features.
	:Example:

	>>> from modlamp.database import load_AMPvsTM
	>>> data = load_AMPvsTM()
	>>> data.sequences[:5]
	array([['AAGAATVLLVIVLLAGSYLAVLA'],['LWIVIACLACVGSAAALTLRA'],['FYRFYMLREGTAVPAVWFSIELIFGLFA'],['GTLELGVDYGRAN'],['KLFWRAVVAEFLATTLFVFISIGSALGFK']])
	>>> list(data.target_names)
	['TM', 'AMP']
	>>> data.sequences.shape
	(412, 1)
	"""

	module_path = dirname(__file__)
	with open(join(module_path, 'data', 'AMPvsTM.csv')) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		n_samples = int(temp[0])
		n_features = int(temp[1])
		target_names = np.array(temp[2:])
		sequences = np.empty((n_samples,n_features), dtype='|S100')
		target = np.empty((n_samples,), dtype=np.int)

		for i, ir in enumerate(data_file):
			sequences[i] = ir[0]
			target[i] = ir[-1]

	return Bunch(sequences=sequences, target=target,
				 target_names=target_names,
				 feature_names=['Sequence'])

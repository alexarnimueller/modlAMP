# -*- coding: utf-8 -*-
"""
.. module:: modlamp.database

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to connect to the modlab internal peptide database server and query sequences from
the different database tables.
"""

import numpy as np
import mysql.connector
from getpass import getpass
from mysql.connector import Error

__author__ = "modlab"
__docformat__ = "restructuredtext en"


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


def query_sequences(table='modlab_peptides'):
	"""
	This function extracts all sequences from a given table in the modlab peptide database.

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
		query = ("SELECT sequence FROM " + table)
		cursor.execute(query)
		row = cursor.fetchone()
		rows = []
		while row is not None:
			rows.append(row[0].encode('utf-8', 'ignore').strip())  # encode from unicode to utf-8 string
			row = cursor.fetchone()

		return rows

	except Error as e:
		print(e)


def query_experiments(table='modlab_experiments', columns=['sequence']):
	"""
	This function extracts experimental results from the modlab peptide database. All data from the given table and
	column names is extracted and returned.

	:param table: the mysql database table to be queried
	:param columns: a list of the column names {str} to be extracted from the table
	:return: queried data as a numpy array
	:Example:

	>>> data = query_experiments(table='modlab_experiments', columns=['sequence', 'MCF7_activity', 'Saureus_activity'])
	Password: *********
	Connecting to MySQL database...
	connection established!
	>>> data[:5]
	array([	['ILGTILGILKGL', None, 1.0],
		['ILGTILGFLKGL', None, 1.0],
		['ILGNILGFLKGL', None, 1.0],
		['ILGQILGILKGL', None, 1.0],
		['ILGHILGYLKGL', None, 1.0]], dtype=object)

	.. note::
		If 'None' appears as a value, this means no data was measured for this peptide and not that activity is None
		(inactive).
	"""
	try:
		conn = _connect()
		cursor = conn.cursor()
		query = ("SELECT " + ', '.join(columns) + " FROM " + table)
		cursor.execute(query)
		rows = np.asarray(cursor.fetchall(), dtype='object')  # return data in a numpy array

		for i, c in enumerate(rows[0]):
			if isinstance(c, unicode):  # if a column is type 'unicode', it gets converted to 'utf-8'
				for n, e in enumerate(rows[:, i]):
					rows[:, i][n] = e.encode('utf-8', 'ignore').strip()
		return rows

	except Error as e:
		print(e)

"""
.. module:: database

.. moduleauthor:: modlab Alex Mueller ETH Zurich <alex.mueller@pharma.ethz.ch>

This module incorporates functions to connect to the modlab internal peptide database server and query sequences from
the different database tables.
"""

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
		password = raw_input('Password: ')

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


def query_sequences(table='ACP_annotation'):
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

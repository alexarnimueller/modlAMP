"""
.. warning::
	The delta_db module is used to connect to the internal modlab peptide database on the delta641 machine.
	This connection is only possible in the modlab intranet.
	On the delta machine, all mysql-tables containing sequences must be included in the peptides database and having
	sequences stored in a column called **sequence**.
"""

import mysql.connector

def read_db_config():
	"""
	Read database configuration and return a dictionary object.
	This function generally does not need to be used as it is called by the function :func:`query_sequences` directly.

	:return: a dictionary of database parameters
	"""
 
	db = {	'host' : 'gsdelta641.ethz.ch',
			'database' : 'peptides',
			'user' : 'modlab',
			'password' : '28.8.1749'
			}
 
	return db


def connect():
	"""
	Connect to a given MySQL database (in config.ini file).
	This function is called by the function :func:`query_sequences`.

	:return: a mysql.connector connection object
	:Example:

	>>> conn = connect()
	>>> conn
	<mysql.connector.connection.MySQLConnection at 0x105ec6650>
	"""
	config = read_db_config()

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

if __name__ == "__main__":
	connect()

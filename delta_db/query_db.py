from mysql.connector import Error
from connect_db import *


def query_sequences(table='ACP_annotation'):
	"""
	This function extracts all sequences stored in a table column "sequence" and returns them as a list.

	:param table: the mysql database table to be queried
	:return: a list of sequences as strings
	:Example:

	>>> query_sequences(table='modlab_peptides')
	Connecting to MySQL database...
	connection established!
	['YGGFL','WGKFFAGVKKLTKAILGEIA','WGKFFAGVKKLTKAILGEIA',....]
	"""
	try:
		conn = connect()
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


if __name__ == "__main__":
	query_sequences()
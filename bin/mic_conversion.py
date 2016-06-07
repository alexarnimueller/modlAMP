"""
A script to convert MIC values from ug/mL to uM.

The script reads a .csv file with 3 columns. Example:
Sequence, MIC, unit
IHHQGLPQE,,2500,uM
RKSKEKIGKEFKRIVQRIKDF,0.4,ug/ml
"""

from modlamp.descriptors import GlobalDescriptor


sequences = []
MIC = []
units = []
actives = {}

# read the file with 4 columns
with open('/Volumes/Platte1/x/projects/DBs/DBAASP/160606_Ecoli_clean.csv', 'r') as f:
	for line in f:
		sequences.append(line.split(',')[0])
		MIC.append(line.split(',')[1])
		units.append(line.split(',')[2])

D = GlobalDescriptor(sequences)
D.calculate_MW()
MW = D.descriptor.tolist()

for i, u in enumerate(units):
	if u == 'ug/ml\r\n':  # find MIC values in ug/mL
		if '+' in MIC[i]:
			mic = float(MIC[i].split('+')[0]) + float(MIC[i].split('+')[1])  # if with stdev, be conservative and take upper bound
			actives[sequences[i]] = round((mic / float(MW[i][0])) * 1000., 1)  # convert ug/mL to uM
		elif '-' in MIC[i]:
			mic = float(MIC[i].split('-')[0])  # if with stdev, be conservative and take upper bound
			actives[sequences[i]] = round((mic / float(MW[i][0])) * 1000., 1)  # convert ug/mL to uM
		else:
			actives[sequences[i]] = round((float(MIC[i]) / float(MW[i][0])) * 1000., 1)  # convert ug/mL to uM

s_inactive = [s for s, v in actives.items() if v >= 25.0]
s_active = [s for s, v in actives.items() if v < 25.0]
i = GlobalDescriptor(s_inactive)
a = GlobalDescriptor(s_active)

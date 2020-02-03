import numpy as np

def ddnewton(x,v):
	""" Create the table of Newtons divided differences based on (x,v) """
	m = len(x)
	DDNmat = np.zeros((m,m+1))

	# Inserting x into the 1st column and f into 2nd colume of table
	DDNmat[:m,0] = x
	DDNmat[:m,1] = v

	# create divided difference coefficients by recurrence
	for j in range(m-1):
		for k in range(m-j-1):
			DDNmat[k,j+2] = (DDNmat[k+1,j+1] - DDNmat[k,j+1])/(DDNmat[k+j+1,0] - DDNmat[k,0])

	# Extract max coefficient
	maxDDN = np.abs(DDNmat[0,m])

	return maxDDN,DDNmat


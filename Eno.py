import numpy as np
from Differences import *

def ENO(xloc, uloc, m, Crec):
	"""Purpose: Reconstruct the left (mu) and right (up) cell interface values 
	     using an ENO reconstruction based on 2m-1 long vectors."""

	# Treat special case of m=1 - no stencil to select
	if m == 1:
		um = uloc[0]
		up = uloc[0]

	# Apply ENO procedure to build stencil
	S = np.zeros(m).astype(int)
	S[0] = m

	for mm in range(m-1):
		# Left stencil
		a = np.array([S[0]-1])
		b = np.array(S[:mm+1])
		Sindxl = np.concatenate((a, b)).astype(int)
		Vl, DDNmat = ddnewton(xloc[Sindxl-1],uloc[Sindxl-1])
		# Right stencil:
		a = np.array(S[:mm+1])
		b = np.array([S[mm]+1])
		Sindxr = np.concatenate((a,b)).astype(int)
		Vr, DDNmat = ddnewton(xloc[Sindxr-1],uloc[Sindxr-1])

		# Choose stencil by divided differences
		S[:mm+2] = Sindxl
		if Vl > Vr:
			S[:mm+2] = Sindxr
	
	# Compute stencil shift 'r' and cell interface values
	r = m - S[0]

	up = np.dot(Crec[r+1,:],uloc[S-1])
	um = np.dot(Crec[r,:],uloc[S-1])

	return um,up


def ReconstructWeights(m,r):
	"""Purpose: Compute weights c_ir for reconstruction 
   v_{j+1/2} = sum_{j=0}^{m-1} c_{ir} v_{i-r+j}
   with m=order and r=shift (-1<=r<=m-1)."""

	c = np.zeros(m)
	harmonic = lambda n: np.sum([1.0/d for d in range(1,n+1)])

	fh = lambda s: (-1.0)**(s+m)*np.prod(np.arange(1,s+1))*np.prod(np.arange(1,m-s+1))

	for i in range(m):
		for q in range(i+1,m+1):
			if q != r+1:
				c[i] += fh(r+1)/fh(q)/(r+1-q)
			else:
				c[i] -= (harmonic(m-r-1)-harmonic(r+1))

	return c


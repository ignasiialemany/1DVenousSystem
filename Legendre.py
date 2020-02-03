import numpy as np

def LegendreGQ(m):
	"""Purpose: Compute the m'th order Legendre Gauss quadrature points, x, and weights, w"""

	if (m==0):
		x = 0.0
		w = 2.0
		return x,w

	# Form symmetric matrix from recurrence.
	J = np.zeros((m+1,m+1))
	h1 = 2.0*np.arange(m)

	J = np.diag( 2.0/(h1+2.0)*np.sqrt( np.arange(1,m+1)**4 / (h1+1)/ (h1+3) ), k=1 )
	J[0,0] = 0.0

	J += np.transpose(J)

	# Compute quadrature by eigenvalue solve
	x,V = np.linalg.eig(J)

	# sort eigenvalues
	idx = x.argsort()  
	x = x[idx]
	V = V[:,idx]

	w = 2.0*(V[0,:])**2

	return x,w



def LegendreGL(m):
	"""Purpose: Compute the m'th order LGL quadrature points, x, and weights, w"""

	x = np.zeros(m+1)
	w = np.zeros(m+1)

	if (m==1):
		x[0] = -1.0
		x[1] = 1.0
		w[0] = 1.0
		w[1] = 1.0
		return x,w

	if (m==2):
		x[0] = -1.0
		x[1] = 0.0
		x[2] = 1.0
		w[0] = 1.0/3
		w[1] = 4.0/3
		w[2] = 1.0/3
		return x,w

	# Form symmetric matrix from recurrence.
	J = np.zeros((m-1,m-1))
	h1 = 2*np.arange(m-2) + 2.0

	J = np.diag( 2.0/(h1+2.0)*np.sqrt( np.arange(1,m-1) * \
									(np.arange(1,m-1)+2.0) * \
									(np.arange(1,m-1)+1.0)**2 / (h1+1.0)/ (h1+3.0) ), k=1 )
	J[0,0] = 0.0

	J += np.transpose(J)

	# Compute quadrature by eigenvalue solve
	x,V = np.linalg.eig(J)

	# sort eigenvalues
	idx = x.argsort()  
	x = x[idx]
	#V = V[:,idx]

	x = np.concatenate(([-1.0],x,[1.0]))
	P = LegendreP(x,m)
	w = (2*m+1.0)/m/(m+1.0)/P**2

	return x,w



def LegendreP(x,m):
	""" Purpose: Evaluate orhonormal m'th order Legendre Polynomial at point x """

	xp = x
	dims = np.size(xp)

	# Initial values P_0(x) and P_1(x)
	PL = np.zeros((m+1,len(xp)))
	PL[0,:] = np.sqrt(0.5)
	if (m==0):
		return PL
	PL[1,:] = np.sqrt(1.5)*xp
	if (m==1):
		return PL[m,:]



	# Forward recurrence using the symmetry of the recurrence.
	aold = np.sqrt(1.0/3.0)
	for i in range(1,m):
		anew = 2.0/(2.0*i+2.0)*np.sqrt((i+1.0)**4/(2.0*i+1.0)/(2.0*i+3.0))
		PL[i+1,:] = 1.0/anew*(-aold*PL[i-1,:] + xp*PL[i,:])
		aold = anew

	return PL[m,:]

def GradLegendreP(r,m):
	"""Purpose: Evaluate the derivative of the m'th order Legendre polynomial
				at points r"""

	eps = np.finfo(float).eps

	dP = np.zeros(len(r))
	if (m>0):
		Ph = -m*r*LegendreP(r,m) + m*np.sqrt((2.0*m+1.0)/(2.0*m-1.0))*LegendreP(r,m-1)
		dPe = r**(m+1.0)*m*(m+1.0)/2.0*np.sqrt((2.0*m+1.0)/2.0)
		endp = (np.abs(np.abs(r)-1.0)>10*eps)
		rh = r*endp
		dP = np.logical_not(endp)*dPe + endp*Ph/(1.0-rh**2)

	return dP
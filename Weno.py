import numpy as np
from Legendre import *
from Lagrange import *
from Eno import ReconstructWeights

def WENO(xloc,uloc,m,Crec,dw,beta):
	""" Purpose: Compute the left and right cell interface values using an WENO
	         approach based on 2m-1 long vectors uloc with cell """


	# Set WENO parameters
	p = 1
	q = m-1
	vareps = 1e-6

	# Treat special case of m=1 - no stencil to select
	if (m==1):
		um = uloc[0]
		up = uloc[0]
	else:
		alpham = np.zeros(m)
		alphap = np.zeros(m)
		upl = np.zeros(m)
		uml = np.zeros(m) 
		betar = np.zeros(m)

		# Compute um and up based on different stencils and 
		# smoothness indicators and alpha
		for r in range(m):
			umh = uloc[m-r+np.arange(m)-1]
			upl[r] = np.dot(Crec[r+1,:],umh) 
			uml[r] = np.dot(Crec[r,:],umh)
			betar[r] = np.dot(umh, np.dot(beta[:,:,r],umh))

			# Compute alpha weights - classic WENO
			alphap = dw/(vareps+betar)**(2*p)
			alpham = np.flipud(dw)/(vareps+betar)**(2*p)
	    
			# Compute alpha weights - WENO-Z
			#tau = np.abs(betar[0] - betar[m-1])
			#if np.mod(m,2)==0:
			 #   tau = np.abs(betar[0]-betar[1] - betar[m-2] + betar[m-1])
			#alphap = dw*(1 + (tau/(vareps+betar))**q)
			#alpham = np.flipud(dw)*(1 + (tau/(vareps+betar))**q)

			# Compute nonlinear weights and cell interface values
			um = np.dot(alpham,uml)/np.sum(alpham)
			up = np.dot(alphap,upl)/np.sum(alphap)

	
	return um,up



def Qcalc(D,m,l):
	""" Purpose: Evaluate entries in the smoothness indicator for WENO """
	x,w = LegendreGQ(m)
	xq = x/2
	Qelem = 0

	for i in range(m+1): 
		xvec = np.zeros(m-l+1)
		for k in range(m-l+1):
			xvec[k] = xq[i]**(k)/np.prod(np.arange(1,k+1)) 
		Qelem += 0.5*w[i]*np.dot(xvec, np.dot(D,xvec)) 
	
	return Qelem


def betarcalc(x,m):
	""" Purpose: Compute matrix to allow evaluation of smoothness indicator in
        WENO based on stencil [x] of length m+1 .
        Returns sum of operators for l=1.m-1"""

	cw = lagrangeweights(x)
	errmat = np.zeros((m,m))

	for l in range(2,m+1):
		dw = np.zeros((m,m-l+1))
		for k in range(m-l+1):
			for q in range(m):
				dw[q,k] = np.sum(cw[(q+1):m+1,k+l])
	#Evaluate entries in matrix for order l
		Qmat = np.zeros((m,m))

		for p in range(m):
			for q in range(m):
				D = np.outer(dw[q,:],dw[p,:])
				Qmat[p,q] = Qcalc(D,m,l)

		errmat += Qmat

	return errmat

def LinearWeights(m, r0):
	""" Purpose: Compute linear weights for maximum accuracy 2m-1,
	using stencil shifted $r_0=-1,0$ points upwind."""
	A = np.zeros((m,m))
	b = np.zeros(m)

	# Setup linear system for coefficients
	for i in range(m):
		col = ReconstructWeights(m,i+r0)
		A[:(m-i),i] = col[i:m]

	# Setup righthand side for maximum accuracy and solve
	crhs = ReconstructWeights(2*m-1,m-1+r0)
	b = crhs[m-1:(2*m-1)]
	d = np.linalg.solve(A,b)

	return d
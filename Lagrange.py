import numpy as np

def lagrangeweights(x):
	"""Compute weights for Taylor expansion of Lagrange polynomial based
         on x and evaluated at 0.
         Method due to Fornberg (SIAM Review, 1998, 685-691)"""

	N = len(x)
	cw = np.zeros((N,N))

	cw[0,0]=1.0
	c1 = 1.0
	c4 = x[0]

	for i in range(2,N+1): #i=2:np
		mn = min(i,N-1)+1
		c2 = 1.0
		c5 = c4
		c4 = x[i-1]

		for j in range(i-1): #j=1:i-1
			c3 = x[i-1]-x[j]
			c2 = c2*c3

			if (j==i-2):
				for k in range(mn,1,-1): #k=mn:-1:2
					cw[i-1,k-1] = c1*((k-1)*cw[i-2,k-2]-c5*cw[i-2,k-1])/c2
				cw[i-1,0] = -c1*c5*cw[i-2,0]/c2;

			for k in range(mn,1,-1): #k=mn:-1:2
				cw[j,k-1] = (c4*cw[j,k-1]-(k-1)*cw[j,k-2])/c3
			cw[j,0] = c4*cw[j,0]/c3;
		c1=c2

	return cw
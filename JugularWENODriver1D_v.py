"""Driver script for solving the 1D Euler equations using a WENO scheme"""

import numpy as np
import matplotlib.pyplot as plt
from JugularWeno_v import *
from Legendre import LegendreGQ

#Space between nodes in both tubes
h1 = L1/N
h2 = L2/N

#Variables
q1 = np.zeros((2,N+1))
q2 = np.zeros((2,N+1))

# Initialize for Sod's problem - piecewise constant so no integration
#L = 1.0
#h = L/N
#x = np.linspace(0.0, L, N+1)

#for i in range(N+1):
#	if x[i] < 0.5:
#		q[0,i] = 1.0
#		q[2,i] = 1.0/(gamma-1)
#	else:
#		q[0,i] = 0.125
#		q[2,i] = 0.1/(gamma-1)

# Initialize for shock entropy problem
x1 = np.linspace(0.0,L1, N+1)
x2 = np.linspace(0.0,L2, N+1)

q1 = initialise1(q1,x1)
q2 = initialise2(q2,x2)

# Solve Problem
q1,q2 = VenousWENO1D([x1,x2],q1,q2,Av0,[h1,h2],m,CFL,FinalTime)

# Plot
plt.plot(x1,q1[0,:])
plt.title("Solution rho at time {}".format(FinalTime))
plt.show()
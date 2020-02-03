"""Driver script for solving the 1D venous equations using a WENO scheme"""

import numpy as np
import matplotlib.pyplot as plt
from JugularWeno import *
from Legendre import LegendreGQ
import pickle
import simplejson
import cProfile

h = L/N
q = np.zeros((2,N+1))

# Initialisation
x = np.linspace(0.0,L, N+1)
q = initialise(q,x)


# Solve Problem
q = VenousWENO1D(x,q,h,m,CFL,FinalTime)

Vel = q[1]/q[0]
speedofsound = c(q)
Sindex = Vel/speedofsound


plt.plot(x / L, q[0] / A0)
plt.title('T')
plt.ylabel("A/A0")
plt.xlabel("Length (m) ")

plt.plot(x / L, q[1] / q[0])
plt.title('T')
plt.ylabel("A/A0")
plt.xlabel("Length (m) ")

plt.plot(x / L, Sindex)
plt.title('T')
plt.ylabel("A/A0")
plt.xlabel("Length (m) ")

plt.show()

f1 = open('A-Portal-1Exact(t=0.3)PROVA.txt','w')
f2 = open('V-Portal-1Exact(t=0.3)PROVA.txt','w')
f3 = open('X-Portal-1Exact(t=0.3)PROVA.txt','w')
f4 = open('S-Portal-1Exact(t=0.3)PROVA.txt','w')

Area = np.array(q[0]).tolist()
Vel = np.array(Vel).tolist()
Sindex = np.array(Sindex).tolist()
x = np.array(x).tolist()

simplejson.dump(Area,f1)
simplejson.dump(Vel,f2)
simplejson.dump(x,f3)
simplejson.dump(Sindex,f4)


f1.close()
f2.close()
f3.close()
f4.close()
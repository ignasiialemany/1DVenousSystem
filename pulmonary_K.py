"""
This file contains all the basic implementations of functions to test the coupling between 1D and valve model,
to recreate Figure 6 in Mynard (10.1002/cnm.1466).
"""
from valves import Valveflow
from artery import *
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Order of the method
m = 1
CFL = 1
N = 10
FinalTime = 1.5#########################################
rawfile = ['/Users/ignasi/Desktop/1D HoiMing/output/tubeg2.csv','/Users/ignasi/Desktop/1D HoiMing/output/tubeg1.csv']
rawfilev = '/Users/ignasi/Desktop/1D HoiMing/output/valveg.csv'
L1   = 0.001
L2   = 0.006
A0   = 7.1e-4

Av0     = 0      #initialise valve area
lv      = 10    #valve length
K_open  = 0.2
K_close = 0.1
dp_open = 0

Avmax = A0
Q = 0

mu = 3e-3

###################################################################################
def initialise1(q,x):
    q[0] = A_in(0)
    q[1] = 0
    return q

def initialise2(q,x):
    q[0] = A_out(0)
    q[1] = 0
    return q

def friction(q,x):
    A = q[0]
    u = uu(q)
    f = -22*np.pi*mu*(u/A)
    return f

tt = 0

P,A = np.loadtxt('A2P.csv', delimiter=',', unpack=True)
plt.plot(P,A)
plt.show()
P2A = interp1d(P,A)   #A(P)

t,P = np.loadtxt('verify_valve_Fig6_rv.csv', delimiter=',', unpack=True)
P *= 133.322
A_in = interp1d(t,P2A(P))

t,P = np.loadtxt('verify_valve_Fig6_pt.csv', delimiter=',', unpack=True)
P *= 133.322
A_out = interp1d(t,P2A(P))

def boundary_v(q1,q2,Av,k,tt):
    tt %= 0.895
    # BOUNDARY CONDITIONS: LHS of tube 1
    q1L = boundaries(3,q1[:,0],A_in(tt))

    # BOUNDARY CONDITIONS: valve
    global Q
    (Q,dA) = Valveflow(Q ,p(q1[:,-1]) , p(q2[:,0]),Av,k,rho,lv,[Avmax, K_open,K_close,dp_open])
    q1R = boundaries(4,q1[:,-1],Q)
    q2L = boundaries(5,q2[:, 0],Q)

    # BOUNDARY CONDITIONS: RHS of tube 2
    q2R = boundaries(2,q2[:,-1],A_out(tt))
    return q1L,q1R,q2L,q2R,dA
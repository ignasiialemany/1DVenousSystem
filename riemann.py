"""
This file contains all the basic implementations of functions to compute
quantities for the specific case of the Euler equations.
"""
import numpy as np
import tinyarray as ta
from scipy.optimize import fsolve
from scipy import integrate

# Order of the method
m = 2

# Set problem parameters
CFL = 0.4
FinalTime = 6e-4
N = 1200
#########################################
outputDir = 'output/riemann.csv'

E = 10**5
L = 3e-2
A0 = np.pi *10**-6
K = E/(12*(3/4))*(10**-2)**3
rho = 1050
p_d = 0

q_inflow = 40e-6
mu = 4e-3

## 1D ##
###################################################################################
def initialise(q,x):
    for i in range(len(x)):
        if x[i]<0.8*L:
            q[0,i] = 5*A0
        else:
            q[0,i] = 1e-9
    q[1] = 0
    return q

def friction(q,x):
    f = 0
    return f

def boundary(q):
    AL = 5*A0
    uL = 0

    AR = 1e-9
    uR = 0
    return AL,AR,uL,uR

####
def p(q):
    α = q[0]/A0
    m=10
    n=-3/2
    p = p_d + K*(α**m-α**n)
    return p

def c(q):
    α = q[0]/A0
    m=10
    n=-3/2
    c = np.sqrt(K/rho * (m*α**m - n*α**n))
    return c

def uu(q):
    u = q[1]
    return u

def speed(A):
    speed = 4*c([A,0])
    return speed


def VenousExact(qL,qR,maxvel):
    AL = qL[0,:]
    AR = qR[0,:]
    uL = qL[1,:]
    uR = qR[1,:]
    flux = np.empty_like(qL)
    for i in range(len(AL)):
        func = lambda A : fK(abs(A),AR[i],A0) + fK(abs(A),AL[i],A0)+uR[i]-uL[i]
        A = abs(fsolve(func, (AR[i]+AL[i])*0.5)[0])
        u = 0.5*(uL[i]+uR[i]+fK(A,AR[i],A0)-fK(A,AL[i],A0))
        
        flux[:,i] = EulerFlux([A,u])
    return flux
def fK(A,AK,A0):
    m = 10
    n = -3/2
    if A<=AK:
        fK = integrate.quad(lambda a: c([a,0])/a, AK,A)[0]
    else:
        #BK = m/(m+1)*(A**(m+1)-AK**(m+1))/A0**m - n/(n+1)*(A**(n+1)-AK**(n+1))/A0**n
        #fK = np.sqrt(K/rho*BK*(A-AK)/(A*AK))
        fK = np.sqrt(2/rho * (p([A,0])-p([AK,0]))/(A**2-AK**2)) * (A-AK)
    return fK

def EulerFlux(q):
    """Purpose: Compute flux for 1D Euler equations."""
    f1 = q[0]*q[1]
    f2 = 0.5*q[1]**2 + p(q)/rho
    flux = np.array((f1, f2))

    return flux

def EulerChar(q0):
    """Purpose: Compute characteristic decomposition for Euler equations at q0"""
    #iS A S = Lam
    n = q0.shape[0]

    c0 = c(q0)
    A = q0[0]
    u0 = q0[1]
    S = np.zeros((n,n))
    iS = np.zeros((n,n))
    Lam = np.zeros((n,n))

    S[0,0] = 1
    S[0,1] = -1
    S[1,0] = c0/A
    S[1,1] = c0/A
    S /= c0/A+c0/A

    iS[0,0] = c0/A
    iS[0,1] = 1
    iS[1,0] = -c0/A
    iS[1,1] = 1

    Lam[0,0] = u0+c0
    Lam[1,1] = u0-c0

    return S,iS,Lam


#######
def write2file(U,t,outputDir):
    file=open(outputDir,'a')
    np.savetxt(file,np.r_[t, U.ravel()][None],delimiter=',')
    file.close()

def EulerLF(u, v, maxvel):
    """Purpose: Evaluate global Lax Friedrich numerical flux for 
                the Euler equations
	"""
    fu = EulerFlux(u)
    fv = EulerFlux(v)
    
    flux = 0.5*(fu+fv)-0.5*maxvel*(v-u)
    return flux
"""
This file contains all the basic implementations of functions for venous model.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy import integrate

A0 = 7.1e-4
K = 25*133.322
rho = 1060
p_d = 15*133.322
m=10
n=-3/2

def boundaries(type,q,param):
    if type==1:
        #u_inflow
        u_inflow = param
        W1 = u_inflow + speed(q[0])  # Left-running characteristic
        W2 = uu(q[:]) - speed(q[0])  # Right-running characteristic
        func = lambda A : W1-W2-2*speed(A)
        A  = fsolve(func, q[0])
        u  = (W1+W2)/2
    elif type==2:
        #A_out
        A_out = param
        W1 = uu(q) + speed(q[0])
        W2 = W1 - 2*speed(A_out)
        func = lambda A : W1-W2-2*speed(A)
        A = A_out#fsolve(func, A_out)
        u = (W1+W2)*0.5
    elif type==3:
        #A_in
        A_in = param
        W2 = uu(q) - speed(q[0])
        W1 = W2 + 2*speed(A_in)
        func = lambda A : W1-W2-2*speed(A)
        A = A_in #fsolve(func, A_out)
        u = (W1+W2)*0.5
    elif type==4:
        #valves
        Q = param
        W1 = uu(q) + speed(q[0])
        func = lambda A : W1 - Q/A - speed(A)
        A = fsolve(func, q[0])
        u = Q/A
    elif type==5:
        #valves
        Q = param
        W2 = uu(q) - speed(q[0])
        func = lambda A : W2 - Q/A + speed(A)
        A = fsolve(func, q[0])
        u = Q/A
    return [A,u]

####
def p(q):
    α = q[0]/A0
    p = p_d + K*(α**m-α**n)
    return p

def c(q):
    α = q[0]/A0
    c = np.sqrt(K/rho * (m*α**m - n*α**n))
    return c

def uu(q):
    u = q[1]
    return u

def speed(A):
    #speed = 4*c([A,0])
    speed = integrate.quad(lambda a: c([a,0])/a, A0,A)[0]
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
    #m = 10
    #n = -3/2
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
def write2file(U,t,rawfile):
    file=open(rawfile,'a')
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
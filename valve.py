"""
This file contains all the basic implementations of functions to compute
quantities for the specific case of the Euler equations.
"""
import numpy as np
import tinyarray as ta
from scipy.optimize import fsolve
from scipy import integrate
from valves import *

# Order of the method
m = 2

# Set problem parameters
CFL = 1
FinalTime = 20
N = 500
#########################################
rawfile = 'valves.csv'

L = 2
A0 = 5e-4
K = 5
rho = 1000
p_d = 0


Av = 0    #initialise valve area

q_inflow = 40e-6
mu = 4e-3

## 1D ##
###################################################################################
def initialise(q,x):
    q[0] = A0*(0.2+1.8*x/L)
    q[1] = q_inflow/q[0]
    return q

def friction(q,x):
    A = q[0]
    u = uu(q)
    #f = 9.81 - 8*math.pi*mu/(ρ*A) * u * (1+A/A0_in * (A<A0_in))
    f = 9.81 - 8*mu*np.pi/rho*(u/A) #f2
    #f = 9.81 - 0.96e-4 * np.sqrt(A/A0) *(u/A)
    #f = 9.81 -8*mu*math.pi/ρ*(u/A) * np.sqrt(A/A0_in)   #f
    #f = 9.81
    #f = -0.96e-4*(u/A)
    return f

def boundary(q):
    # BOUNDARY CONDITIONS: LHS of domain
    u_inflow = q_inflow/q[0,0]  # u = Q_inflow/A
    (AL,uL) = boundaries(1,q[:,0],u_inflow)
    # BOUNDARY CONDITIONS: RHS of domain
    A_out = 2*A0
    (AR,uR) = boundaries(2,q[:,-1],A_out)
    return AL,AR,uL,uR

def boundary_v(q,Av):
    # BOUNDARY CONDITIONS: LHS of tube 1
    u_inflow = q_inflow/q[0,0]  # u = Q_inflow/A
    (AL,uL) = boundaries(1,q[:,0],u_inflow)
    
    # BOUNDARY CONDITIONS: RHS of domain
    A_out = 2*A0
    (AR,uR) = boundaries(3,q[:,-1],[Av , p(q[0,-1]) , p(q[0,0])])
    
    # BOUNDARY CONDITIONS: RHS of tube 2
    A_out = 2*A0
    (AR,uR) = boundaries(2,q[:,-1],A_out)
    return AL,AR,uL,uR

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
        #valves
        Av = param[0]
        p1 = param[1]
        p2 = param[2]
        W1 = uu(q) + speed(q[0])
        Q = Valveflow(q[0]*q[1] ,p1,p2,Av,h,rho,lv)
        func = lambda A : W1 - Q/A - speed(A)
        A = fsolve(func, q[0])
        u = Q/A

    return A,u

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
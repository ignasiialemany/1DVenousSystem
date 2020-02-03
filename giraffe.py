"""
This file contains all the basic implementations of functions to compute
quantities for the specific case of the Euler equations.
"""
import numpy as np
from scipy.optimize import fsolve
from scipy import integrate


# Order of themethod
m = 1

# Set problem parameters
CFL = 0.45
FinalTime = 0.3
N = 100

#########################################
outputDir = 'output/giraffe.csv'


# q_inflow = 0
# L = 3*10**-2
# A0 = np.pi * (10**-3)**2 #5e-4 # m^2
# mu = 4e-3
# E=10**5
# h0 = 10**-5
# r0 = 10**-3
# K = ((E)/(12*(1-0.25)))*((h0/r0)**3)
# rho = 1050
# p_d = 0
# Acol = 1*10**-9

#Giraffe
#q_inflow = 40e-6
#L = 2
#A0 = 5e-4 # m^2
#E=10**5
#h0 = 10**-5
#r0 = 10**-3
#mu = 0.004
#K = 5
#rho = 1050
#p_d = 0
#Acol = 0.002*A0

#Clamp example
L = 0.06
A0 = 0.8*10**-4
K = 1
rho = 1000
p_d = 0
q_inflow = 0
mu = 4e-3
Acol = 0.005*A0

## 1D ##
###################################################################################
def initialise(q,x):
    for i in range(0, len(x)):
        if x[i] < (4 * L) / 10:
            q[0, i] = (1.1 - x[i] * (10.95 / (4 * L))) * A0
        elif x[i] <= (6 * L) / 10 and x[i] >= (4 * L) / 10:
            q[0, i] = 0.005 * A0
        else:
            q[0, i] = (-1.6375 + x[i] * (10.95 / (4 * L))) * A0

    #q[1] = 0
    #q[0] = 2*A0
    q[1] = 0
    #q[0] = (0.2 + 1.8 * (x / L)) * A0
    #q[1] = q_inflow

    # for i in range (len(x)):
    #    if x[i]/L < 0.8:
    #        q[0,i]=5*A0
    #    else:
    #        q[0,i] = 0.5*A0

    #q[0] = (0.2 + 1.8 * (x / L)) * A0
    # for i in range (0,len(x)):
    #     if x[i] < (4*L)/10:
    #         q[0,i] = (1.1 - x[i]*(10.95/(4*L)))*A0
    #     elif x[i] <= (6*L)/10 and x[i] >= (4*L)/10:
    #         q[0,i] = 0.005*A0
    #     else:
    #         q[0,i] = (-1.6375 + x[i]*(10.95/(4*L)))*A0
    return q

def friction(q,x,t):
    A = q[0]
    u = q[1]/A
    #f = 9.81 - 8*math.pi*mu/(ρ*A) * u * (1+A/A0_in * (A<A0_in))
    #f = - 0.96*10**-4*(u)
    #f = -((8*np.pi*mu*np.sqrt(A))/(rho*np.sqrt(A0)))*u
    f = -((8*np.pi*mu)/(rho))*u
    #f = - ((8 * np.pi * mu * np.sqrt(A)) / (rho * np.sqrt(A0))) * u
    #f = - 8*mu*np.pi/rho*(u/A) #f2
    #f = 9.81 - 0.96e-4 * np.sqrt(A/A0) *(u/A)
    #f = 9.81 -8*mu*math.pi/ρ*(u/A) * np.sqrt(A/A0_in)   #f
    #f = 9.81
    #f = -0.96e-4*(u/A)
    return f

def boundary(q):
    n = -(3/2)
    #Inlet boundary
    #We fix u
    u_inlet = q_inflow/q[0,0]
    #W1inlet = integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), A0, q[0,0], weight='cauchy', wvar=0)[0] - u_inlet
    #W2inlet = integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), A0, q[0,-1], weight='cauchy', wvar=0)[0] + q[1,-1]/q[0,-1]


    # AR = 2 * A0
    # uR = q[1, -1]
    # uL = q_inflow
    # AL = q[0,1]

    AR = 1.1 * A0
    uR = q[1, -1]
    uL = q[1, 0]
    AL = 1.1 * A0

    return AL,AR,uL,uR

####
def p(q):
    α = q[0]/A0
    m=10
    n=-3/2
    p = 0 + K*(α**m-α**n)
    return p

def c(q):
    α = q[0]/A0
    m=10
    n=-3/2
    c = np.sqrt(K/rho * (m*α**m - n*α**n))
    return c

def uu(q):
    u = q[1]/q[0]
    return u

def VenousExact(qL,qR,q,maxvel,x):

    n = -(3/2)

    AL = qL[0,:]
    AR = qR[0,:]

    uL = qL[1,:]/AL
    uR = qR[1,:]/AR

    cL = c([AL, 0])
    cR = c([AR, 0])

    pL = p([AL, 0])
    pR = p([AR, 0])

    flux = np.empty_like(qL)

    for i in range(len(AL)):

        if q[0,i] <= Acol:
            Astar = Acol
            ustar = 0
        else:

            #Solve Star Variables
            func4 = lambda A: fK(A, AR[i]) + fK(A, AL[i]) + (uR[i] - uL[i])
            Astar = fsolve(func4, (AR[i]+AL[i])*0.5)[0]

            if Astar <= Acol:
                Astar = Acol

            ustar = 0.5*(uL[i]+uR[i]+fK(Astar,AR[i])-fK(Astar,AL[i]))
            Cstar = c([Astar,0])
            pstar = p([Astar,0])


             #   Astar = Acol
              #  pstar = p([Astar, 0])

            if 0 <= ustar:

                #Left Rarefraction Wave
                    if Astar <= AL[i]:

                    #Head Rarefraction
                        SHL = uL[i]-cL[i]

                    #Left state in the interface
                        if 0 <= SHL:

                            ustar = uL[i]
                            Astar = AL[i]

                        else:
                        #Compute Tail Rarefraction
                            STL = ustar - Cstar

                        #Star Region in the interface
                            if STL < 0:

                                Astar = Astar
                                ustar = ustar
                                #ustar= uL[i] - integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AL[i], Astar, weight='cauchy', wvar=0)[0]

                            else: #Inside Rarefraction
                                bx = integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AL[i], AL[i], weight='cauchy', wvar=0)[0]
                                #func = lambda A: uL[i] + bx - np.sqrt((K / rho) * (10 * (A / A0) ** 10 - n * (A / A0) ** n)) - integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AL[i], A, weight='cauchy', wvar=0)[0]
                                func = lambda A: uL[i] + bx - np.sqrt((K / rho) * (10 * (A / A0) ** 10 - n * (A / A0) ** n)) + integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AL[i], A, weight='cauchy', wvar=0)[0]
                                Astar = fsolve(func, AL[i])[0]
                                ustar = np.sqrt((K / rho) * (10 * (Astar / A0) ** 10 - n * (Astar / A0) ** n))

                    else: #Left shock

                    #Compute Speed Shock
                        BL = (K / rho) * ((10 / (10 + 1)) * ((Astar ** (10 + 1) - AL[i] ** (10 + 1)) / (A0 ** 10)) - (n / (n + 1)) * ((Astar ** (n + 1) - AL[i] ** (n + 1)) / (A0 ** n)))
                        ML = np.sqrt(BL * ((Astar * AL[i]) / (Astar - AL[i])))
                        SL = uL[i]-(ML/AL[i])

                    #Left State in the interface
                        if 0 <= SL:
                            ustar = uL[i]
                            Astar = AL[i]

                        else: #Inside Star Region
                            Astar = Astar
                            ustar = ustar
                            #ustar = uL[i] - np.sqrt(BL * ((Astar - AL[i]) / (Astar * AL[i])))

                #Right side
            else:

                    #Right shock
                    if Astar > AR[i]:

                    #Compute speed shock
                        BR = (K / rho) * ((10 / (10 + 1)) * ((Astar ** (10 + 1) - AR[i] ** (10 + 1)) / (A0 ** 10)) - (n / (n + 1)) * ((Astar ** (n + 1) - AR[i] ** (n + 1)) / (A0 ** n)))
                        MR = np.sqrt(BR * ((Astar * AR[i]) / (Astar - AR[i])))
                        SR = uR[i] + (MR / AR[i])
                        #print('hey there2')
                        if 0 >= SR:
                            ustar = uR[i]
                            Astar = AR[i]

                        else:
                            #ustar2 = uR[i] + np.sqrt(BR * ((Astar - AR[i]) / (Astar * AR[i])))
                            #print(ustar2-ustar)
                            ustar = ustar
                            Astar = Astar

                    else:#Right Rarefraction

                    #Head Rarefraction
                        SHR = uR[i]+cR[i]

                        if 0>=SHR:
                            ustar = uR[i]
                            Astar = AR[i]

                        else:
                        #Tail
                            STR = ustar + Cstar

                            if 0 <= STR:
                                #print('hey there')
                                Astar = Astar
                                ustar = ustar
                                #ustar = uR[i] + integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AR[i], Astar, weight='cauchy',wvar=0)[0]

                            else: #Inside Rarefraction
                                by = integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AR[i], AR[i], weight='cauchy', wvar=0)[0]
                                #func = lambda A: -uR[i] + by - np.sqrt((K / rho) * (10 * (A / A0) ** 10 - n * (A / A0) ** n)) - integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AR[i], A, weight='cauchy', wvar=0)[0]
                                func = lambda A: -uR[i] + by - np.sqrt((K / rho) * (10 * (A / A0) ** 10 - n * (A / A0) ** n)) + integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AR[i], A, weight='cauchy', wvar=0)[0]
                                Astar = fsolve(func, AR[i])[0]
                                ustar = -np.sqrt((K / rho) * (10 * (Astar / A0) ** 10 - n * (Astar / A0) ** n))
                                #ustar = -Cstar

                #Compute flux from Astar and ustar


        flux[:, i] = EulerFlux([Astar, ustar])

    return flux


def VenousExact2(qL,qR,q,maxvel,x):

    n = -(3/2)

    AL = qL[0]
    AR = qR[0]

    uL = qL[1]/AL
    uR = qR[1]/AR

    cL = c([AL, 0])
    cR = c([AR, 0])

    pL = p([AL, 0])
    pR = p([AR, 0])

    flux = np.empty_like(qL)

    for i in range(0,1):

        if q < Acol:
            Astar = Acol
            ustar = 0

        else:

            #Solve Star Variables
            func4 = lambda A: fK(A, AR) + fK(A, AL) + (uR - uL)
            Astar = fsolve(func4, (AR+AL)*0.5)[0]

            ustar = 0.5*(uL+uR+fK(Astar,AR)-fK(Astar,AL))
            Cstar = c([Astar,0])
            pstar = p([Astar,0])

            if Astar <= Acol:
                Astar = Acol
                ustar = 0

            if 0 <= ustar:

                #Left Rarefraction Wave
                    if Astar <= AL:

                    #Head Rarefraction
                        SHL = uL-cL

                    #Left state in the interface
                        if 0 <= SHL:

                            ustar = uL
                            Astar = AL

                        else:
                        #Compute Tail Rarefraction
                            STL = ustar - Cstar

                        #Star Region in the interface
                            if STL < 0:

                                Astar = Astar
                                ustar = ustar
                                #ustar= uL[i] - integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AL[i], Astar, weight='cauchy', wvar=0)[0]

                            else: #Inside Rarefraction
                                bx = integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AL, AL, weight='cauchy', wvar=0)[0]
                                func = lambda A: uL + bx - np.sqrt((K / rho) * (10 * (A / A0) ** 10 - n * (A / A0) ** n)) - integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AL, A, weight='cauchy', wvar=0)[0]
                                Astar = fsolve(func, AL)[0]
                                ustar = np.sqrt((K / rho) * (10 * (Astar / A0) ** 10 - n * (Astar / A0) ** n))

                    else: #Left shock

                    #Compute Speed Shock
                        BL = (K / rho) * ((10 / (10 + 1)) * ((Astar ** (10 + 1) - AL ** (10 + 1)) / (A0 ** 10)) - (n / (n + 1)) * ((Astar ** (n + 1) - AL ** (n + 1)) / (A0 ** n)))
                        ML = np.sqrt(BL * ((Astar * AL) / (Astar - AL)))
                        SL = uL-(ML/AL)

                    #Left State in the interface
                        if 0 <= SL:
                            ustar = uL
                            Astar = AL

                        else: #Inside Star Region
                            Astar = Astar
                            ustar = ustar
                            #ustar = uL[i] - np.sqrt(BL * ((Astar - AL[i]) / (Astar * AL[i])))

                #Right side
            else:

                    #Right shock
                    if Astar > AR:

                    #Compute speed shock
                        BR = (K / rho) * ((10 / (10 + 1)) * ((Astar ** (10 + 1) - AR ** (10 + 1)) / (A0 ** 10)) - (n / (n + 1)) * ((Astar ** (n + 1) - AR ** (n + 1)) / (A0 ** n)))
                        MR = np.sqrt(BR * ((Astar * AR) / (Astar - AR)))
                        SR = uR + (MR / AR)
                        #print('hey there2')
                        if 0 >= SR:
                            ustar = uR
                            Astar = AR

                        else:
                            #ustar2 = uR[i] + np.sqrt(BR * ((Astar - AR[i]) / (Astar * AR[i])))
                            #print(ustar2-ustar)
                            ustar = ustar
                            Astar = Astar

                    else:#Right Rarefraction

                    #Head Rarefraction
                        SHR = uR+cR

                        if 0>=SHR:
                            ustar = uR
                            Astar = AR

                        else:
                        #Tail
                            STR = ustar + Cstar

                            if 0 <= STR:
                                #print('hey there')
                                Astar = Astar
                                ustar = ustar
                                #ustar = uR[i] + integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AR[i], Astar, weight='cauchy',wvar=0)[0]

                            else: #Inside Rarefraction
                                by = integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AR, AR, weight='cauchy', wvar=0)[0]
                                func = lambda A: -uR + by - np.sqrt((K / rho) * (10 * (A / A0) ** 10 - n * (A / A0) ** n)) - integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AR, A, weight='cauchy', wvar=0)[0]
                                Astar = fsolve(func, AR)[0]
                                ustar = -np.sqrt((K / rho) * (10 * (Astar / A0) ** 10 - n * (Astar / A0) ** n))
                                #ustar = -Cstar

                #Compute flux from Astar and ustar

        flux[:] = EulerFlux([Astar, ustar])

    return flux

def fK (A, AK):

    n = -(3 / 2)

    if A <= AK:
        fk = integrate.quad(lambda b: np.sqrt((K / rho) * (10 * (b / A0) ** 10 - n * (b / A0) ** n)), AK, A, weight='cauchy', wvar=0)[0]

    else:

        B = (K / rho) * ((10 / (10 + 1)) * ((A ** (10 + 1) - AK ** (10 + 1)) / (A0 ** 10)) - (n / (n + 1)) * ((A ** (n + 1) - AK ** (n + 1)) / (A0 ** n)))
        fk = np.sqrt((B * (A - AK)) / (AK * A))

    return fk

def EulerFluxFriedrich(q):
    """Purpose: Compute flux for 1D Euler equations."""

    A = q[0]
    u = q[1]/A
    mx = 10
    n = - (3/2)
    f1 = A*u
    f2 = A*u*u + ((K*A0)/rho)*(((mx)/(mx+1))*((A/A0)**(mx+1)) - ((n)/(n+1))*((A/A0)**(n+1)))

    flux = np.array((f1, f2))

    return flux

def EulerFlux(q):
    """Purpose: Compute flux for 1D Euler equations."""

    A = q[0]
    u = q[1]
    mx = 10
    n = - (3/2)
    f1 = A*u
    f2 = A*u*u + ((K*A0)/rho)*(((mx)/(mx+1))*((A/A0)**(mx+1)) - ((n)/(n+1))*((A/A0)**(n+1)))

    flux = np.array((f1, f2))

    return flux


def EulerChar(q0):
    """Purpose: Compute characteristic decomposition for Euler equations at q0"""
    #iS A S = Lam
    n = q0.shape[0]

    c0 = c(q0)
    A = q0[0]
    u0 = q0[1]/q0[0]

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
    fu = EulerFluxFriedrich(u)
    fv = EulerFluxFriedrich(v)
    
    flux = 0.5*(fu+fv)-0.5*maxvel*(v-u)
    return flux

def checkarea(q):
    for i in range (len(q[0])):
        if q[0,i] <= Acol:
            q[0, i] = Acol

    return q
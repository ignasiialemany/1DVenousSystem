"""
This file contains the specific functions to solve venous or arterial flow.
"""

import numpy as np
from giraffe import *        #change problem config file
from Weno import *
from helpers import extend
import matplotlib.pyplot as plt
import simplejson

## 1D ##
###################################################################################
def EulerWENOcharrhs1D(x,q,h,k,m,Crec,dw,beta,maxvel,t):
    """Evaluate the RHS of Euler equations using a WENO reconstruction
    on the characteristic variables."""
    
    n = q.shape[0]
    N = len(x)
    dq = np.zeros((n,N))
    qe = np.zeros((n,N+2*m))

    # Retrieve boundary conditions
    AL,AR,uL,uR = boundary(q)

    # # Extend data and assign boundary conditions
    xe,qe[0,:] = extend(x, q[0,:], m, "D", AL, "D", AR)
    xe,qe[1,:] = extend(x, q[1,:], m, "D", uL, "D", uR)
    
    #define cell left and right interface values
    Rlh = np.zeros(n)
    Rrh = np.zeros(n)
    qm = np.zeros((n,N+2))
    qp = np.zeros((n,N+2))


    for i in range(N+2):
        qloc = qe[:,i:1+(i+2*(m-1))]
        q0 = qloc[:,m-1]
        S,iS,Lam = EulerChar(q0)
        Rloc = np.dot( iS, qloc )
        Rlh[0], Rrh[0] = WENO(xe[i:1+(i+2*(m-1))],Rloc[0,:],m,Crec,dw,beta)
        Rlh[1], Rrh[1] = WENO(xe[i:1+(i+2*(m-1))],Rloc[1,:],m,Crec,dw,beta)
        qm[:,i] = np.dot(S, Rlh)
        qp[:,i] = np.dot(S, Rrh)


    f1 = open('LeftFirstXWENOreconstruction.txt','w+')
    #f2 = open('RightFirstWENOreconstruction.txt','w+')
    xARlh = np.array(xe).tolist()
    #qARrh = np.array(qp[0]).tolist()

    simplejson.dump(xARlh,f1)

    f1.close()

    #qm =checkarea(qm)
    #qp =checkarea(qp)

    #Right flux is computed by getting the left side 1+N+1 and the right side 2+N+2
    #The left flux is computed by getting as left side 0N and right side 1 N+1

    RFluxLF,RFluxExact = fluxright(qp, qm,q, maxvel,x)
    LFluxLF,LFluxExact = fluxleft (qp, qm,q, maxvel,x)

    UpositiveExact = q - 2*(CFL/maxvel)*RFluxExact
    UnegativeExact = q + 2*(CFL/maxvel)*LFluxExact
    UpositiveLF = q - 2 * (CFL / maxvel) * RFluxLF
    UnegativeLF = q + 2 * (CFL / maxvel) * LFluxLF

    Rightflux = RFluxExact
    #Rightflux = positivitypreserving(UpositiveExact,UnegativeExact,UpositiveLF,UnegativeLF,RFluxLF,RFluxExact,LFluxLF,LFluxExact)
    Leftflux = LFluxExact

    dq = -(Rightflux-Leftflux)/h

    dq[1,:] += friction(q,x,t)
    
    return dq

def positivitypreserving(UpositiveExact,UnegativeExact,UpositiveLF,UnegativeLF,RFluxLF,RFluxExact,LFluxLF,LFluxExact):
    param1 = 1
    param2 = 1
    epsilonA = min(10**-13,0.005*A0)
    PPFluxRight = np.zeros_like(RFluxLF)
    #We set the last flux to real flux and analyze the others from i to i+1
    PPFluxRight[:,-1] = RFluxExact[:,-1]
    for i in range (len(UpositiveExact[0])-1):
        if UpositiveExact[0,i] < epsilonA:
            funcepsilonA = lambda ep: (1-ep)*UpositiveLF[0,i]+ep*UnegativeExact[0,i]-epsilonA
            param1 = fsolve(funcepsilonA,1)
        elif UnegativeExact[0,i+1] < epsilonA:
            funcepsilonA = lambda ep: (1-ep)*UnegativeLF[0,i+1] + ep*UnegativeExact[0,i+1]-epsilonA
            param2 = fsolve(funcepsilonA,1)

        paramF = min(param1,param2)
        PPFluxRight[:,i] = (1-paramF)*RFluxLF[:,i] + paramF*RFluxExact[:,i]

    return PPFluxRight

def fluxleft (qp, qm,q, maxvel,x):

    FluxLF = np.zeros_like(q)
    FluxExact = np.zeros_like(q)

    #We will split two cases; When A<= than Acol and when A>Acol.
    #First we will loop through the entire q array
    for i in range (0,len(q[0])):
        leftside = qp[:,i]
        rightside = qm[:,i+1]
        FluxLF[:,i] = EulerLF(leftside,rightside,maxvel)
        FluxExact[:,i] = VenousExact2(leftside,rightside,q[0,i],maxvel,x)

    return FluxLF,FluxExact

def fluxright (qp, qm,q, maxvel,x):

    FluxLF = np.zeros_like(q)
    FluxExact = np.zeros_like(q)

    # We will split two cases; When A<= than Acol and when A>Acol.
    # First we will loop through the entire q array
    for i in range(0, len(q[0])):
        leftside = qp[:, i+1]
        rightside = qm[:, i + 2]
        FluxLF[:, i] = EulerLF(leftside, rightside, maxvel)
        FluxExact[:, i] = VenousExact2(leftside, rightside, q[0,i], maxvel, x)

    return FluxLF, FluxExact


def VenousWENO1D(x,q,h,m,CFL,FinalTime):
    """ Integrate 1D Euler equation until FinalTime using an ENO
        scheme and a 3rd order SSP-RK
    """   
    t = 0.0
    tstep = 0

    #Initialize reconstruction weights
    Crec = np.zeros((m+1,m))
    for r in range(-1,m):
        Crec[r+1,:] = ReconstructWeights(m,r)

    # Initialize linear weights
    dw = LinearWeights(m,0)

    # Compute smoothness indicator matrices
    beta = np.zeros((m,m,m))
    for r in range(m):
        xl = -1/2 + np.arange(-r,m-r+1)
        beta[:,:,r] = betarcalc(xl,m)

    print(outputDir)
    file=open(outputDir,'w')
    file.write("t, ")
    np.savetxt(file,np.r_[x.ravel(),x.ravel()][None],delimiter=',', fmt='%f')
    file.close()
    write2file(q,0,outputDir)

    global AminInitial
    AminInitial = min(q[:,0])

    while t < FinalTime:

        maxvel = (c(q) + abs(uu(q))).max()
        k = min(FinalTime-t, CFL*h/maxvel)
        #((4*np.pi*h**2)/(1-16*(np.pi*h)**4))
        #Update solution
        t = t+k

        rhsq  = EulerWENOcharrhs1D(x,q,h,k,m,Crec,dw,beta,maxvel,t)
        q1 = q + k*rhsq
        rhsq  = EulerWENOcharrhs1D(x,q1,h,k,m,Crec,dw,beta,maxvel,t)
        q2 = (3*q + q1 + k*rhsq)/4
        rhsq  = EulerWENOcharrhs1D(x,q2,h,k,m,Crec,dw,beta,maxvel,t)
        q  = (q + 2*q2 + 2*k*rhsq)/3

        tstep += 1

        plt.plot(x / L, q[0] / A0)
        plt.title('T= %d' % tstep)
        plt.ylabel("A/A0")
        plt.xlabel("Length (m) ")

        cspeeds = c(q)
        u =  q[1] / q[0]
        #Sindex = u/cspeeds
        plt.plot(x / L, u)
        plt.title('T= %d' % tstep)
        plt.ylabel("A/A0")
        plt.xlabel("Length (m) ")

        plt.ion()
        plt.draw()
        plt.pause(0.0000000000000001)

        if tstep % 10 == 0:
            print("Iteration %s" % tstep, "t=%fs" % t)


        
    return q

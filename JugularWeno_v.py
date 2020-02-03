"""
This file contains the specific functions to solve Euler equations
in 1D or 2D using a WENO scheme.
"""

import numpy as np
from pulmonary_K import *
from Weno import WENO,ReconstructWeights,LinearWeights,betarcalc
from helpers import extend

## 1D ##
###################################################################################
def EulerWENOcharrhs1D(x,q1,q2,Av,h,k,m,Crec,dw,beta,maxvel):
    """Evaluate the RHS of Euler equations using a WENO reconstruction
    on the characteristic variables."""

    # Retrieve boundary conditions
    q1L,q1R,q2L,q2R,dA = boundary_v(q1,q2,Av,k,tt)

    d1q = applyBoundary(q1,q1L,q1R,h[0],x[0],m,Crec,dw,beta,maxvel)
    d2q = applyBoundary(q2,q2L,q2R,h[1],x[1],m,Crec,dw,beta,maxvel)
    return d1q,d2q,dA

def applyBoundary(q,qL,qR,h,x,m,Crec,dw,beta,maxvel):
    n = q.shape[0]
    N = len(x)
    dq = np.zeros((n,N))
    qe = np.zeros((n,N+2*m))

    # # Extend data and assign boundary conditions
    xe,qe[0,:] = extend(x, q[0,:], m, "D", qL[0], "D", qR[0])
    xe,qe[1,:] = extend(x, q[1,:], m, "D", qL[1], "D", qR[1])
    
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

    # Change numerical flux here
    dq = - (EulerLF(qp[:,1:N+1], qm[:,2:N+2], maxvel) - \
            EulerLF(qp[:,:N], qm[:,1:N+1], maxvel))/h

    dq[1,:] += friction(q,x)

    return dq

def VenousWENO1D(x,q1,q2,Av,h,m,CFL,FinalTime):
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

    for i in range(len(rawfile)):
        print(rawfile[i])
        file=open(rawfile[i],'w')
        file.write("t, ")
        np.savetxt(file,np.r_[x[i].ravel(),x[i].ravel()][None],delimiter=',', fmt='%f')
        file.close()
    write2file(q1,0,rawfile[0])
    write2file(q2,0,rawfile[1])
    f= open(rawfilev,"w+")

    while t < FinalTime:
        
        maxvel = np.array([ c(q1) + abs(uu(q1)) , c(q2) + abs(uu(q2)) ]).max()
        k = min(FinalTime-t, CFL*h[0]/maxvel, CFL*h[1]/maxvel)

        #Update solution
        rhsq1,rhsq2,dA  = EulerWENOcharrhs1D(x,q1,q2,Av,h,k,m,Crec,dw,beta,maxvel)
        q1_1 = q1 + k*rhsq1
        q2_1 = q2 + k*rhsq2
        Av_1 = Av + k*dA

        rhsq1,rhsq2,dA  = EulerWENOcharrhs1D(x,q1_1,q2_1,Av_1,h,k,m,Crec,dw,beta,maxvel) 
        q1_2 = (3*q1 + q1_1 + k*rhsq1)/4
        q2_2 = (3*q2 + q2_1 + k*rhsq2)/4
        Av_2 = (3*Av + Av_1 + k*dA)/4

        rhsq1,rhsq2,dA  = EulerWENOcharrhs1D(x,q1_2,q2_2,Av_2,h,k,m,Crec,dw,beta,maxvel) 
        q1 = (q1 + 2*q1_2 + 2*k*rhsq1)/3
        q2 = (q2 + 2*q2_2 + 2*k*rhsq2)/3        
        Av = (Av + 2*Av_2 + 2*k*dA)/3

        t = t+k
        global tt
        tt = t
        tstep += 1

        if (q1!=q1).any():
            write2file(q2,t,rawfile[0])
            asd
            print("error")
        if tstep%100==0:
            print("Iteration %s" % tstep, "t=%fs" %t)
            write2file(q1,t,rawfile[0])
            write2file(q2,t,rawfile[1])

            f= open(rawfilev,"a+")
            f.write("%f ," %t)
            f.write("%f ," %(p(q1[:,0])-p(q2[:,-1])))
            f.write("%f \n" %(Av/Avmax))
            f.close()
        
        
    return q1,q2
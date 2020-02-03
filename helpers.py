""" 
This file contains all the helper functions used in the codes for conservation laws.
In particular, functions to extend the stencils, the slope limiters and the functions 
to initialize the problems.
"""
import numpy as np


def extend(x, q, m, BCl, ql, BCr, qr):
    """Purpose: Extend dependent and independent vectors (x,u), by m cells 
                subject to approproate boundary conditions.
                BC = "D" - Dirichlet
                BC = "N" - Neumann
                BC = "P" - periodic
                ul/ur - BC value - only active for Dirichlet BC

    """
    N = len(x)
    h = x[1]-x[0]

    qe = np.zeros(N+2*m)
    xe = np.zeros(N+2*m)

    qe[m:-m] = q
    xe[m:-m] = x

    xe[:m] = x[0] - np.arange(m,0,-1)*h
    xe[N+m:] = x[-1] + np.arange(1,m+1)*h

    if BCl == "P" or BCr == "P": #periodic
        qe[:m] = q[-m-1:-1]
        qe[N+m:] = q[1:m+1]
        return xe,qe

    if BCl == "D":       
        qe[:m] = -q[m:0:-1] + 2*ql
    else:
        qe[:m] = q[m:0:-1]

    if BCr == "D":
        qe[N+m:] = -q[-2:-2-m:-1] + 2*qr
    else:
        qe[N+m:] = q[-2:-2-m:-1]

    return xe,qe


def extendstag(x, q, m, BCl, ql, BCr, qr):
    """Purpose: Extend dependent and independent vectors (x,u), by m cells 
                subject to approproate boundary conditions. Grid is assumed staggered
                BC = "D" - Dirichlet
                BC = "N" - Neumann
                BC = "P" - periodic
                ul/ur - BC value - only active for Dirichlet BC
    """
    N = len(q)
    h = x[1]-x[0]

    xe = np.zeros(N+2*m)
    qe = np.zeros(N+2*m)
    qe[m:-m] = q

    # Extend x
    xe[m:-m] = x

    xe[:m] = x[0] - np.arange(m,0,-1)*h
    xe[N+m:] = x[-1] + np.arange(1,m+1)*h

    # Periodic extension of u
    if BCl == "P" or BCr == "P":
        qe[:m] = q[-m:]
        qe[N+m:] = q[:m]
        return xe, qe

    # Left extension
    if BCl == "D":       
        qe[:m] = -q[m-1::-1] + 2*ql
    else:
        qe[:m] = q[m-1::-1]

    # Right extension
    if BCr == "D":
        qe[N+m:] = -q[-1:-1-m:-1] + 2*qr
    else:
        qe[N+m:] = q[-1:-1-m:-1]

    return xe, qe


def extendDG(u, BCl, ul, BCr, ur):
    """Purpose: Extend dependent and independent vectors u with m+1 entries
     subject to approproate boundary conditions for DG formulation
     BC = "D" - Dirichlet
     BC = "N" - Neumann
     BC = "P" - periodic
     u - BC value - only active for Dirichlet BC
    """

    dim = u.shape

    if (len(dim)>1):
        m = dim[0]-1
        N = dim[1]

        ue = np.zeros((m+1,N+2))
        ue[:,1:N+1] = u

        if BCl == "P" or BCr == "P": #periodic
            ue[:,0] = u[:,N-1]
            ue[:,N+1] = u[:,0]
            return ue

        if BCl == "D":       
            ue[:,0] = -np.flipud(u[:,0]) + 2*ul
        else:
            ue[:,0] = np.flipud(u[:,0])

        if BCr == "D":
            ue[:,N+1] = -np.flipud(u[:,N-1]) + 2*ur
        else:
            ue[:,N+1] = np.flipud(u[:,N-1])


    else:
        m = 1
        N = dim[0]

        ue = np.zeros(N+2)
        ue[1:N+1] = u

        if BCl == "P" or BCr == "P": #periodic
            ue[0] = u[N-1]
            ue[N+1] = u[0]
            return ue

        if BCl == "D":       
            ue[0] = -np.flipud(u[0]) + 2*ul
        else:
            ue[0] = np.flipud(u[0])

        if BCr == "D":
            ue[N+1] = -np.flipud(u[N-1]) + 2*ur
        else:
            ue[N+1] = np.flipud(u[N-1])


    return ue



def SlopeLimit(a, b, typ, c, M, h):
    """Define slope limiter function based on type. 
        c is used in one limiter
        M,h is used for TVB limiting"""

    duL = np.zeros(a.shape)
    # No slope
    if typ==0:
        return duL
    # minmod limiter
    elif typ==1:
        return minmod(np.array((a,b)))
    #MUSCL limiter
    elif typ == 2:
        return minmod(np.array((2*a, 2*b, 0.5*(a+b))))
    # Superbee limiter
    elif typ == 3:
        return minmod( np.array(( maxmod( np.array((a,b)) ), \
                                  minmod( np.array((2.0*a,2.0*b)) ) )) )
    # van Albada limiter
    elif typ ==  4:
        return minmod( np.array(( ((a**2.0+c**2.0)*b+(b**2+c**2)*a)/(a**2+b**2+c**2) , \
                                    2*a, 2*b)))
    # van Leer limiter
    elif typ == 5:
        return minmod( np.array(( 2*a*b/(a+b), 2*a, 2*b )) )

    # TVB limiting
    elif typ == 6:
        return minmodTVB( np.array(( 0.5*(a+b), 2*a, 2*b )), M, h)

    return np.zeros(a.shape)



def FluxLimit(r, typ, beta):
    """Purpose: Define flux limiter function based on type. 
                Beta is used in some limiters"""

    r = np.fmax(0.0,r)

    # No flux limiter
    if typ == 0:
        phi = np.zeros(len(r))

    # Chakravarthy/Osher limiter
    elif typ == 1:
        phi = np.fmin(r,beta)

    # Koren limiter
    elif typ == 2:
        phi = np.fmin( np.fmin(2.0*r, (2.0+r)/3.0), 2.0 )

    # Sweby limiter
    elif typ == 3:
        phi = np.fmax( np.fmin(beta*r, 1.0), np.fmin(r,beta) )

    # OSPRE limiter
    elif typ == 4:
        phi = 1.5*(r**2+r)/(r**2+r+1.0)

    # van Leer limiter
    elif typ == 5:
        phi = 2*r/(1.0+r)

    phi = np.fmax(0.0,phi)
    return phi


def minmod(v):
    """Purpose: Implement the midmod function v is a vector"""
    m,N = v.shape
    psi = np.zeros(N)
    s = sum( np.sign(v), 0 )/m
    indeces = np.where(np.abs(s) == 1)
    psi[indeces] = s[indeces]*np.min(np.abs(v[:,indeces]),0)
    return psi


def minmodTVB(v,M,h):
    """Purpose: Implement the TVB modified midmod function on row vector v"""
    psi = v[0,:]
    ids = np.where( np.abs(psi) > M*h**2 )
    if ids[0].size:
        psi[ids[0]] = minmod(v[:,ids[0]])

    return psi


def maxmod(v):
    """Purpose: Implement the maxmod function on vector v"""

    m,N = v.shape
    psi = np.zeros(N)
    s = sum( np.sign(v), 0 )/m
    indeces = np.where(np.abs(s) == 1)
    psi[indeces] = s[indeces]*np.max(np.abs(v[:,indeces]),0)
    return psi



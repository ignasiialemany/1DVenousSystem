"""
This file contains all the basic implementations of functions to implement valves.
"""


def Valveflow(Q,p1,p2,Av,k,rho,lv,param):
    dP = p1-p2
    dA = Valvearea(Av,dP,param)
    Av += k*dA

    Avmax = param[0]
    B = rho/2/(Av+Avmax*1e-6)**2
    L = rho*lv/(Av+Avmax*1e-6)
    Q = Q + k/L * (dP - B*Q*abs(Q))
    return (Q,dA)

def Valvearea(Av, dp,param):
    """Implement valves"""
    #valves(q[:,-1],dq[:,-1])
    Avmax = param[0]
    K_open = param[1]
    K_close = param[2]
    dp_open = param[3]

    ξ = Av/Avmax
    if dp>dp_open:
        dξ = (1-ξ)*K_open*(dp-dp_open)
    elif dp<dp_open:
        dξ = ξ *K_close*(dp-dp_open)
    else:
        dξ = 0
    dA = Avmax*dξ
    return dA
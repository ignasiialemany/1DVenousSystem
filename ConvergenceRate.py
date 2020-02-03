import numpy as np
import sympy as sym

x = sym.Symbol('x')
t = sym.Symbol('t')
Ainitial= 10**-4
A0 = 10**-4
AU0= 10**-7
T0= 3
L=  1
a= 10**-5



Area = Ainitial + a*sym.sin(((2*np.pi)/(L))*x)*sym.cos(((2*np.pi)/(T0))*t)
Caudal = AU0 -((a*L)/(T0))*sym.cos(((2*np.pi)/(L))*x)*sym.sin(((2*np.pi)/(T0))*t)
Vel = Area/Caudal
Flux2 = Area * Vel * Vel
#First equation dA/dt + dAu/dx = 0
dAdt = sym.diff(Area,t)
dAUdx = sym.diff(Caudal,x)

#Second equation
dAUdt = sym.diff(Caudal,t)
Dflux2dx = sym.diff(Flux2,x)

#print(dAdt)
def sourceterms(xval,tval):
    termdAUdt = dAUdt.subs({x:xval,t:tval})
    termflux2 = Dflux2dx.subs({x:xval,t:tval})
    sourceterms = termdAUdt + termflux2
    return sourceterms


As = sourceterms(0.2,3)
print(As)



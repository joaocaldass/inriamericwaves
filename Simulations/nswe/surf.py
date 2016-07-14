import numpy as np
import matplotlib.pyplot as plt
import solver
plt.style.use('ggplot')
#%% Caso 1: gaussian
g = 9.81
nx = 100
nt = 100
cfl = 0.45
hmin = 1e-3

x = np.linspace(-1,1,nx)
dx = np.diff(x)[0]

b = np.exp(-x**2/0.1)*0.0
h0 = 1.0*np.ones_like(b)
u0 = np.zeros_like(h0)


plt.plot(x,b)
plt.plot(x,h0,'o-')

def bcs_open(h,u,b):
    bcl = np.zeros((3,2))
    bcr = np.zeros((3,2))
    bcl[0,0] = h[1]
    bcl[1,0] = u[1]
    bcl[2,0] = b[1]
    
    bcl[0,1] = h[0]
    bcl[1,1] = u[0]
    bcl[2,1] = b[0]
    
    bcr[0,0] = h[-1]
    bcr[1,0] = u[-1]
    bcr[2,0] = b[-1]
    
    bcr[0,1] = h[-2]
    bcr[1,1] = u[-2]
    bcr[2,1] = b[-2]    
    
    return bcl, bcr
  

#%% El solver
h = np.zeros((nt,nx))
u = np.zeros((nt,nx))
bcl = np.zeros((nt,3,2))
bcr = np.zeros((nt,3,2))

h[0,:] = h0
#hu = np.zeros((nt,nx))

t = 0.0
n = -1

#%% loop
#for n in range(nt-1): 
n = n+1
dt = cfl*dx/np.max(np.abs(u[n,:])+np.sqrt(g*h[n,:])) #new dt
t = t+dt #old dt
bcl[n,:,:], bcr[n,:,:] = bcs_open(h[n,:], u[n,:], b)
fr, fl, s = solver.fluxes(bcl[n,:,:],bcr[n,:,:],h[n,:],u[n,:],b,dx,hmin,solver.roe)
U = np.vstack([h[n,:],h[n,:]*u[n,:]]) -dt/dx*(fl-fr-s)

h[n+1,:] = U[0,:]
u[n+1,:] = np.where(h[n+1,:]>hmin,U[1,:]/h[n+1,:],0.0)

plt.close()
plt.subplot(211)
plt.plot(h[n+1,:])
plt.subplot(212)
plt.plot(x,u[n+1,:])
plt.savefig('figs/fig%i'%(n+1))





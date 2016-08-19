
import numpy as np
import matplotlib.pyplot as plt
g = 9.81
def roe2(hl,hul,hr,hur):
    """
        otra version del solver de roe
    """
    if hl<0 or hr<0:
        print 'NEGATIVE DEPTH :('
    ul = 0.0
    if hl>0: ul = hul/hl
    
    ur = 0.0
    if hr>0: ur = hur/hr
        
    #averaged state
    uhat = 0.5*(ul+hr)
    chat = 0.5*(np.sqrt(g*hl)+np.sqrt(g*hr))
    
    #averaged eigenvalues
    #notice l1 <= l3
    lhat_1 = uhat - chat
    lhat_3 = uhat + chat
    
    #interface state
    us = ul
    hs = hl
    if lhat_1>0:
        us = ul
        hs = hl
    elif lhat_3 <0:
        us = ur
        hs = hr
    else:
        cs = chat- 0.25*(ur-ul)
        hs = cs*cs/g
        us = uhat - (np.sqrt(g*hr)-np.sqrt(g*hl))
        
    return hs,us    
g = 9.81
def roe(hl,hul,hr,hur):
    """
        El solver de roe del paper de Marche (2006?)
    """
    if hl<0 or hr<0:
        print hl,hr
    ul = 0
    if hl>0: ul = hul/hl
    
    ur = 0
    if hr>0: ur = hur/hr
        
    wl1 = ul - 2*np.sqrt(g*hl)
    wl2 = ul + 2*np.sqrt(g*hl)

    wr1 = ur - 2*np.sqrt(g*hr)
    wr2 = ur + 2*np.sqrt(g*hr)

    uhat = 0.5*(ul+ur)
    hhat = 0.25*(np.sqrt(hl) + np.sqrt(hr))**2

    l1 = uhat - np.sqrt(g*hhat)
    l2 = uhat + np.sqrt(g*hhat)
    l1l = ul - np.sqrt(g*hl)
    l2l = ul + np.sqrt(g*hl)
    l1r = ur - np.sqrt(g*hr)
    l2r = ur + np.sqrt(g*hr)    
  
    #entropy fix programado en el surfwb-uc
    if l1>0:
        ws1 = wl1
        ws2 = wl2
    else:
        ws1 = wr1        
        if l2>0:
            ws2 = wl2
        else:
            ws2 = wr2

    us = 0.5*(ws1+ws2)
    hs = (ws2-ws1)**2/(16.*g)
    
    #entropy fix de marche
    if l1l<0 and l1r > 0:
        us = uhat
        hs = hhat
        
    if l2l<0 and l2r > 0:
        us = uhat
        hs = hhat
    #print hs,us
    return hs,us
def minmod(slope1, slope2):
    if (slope1>0 and slope2 > 0):
        return min(slope1,slope2)
    if (slope1<0 and slope2<0):
        return max(slope1,slope2)
    return 0.

minmod = np.vectorize(minmod)
def musclrecontr(q0,q1,q2,dx=1):
    """
        Receives states q=(h,hu) from a cell (call it 1) 
        and its first neighbors (0 and 2) 
        and returns muscl reconstruction q_{il} and q_{ir} of 
        conserved variables and bathymetry at  cell boundaries
        
        dx is irrelevant for regular grids
        but is kept for code semantic and mantainability        
    """
    
    #conserved variables u
    slope1 = (q1-q0)/dx
    slope2 = (q2-q1)/dx
#     slope = map(minmod,slope1,slope2)
    slope = minmod(slope1,slope2)
    qil = q1-0.5*dx*slope
    qir = q1+0.5*dx*slope

#     h0 = q0[0]
#     h1 = q1[0]
#     h2 = q2[0]
#     hu0 = q0[1]
#     hu1 = q1[1]
#     hu2 = q1[1]
#     u0 = 0.0
#     u1 = 0.0 
#     u2 = 0.0
#     if h0 > 1e-6: u0 = hu0/h0
#     if h1 > 1e-6: u1 = hu1/h1
#     if h2 > 1e-6: u2 = hu2/h2
    
#     s1 = (h1-h0)/dx
#     s2 = (h2-h0)/dx
#     s =  minmod(s1,s2)
#     hl = h1-0.5*dx*s
#     hr = h1+0.5*dx*s
    
#     s1 = (u1-u0)/dx
#     s2 = (u2-u0)/dx
#     s =  minmod(s1,s2)
#     ul = u1-0.5*dx*s
#     ur = u1+0.5*dx*s
    
#     qil = np.array([hl,hl*ul])
#     qir = np.array([hr,hr*ur])
    return qil, qir
def getMusclReconstr(h,hu):
    """
    Receives 1d arrays h,hu (1xNx+4)
    and returns hl,hr, hul,hur (1\times nx+2)
    """
    nx = h.shape[0]-4
    hl = np.zeros((nx+2,))
    hr = np.zeros((nx+2,))
    hul = np.zeros((nx+2,))
    hur = np.zeros((nx+2,))
    for i in range(1,nx+3,1):
        q0 = np.array([h[i-1],hu[i-1]])
        q1 = np.array([h[i],hu[i]])
        q2 = np.array([h[i+1],hu[i+1]])
        (hl[i-1],hul[i-1]), (hr[i-1],hur[i-1]) = musclrecontr(q0,q1,q2)
    return hl, hr, hul, hur  
def setdt(h,hu,n,dx,cfl):
    """
        Calcula el dt segun condicion de CFL
    """
    u_n = np.where(h[n,:]>1e-5, hu[n,:]/h[n,:], 0.)
    s = np.max(np.abs(u_n)+np.sqrt(9.81*h[n,:]))
    dt = cfl*dx/np.max(s)
    return dt
def flux(h,u):
    """
        Receives scalars h,u
        Returns array F(u) of size 2
    """
    return np.array([h*u, 0.5*g*h**2 + h*u**2])
def fluxes(h,hu,n,riemann_solver=roe):
    """
        Calcula loos flujos en cada interfaz,
        retorna la matriz de 2xninterfaces
    """
    nx = h.shape[1]-4
    hl, hr, hul, hur   = getMusclReconstr(h[n,:],hu[n,:])
    fs = np.zeros((2,nx+1))
    for i in range(nx+1):
        hs,us = riemann_solver(hr[i],hur[i],hl[i+1],hul[i+1])
        fs[:,i] = flux(hs,us)
    return fs

def fluxes2(h,hu,n):
    """
        Calcula loos flujos en cada interfaz,
        retorna la matriz de 2xninterfaces
    """
    nx = h.shape[0]-4
    hl, hr, hul, hur   = getMusclReconstr(h[:],hu[:])
    fs = np.zeros((2,nx+1))
    for i in range(nx+1):
        hs,us = roe(hr[i],hur[i],hl[i+1],hul[i+1])
        fs[:,i] = flux(hs,us)
    return fs
def bcs_closed(h,hu,n):
    """ 
        recibe las matrices y coloca los valores 
        correspondientes a la celda cerrada.
        
        Este es el tipico borde cerrado.
        
        No estoy seguro
        si modificar h,hu aqui dentro
        hace que se modifique fuera,
        asi que uso hb,hub
    """
    hb = 1.*h
    hub = 1.*hu
    hb[n,0] = h[n,2]
    hb[n,1] = h[n,2]
    hub[n,0] = -hu[n,2]
    hub[n,1] = -hu[n,2]
    
    hb[n,-1] = h[n,-3]
    hb[n,-2] = h[n,-3]
    hub[n,-1] = -hu[n,-3]   
    hub[n,-2] = -hu[n,-3]    
    return hb,hub
def bcs_closed_2(h,hu,n):
    """ 
        recibe las matrices y coloca los valores 
        correspondientes a la celda cerrada.
        
        Este es el tipico borde cerrado.
        
        No estoy seguro
        si modificar h,hu aqui dentro
        hace que se modifique fuera,
        asi que uso hb,hub
    """
    hb = 1.*h
    hub = 1.*hu
    hb[n,0] = h[n,3]
    hb[n,1] = h[n,2]
    hub[n,0] = -hu[n,3]
    hub[n,1] = -hu[n,2]
    
    hb[n,-1] = h[n,-4]
    hb[n,-2] = h[n,-3]
    hub[n,-1] = -hu[n,-4]   
    hub[n,-2] = -hu[n,-3]    
    return hb,hub
def bcs_open(h,hu,n):
    """ 
        recibe las matrices y coloca los valores 
        correspondientes a la celda cerrada.
        
        Este es el tipico borde cerrado.
        
        No estoy seguro
        si modificar h,hu aqui dentro
        hace que se modifique fuera,
        asi que uso hb,hub
    """
    hb = 1.*h
    hub = 1.*hu
    hb[n,0] = h[n,2]
    hb[n,1] = h[n,2]
    hub[n,0] = hu[n,2]
    hub[n,1] = hu[n,2]
    
    hb[n,-1] = h[n,-3]
    hb[n,-2] = h[n,-3]
    hub[n,-1] = hu[n,-3]   
    hub[n,-2] = hu[n,-3]    
    return hb,hub
def simulate(h,hu,bcs,dx,cfl,t0,nt,riemann_solver=roe):
    """
        Rutina principal que corre la simulacion
    """
    t = np.zeros((nt,))
    for n in range(nt-1):     
        
        dt = setdt(h,hu,n,dx,cfl)
        
        t[n+1] = t[n] + dt

        h,hu = bcs(h,hu,n)    
        
        f = fluxes(h,hu,n,riemann_solver=roe)

        h[n+1,2:-2] = h[n,2:-2] -dt/dx*(f[0,1:] - f[0,:-1])
        hu[n+1,2:-2] = hu[n,2:-2] -dt/dx*(f[1,1:] - f[1,:-1])
        
    return t,h,hu
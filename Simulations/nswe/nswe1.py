
import numpy as np
g = 9.81
def roe(hl,hul,hr,hur,i=None,kappa=1e-4):
    """
        El solver de roe del paper de Marche (2006?)
    """
    if np.isnan(hl) or np.isnan(hr):
        if not i : 
            print hl,hr
        else:
            print i,hl,hr,'nan'

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
    if np.sqrt(hs*g) <= kappa:
        hs = kappa
        us = 0.0        
    return hs,us
import numpy as np
g = 9.81
def roe_surf(hl,hul,hr,hur, i=None,kappa=1e-4):
    """
        El solver de roe del paper de Marche (2006?)
    """
    if hl<0 or hr<0:
        if not i : 
            print hl,hr
        else:
            print i,hl,hr
#     if np.isnan(hl) or np.isnan(hr):
#         if not i : 
#             print hl,hr
#         else:
#             print i,hl,hr,'nan'
    cl = np.sqrt(g*hl)
    ul = 0
    if hl>0: ul = hul/hl
    
    cr = np.sqrt(g*hr)
    ur = 0
    if hr>0: ur = hur/hr
        
    cm = 0.5*(cl + cr)
    um = 0.5*(ul + ur)
    
    lambda1 = um - cm
    lambda3 = um + cm
    
    cs = 0.0
    us = 0.0
    if hl > kappa or hr > kappa:
        if lambda1 > 0.0:
            cs = cl
            us = ul
        elif lambda3 < 0.0:
            cs = cr
            us = ur
        else:
            cs = cm - 0.25*(ur-ul)
            us = um  -(cr-cl)
        
        #entropy fix
        if ul - cl < 0 and ur - cr > 0 :
            cs = 0.5*(cl + cr)
            us = 0.5*(ul + ur)
        if ul + cl < 0 and ur + cr > 0 :
            cs = 0.5*(cl + cr)
            us = 0.5*(ul + ur)
        
    #detect dry middle state
    if cs <= kappa:
        hs = kappa
        us = 0.0
    else:
        hs = cs**2/g
        
    return hs,us
import numpy as np
g = 9.81
def roe1(hl,hul,hr,hur,i=None):
    """
        El solver de roe del paper de Marche (2006?)
    """
    kappa = 1e-10
    if hl<0 or hr<0:
        if not i : 
            print hl,hr
        else:
            print i,hl,hr
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

    ws1 = 0.0
    ws2 = 0.0
    hs = 0.0
    us = 0.0
    if hl>kappa or hr>kappa:
        if l1>0:
            us = ul
            hs = hl
        elif l2<0:
            us = ur
            hs = hr
        else:
            ws1 = wr1    
            ws2 = wl2
            us = 0.5*(ws1+ws2)
            hs = (ws2-ws1)**2/(16.*g)
    
    #entropy fix de marche
    if ul - np.sqrt(g*hl)<0 and ur - np.sqrt(g*hr) > 0:
        us = uhat
        hs = hhat
        
    if ul + np.sqrt(g*hl)<0 and ur + np.sqrt(g*hr)    > 0:
        us = uhat
        hs = hhat
    return hs,us
def flux(h,u):
    """
        h,u escalares, returna F(U)
    """
    return np.array([h*u, 0.5*g*h*h + h*u*u])
def fluxes_matrix(h,hu,n,riemann_solver,kappa=1e-4):
    """
        Receives full matrices h,hu (nt,nx)
        including ghost cells   and calculates returns
        fluxes for inner cells  interfaces in row of index n
    """
    nx = h.shape[1]
    f = np.zeros((2,nx-1))
    for i in range(nx-1):
        hs,us = riemann_solver(h[n,i],hu[n,i],h[n,i+1],hu[n,i+1],i,kappa=kappa)
        f[:,i] = flux(hs,us)
    return f
def fluxes_row(h0,hu0,nx):
    """
        Like fluxes0, but for a given row 
        h0, hu0 of size (nx,).
    """
    f = np.zeros((2,nx-1))
    for i in range(nx-1):
        hs,us = roe(h0[i],hu0[i],h0[i+1],hu0[i+1])
        f[:,i] = flux(hs,us)
    return f
def bcs_closed(h,hu,n):
    """ 
        Applies first order "closed" boundary
        conditions to ghost cells of row n of h and hu. 
        This is, homogeneous Neumann and Dirichlet for h and hu
        respectively
    """
    hb = 1.*h
    hub = 1.*hu
    hb[n,0] = h[n,1]
    hub[n,0] = -hu[n,1]
    hb[n,-1] = h[n,-2]
    hub[n,-1] = -hu[n,-2]    
    return hb,hub
def bcs_open(h,hu,n):
    """ 
        Applies first order "open" boundary
        conditions on row n of h and hu. 
        This is, homogeneous Neumann for h and hu
        respectively
    """
    hb = 1.*h
    hub = 1.*hu
    hb[n,0] = h[n,1]
    hub[n,0] = hu[n,1]
    hb[n,-1] = h[n,-2]
    hub[n,-1] = hu[n,-2]    
    return hb,hub
def setdt(h,hu,n,dx,cfl,kappa=1e-5):
    """
        Calcula el dt segun condicion de CFL
    """
    u_n = np.where(h[n,:]>kappa, hu[n,:]/h[n,:], 0.)
    s = np.max(np.abs(u_n)+np.sqrt(9.81*h[n,:]))
    dt = cfl*dx/np.max(s)
    return dt
def simulate(h,hu,bcs,dx,cfl,t0,nt,riemann_solver,kappa=1e-4):
    """
        Rutina principal que corre la simulacion
    """
    t = np.zeros((nt,))
    for n in range(nt-1):     
        dt = setdt(h,hu,n,dx,cfl)
        
        t[n+1] = t[n] + dt

        h,hu = bcs(h,hu,n)    
        f = fluxes_matrix(h,hu,n,riemann_solver, kappa=kappa)

        h[n+1,1:-1] = h[n,1:-1] -dt/dx*(f[0,1:]-f[0,:-1])
        hu[n+1,1:-1] = hu[n,1:-1] -dt/dx*(f[1,1:]-f[1,:-1])   
        
    return t,h,hu
#%%riemann solvers
import numpy as np
g = 9.81
def roe(hl,ul,hr,ur,i=None,kappa=1e-4):
    """
        El solver de roe del paper de Marche (2006?)
    """
    if np.isnan(hl) or np.isnan(hr):
        if not i : 
            print hl,hr
        else:
            print i,hl,hr,'nan'
        
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
    
#%% fluxes
    
def minmod(h1,h2,h3,dx):
  s1 = (h3-h2)/dx
  s2 = (h2-h1)/dx
  
  if s1 >0 and s2 >0:
      s = min(s1,s2)
  elif s1<0 and s2<0:
      s = max(s1,s2)
  else:
      s = 0.0
  h2l = h2-s*dx/2.
  h2R = h2+s*dx/2.
  
  return h2l,h2R
#%%
def fluxes(bcl, bcr, h, u, b, dx, hmin, rs):
  """    
    INPUT:
    bcl: (3x2 array) with ghost cells at the left boundary 
        (h,u,b) at each column. 
        bcl[:,0],has ghost cell -2 
        bcl[:,1], has ghost cell -1
    
    bcr: (2x3) array with ghost cells at the right boundary
        (h,u,b) at each column
        bcl[:,0],has ghost cell nx
        bcl[:,1], has ghost cell nx+1
    
    h,hu,b: ((nx,)  1darrays) conserved variables and topography inside the domain                   
    
    rs: riemann solver.

    Output:
        Fr
    domain: [-2,-1 | 0, 1, 2, 3, ..., nx-1 | nx, nx+1]
    
    
  """
  nx = h.shape[0]
  H = h+b
  Hl = np.zeros(nx)    
  Hr = np.zeros(nx)
  hl = np.zeros(nx)
  hr = np.zeros(nx)
  ul = np.zeros(nx)
  ur = np.zeros(nx)
  
  #1.---------MUSCL RECONSTRUCTION OF VARIABLES------
  
  # points 0 and nx-1    
  Hl[0],Hr[0] = minmod(bcl[0,1]+bcl[2,1], H[0], H[1],dx)    
  Hl[-1],Hr[-1] = minmod(H[-2], H[-1], bcr[0,0] + bcr[2,0],dx)
  
  hl[0],hr[0] = minmod(bcl[1,0], h[0], h[1],dx)    
  hl[-1],hr[-1] = minmod(h[-2], h[-1], bcr[0,0],dx)
  
  ul[0],ur[0] = minmod(bcl[1,1], u[0], u[1],dx)    
  ul[-1],ur[-1] = minmod(u[-2], u[-1], bcr[0,1],dx)
  
  #points 1 to nx-2
  for i in range(1,nx-2):
    Hl[i],Hr[i] =  minmod(H[i-1], H[i], H[i+1], dx)
    hl[i],hr[i] =  minmod(h[i-1], h[i], h[i+1], dx)
    ul[i],ur[i] =  minmod(u[i-1], u[i], u[i+1], dx)
  
  # reconstruction of topography
  bl = Hl - hl
  br = Hr - hr
  
  #boundaries
  
  Hol, Hor = minmod(bcl[0,0]+bcl[2,0], bcl[0,1]+bcl[2,1], H[0], dx)
  hol, hor = minmod(bcl[0,0], bcl[0,1], h[0], dx)
  uol, uor = minmod(bcl[1,0], bcl[1,1], u[0], dx)
  bor = Hor - hor
  
  Hnl, Hnr = minmod(H[-1], bcr[0,0]+bcr[2,0], bcr[0,1] + bcr[2,1], dx)
  hnl, hnr = minmod(h[-1], bcr[0,0], bcr[0,1], dx)
  unl, unr = minmod(u[-1], bcr[1,0], bcr[1,1], dx)
  bnl = Hnl - hnl
  
  #2. --------HYDROSTATIC RECONSTRUCTION -------------
  #reconstruct topography and water height at each interface
  #there are nx+1 interfaces in the domain

  bstar = np.zeros(nx+1)    
  hminus = np.zeros(nx+1) # hminus
  hplus = np.zeros(nx+1) # hplus
  
  bstar[0] = max(bor, bl[0])
  bstar[1:-1] = np.maximum(br[:-1],bl[1:])
  bstar[-1] = max(br[-1],bnl)
  
  hminus = np.maximum(hmin,  np.hstack([Hor, Hr])-bstar)
  hplus  = np.maximum(hmin,  np.hstack([Hl, Hnl])-bstar)
  
  #3. Solve the Riemann problem
  
  hstar = np.zeros(nx+1)
  ustar = np.zeros(nx+1)
  
  hstar[0], ustar[0] = rs(hminus[0], uor, hplus[0], ul[0])
  hstar[-1], ustar[-1] = rs(hminus[-1], ur[-1], hplus[-1], unl)
  
  #4. Reconstruct fluxes
  f0 = np.zeros((2,nx+1))
  fr = np.zeros((2,nx))    
  fl = np.zeros((2,nx))
  
  
  f0[0,:] = hstar*ustar
  f0[1,:] = hstar*ustar*ustar + 0.5*g*hstar*hstar
  
  fr[0,:] = f0[0,1:]
  fr[1,:] = f0[1,1:] + 0.5*g*(hr**2-hminus[1:]**2)
  
  fl[0,:] = f0[0,:-1]
  fl[1,:] = f0[1,:-1] + 0.5*g*(hl**2-hplus[:-1]**2)
  
  #5. add source terms
  s = np.zeros((2,nx))
  s[1,:] = 0.5*g*(hl+hr)*(bl-br)

  return fr, fl, s
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

import sys
sys.path.append('../')
sys.path.append('../nswe')

import numpy as np
import matplotlib.pyplot as plt
import nswe
import math
import muscl2
## Impose periodicity to ng ghost cells
def imposePeriodicity(v,ng) :
    for i in range(ng) :
        v[0+i] = v[-2*ng+i]
        v[-1-i] = v[2*ng-(i+1)]
    return v
## Create an array with nx points in [xmin,xmax)
def discretizeSpace(xmin,xmax,nx):
    dx = (xmax-xmin)/nx
    x = np.arange(xmin,xmax,dx)
    
    return x,dx
## Impose Robin conditions to FV scheme, 1st order
## u_{1/2} = (u_0 + u_1)/2
## u_x_{1/2} = (u_1 - u_0)/dx
def robinBC1(h,hu,BC,dx,t):
    
    ### Generic Robin conditions
    ### BC[0] = BC[4]*u + BC[5]*ux ## hLeft
    ### BC[1] = BC[6]*u + BC[7]*ux ## huLeft
    ### BC[2] = BC[8]*u + BC[9]*ux ## hRight
    ### BC[3] = BC[10]*u + BC[11]*ux ##huRight
    
    hb = 1.*h
    hub = 1.*hu
    
    denLh = BC[4]/2. - BC[5]/dx
    denLhu = BC[6]/2. - BC[7]/dx
    denRh = BC[8]/2. + BC[9]/dx
    denRhu = BC[10]/2. + BC[11]/dx
    
    hb[0] = (BC[0] - (BC[4]/2. + BC[5]/dx)*h[1])/denLh
    hub[0] = (BC[1] - (BC[6]/2. + BC[7]/dx)*hu[1])/denLhu
    hb[-1] = (BC[2] - (BC[8]/2. - BC[9]/dx)*h[-2])/denRh
    hub[-1] = (BC[3] - (BC[10]/2. - BC[11]/dx)*hu[-2])/denRhu  
    return hb,hub
## BC for a closed domain with 1 ghost cell
## The function robinBC1 can do the same with BC = [0.,0.,0.,0.,0.,1.,1.,0.,0.,1.,1.,0.]
def closedDomain(h,hu,t):
    hb = 1.*h
    hub = 1.*hu
    hb[0] = h[1]
    hub[0] = -hu[1]
    hb[-1] = h[-2]
    hub[-1] = -hu[-2]    
    return hb,hub
## BC for a closed domain with 2 ghost cells
## The function robinBC1 CANNOT do the same
def closedDomainTwoGC(h,hu,BC,dx,t):
    hb = 1.*h
    hub = 1.*hu
    hb[0] = h[2]
    hub[0] = -hu[2]
    hb[1] = h[2]
    hub[1] = -hu[2]
    
    hb[-1] = h[-3]
    hub[-1] = -hu[-3]
    hb[-2] = h[-3]
    hub[-2] = -hu[-3]    
    return hb,hub
## BC for an open domain
## The function robinBC1 can do the same with BC = [0.,0.,0.,0.,0.,1.,0.,1.,0.,1.,0.,1.]
def openDomain(h,hu,BC,dx,t):
    hb = 1.*h
    hub = 1.*hu
    hb[0] = h[1]
    hub[0] = hu[1]
    hb[-1] = h[-2]
    hub[-1] = hu[-2]    
    return hb,hub
## Impose periodic BC for 1 ghost cell in the FV scheme
## The function robinBC1 CANNOT do the same
def periodicDomain(h,hu,BC,dx,t):
    hb = 1.*h
    hub = 1.*hu
    hb[0] = h[-2]
    hub[0] = hu[-2]
    hb[-1] = h[1]
    hub[-1] = hu[1]    
    return hb,hub
## Impose periodic BC for 2 ghost cells in the FV scheme
## The function robinBC1 CANNOT do the same 
def periodicDomainTwoGC(h,hu,BC,dx,t):
    hb = 1.*h
    hub = 1.*hu
    
    hb[0] = h[-4]
    hub[0] = hu[-4]
    hb[-1] = h[3]
    hub[-1] = hu[3]
    
    hb[1] = h[-3]
    hub[1] = hu[-3]
    hb[-2] = h[2]
    hub[-2] = hu[2]
    
    return hb,hub
keepFirstSolL = 0
keepFirstSolR = 0

## Robin BC for the FD scheme (order 1, imposed in the center of the ghost cell)
## Terrible name :p Choose a better one, like robinBC2
def genericOpenDomain2(M,rhs,t,dx,BC):

    ### Boundary conditions
    ## Structure : BC=[u(left),ux(left),uxx(left),alpha1*u(left) + beta1*ux(right) + gamma1*uxx(right),
    ##                 u(right),ux(right),uxx(right),alpha2*u(right) + beta2*ux(right) + gamma2*uxx(right),
    ##                 alpha1,beta1,gamma1,alpha2,beta2,gamma2,Fleft,Fright] 
    cntBC = 0
    if not math.isnan(BC[0]) : # u L
        cntBC = cntBC+1
        M[0,:] = 0.
        M[0,0] = 1.
        if (not math.isnan(BC[14])) and (BC[14]!=keepFirstSolL) : # time-dependent function
            rhs[0] = BC[14](t)
        else:
            rhs[0] = BC[0]
    if not math.isnan(BC[1]) : # ux L
        cntBC = cntBC+1
        row = 0
        if not math.isnan(BC[0]):
            row = row + 1
        M[row,:] = 0.
        M[row,0] = -1./dx
        M[row,1] = 1./dx
        rhs[row] = BC[1]
    if not math.isnan(BC[2]) : # uxx L
        cntBC = cntBC+1
        row = 0
        if not math.isnan(BC[0]):
            row = row + 1
        if not math.isnan(BC[1]):
            row = row + 1
        M[row,:] = 0.
        M[row,0] = 1./(dx*dx)
        M[row,1] = -2./(dx*dx)
        M[row,2] = 1./(dx*dx)
        rhs[row] = BC[2]
    if not math.isnan(BC[3]) : # Robin L
        cntBC = cntBC+1
        if not (math.isnan(BC[0]) and math.isnan(BC[1]) and math.isnan(BC[2])) :
            sys.exit("Error in left BC : Robin defined with Dirichlet and/or Neumann")
        M[0,:] = 0.
        M[0,0] = BC[8] - BC[9]/dx + BC[10]/(dx*dx)
        M[0,1] = BC[9]/dx  - 2.*BC[10]/(dx*dx)
        M[0,2] = BC[10]/(dx*dx)
        rhs[0] = BC[3]
    if not math.isnan(BC[4]) : # u R
        cntBC = cntBC+1
        M[-1,:] = 0.
        M[-1,-1] = 1.
        if (not math.isnan(BC[15])) and (BC[15]!=keepFirstSolR) : # time-dependent function
            rhs[-1] = BC[15](t)
        else:
            rhs[-1] = BC[4]
    if not math.isnan(BC[5]) : # ux R
        cntBC = cntBC+1
        row = -1
        if not math.isnan(BC[4]):
            row = row - 1
        M[row,:] = 0.
        M[row,-1] = 1./dx
        M[row,-2] = -1./dx
        rhs[row] = BC[5]
    if not math.isnan(BC[6]) : # uxx R
        cntBC = cntBC+1
        row = -1
        if not math.isnan(BC[4]):
            row = row - 1
        if not math.isnan(BC[5]):
            row = row - 1
        M[row,:] = 0.
        M[row,-1] = 1./(dx*dx)
        M[row,-2] = -2. /(dx*dx)  
        M[row,-3] = 1./(dx*dx)
        rhs[row] = BC[6]
    if not math.isnan(BC[7]) : # Robin R
        cntBC = cntBC+1
        if not (math.isnan(BC[4]) and math.isnan(BC[5]) and math.isnan(BC[6])) :
            sys.exit("Error in right BC : Robin defined with Dirichlet and/or Neumann")
        M[-1,:] = 0.
        M[-1,-1] = BC[11] + BC[12]/dx + BC[13]/(dx*dx)
        M[-1,-2] = -BC[12]/dx - 2.*BC[13]/(dx*dx)
        M[-1,-3] = BC[13]/(dx*dx)
        rhs[-1] = BC[7]
     
    if cntBC != 3 :
        print(cntBC)
        print(BC)
        sys.exit("Wrong number of BC")
    
    return M,rhs
## Impose periodic BC for 1 ghost cell in the FD scheme
## The function genericOpenDomain2 (and equivalents) CANNOT do the same
def periodicDomain2(M,rhs,t,dx,BC):
    M[0,:] = 0.
    M[-1,:] = 0.
    
    M[0,0] = 1.
    M[0,-2] = -1.
    
    M[-1,-1] = 1.
    M[-1,1] = -1.
    
    rhs[0] = 0.
    rhs[-1] = 0.
    
    return M,rhs
## Impose periodic BC for 2 ghost cells in the FD scheme
## The function genericOpenDomain2 (and equivalents) CANNOT do the same
def periodicDomain2TwoGC(M,rhs,t,dx,BC):
    M[0,:] = 0.
    M[1,:] = 0.
    M[-1,:] = 0.
    M[-2,:] = 0.
    
    M[0,0] = 1.
    M[0,-4] = -1.
    
    M[1,1] = 1.
    M[1,-3] = -1.
    
    M[-1,-1] = 1.
    M[-1,3] = -1.
    
    M[-2,-2] = 1.
    M[-2,2] = -1.
    
    rhs[0] = 0.
    rhs[1] = 0.
    rhs[-1] = 0.
    rhs[-2] = 0.
    
    return M,rhs
# compute any of the RK4 coefficients (k_i)
def getRK4coef(x,uA,uB,f,dx,dt,nx):
    F = f(uA,uB,nx)
    return -dt/dx*(F[0,1:] - F[0,0:-1]), -dt/dx*(F[1,1:] - F[1,0:-1])

# complete the vector of RK4 coefficients with zeros in the ghost cells (to perform the sum u  + k_i)
def extend2GhostCells(v,ng):
    return np.concatenate((np.zeros(ng),v,np.zeros(ng)))

# RK4 for one time step
def RK4(x,uA,uB,f,bcf,bcp,dx,dt,nx,t,ng):

        uuA = np.copy(uA)
        uuB = np.copy(uB)
        uuA,uuB = bcf(uuA,uuB,bcp,dx,t)
        k1A,k1B = getRK4coef(x,uuA,uuB,f,dx,dt,nx)
        k1A = extend2GhostCells(k1A,ng)
        k1B = extend2GhostCells(k1B,ng)
        
        uuA = uA+k1A/2.
        uuB = uB+k1B/2.
        uuA,uuB = bcf(uuA,uuB,bcp,dx,t)
        k2A,k2B = getRK4coef(x,uuA,uuB,f,dx,dt,nx)
        k2A = extend2GhostCells(k2A,ng)
        k2B = extend2GhostCells(k2B,ng)
        
        uuA = uA+k2A/2.
        uuB = uB+k2B/2.
        uuA,uuB = bcf(uuA,uuB,bcp,dx,t)
        k3A,k3B = getRK4coef(x,uuA,uuB,f,dx,dt,nx)
        k3A = extend2GhostCells(k3A,ng)
        k3B = extend2GhostCells(k3B,ng)
        
        uuA = uA+k3A
        uuB = uB+k3B
        uuA,uuB = bcf(uuA,uuB,bcp,dx,t)
        k4A,k4B = getRK4coef(x,uuA,uuB,f,dx,dt,nx)
        k4A = extend2GhostCells(k4A,ng)
        k4B = extend2GhostCells(k4B,ng)

        return uA + 1./6.*(k1A+2.*k2A+2.*k3A+k4A), uB + 1./6.*(k1B+2.*k2B+2.*k3B+k4B)
    
# Euler for one time step
def Euler(x,uA,uB,f,bcf,bcp,dx,dt,nx,t,ng):

        uuA = np.copy(uA)
        uuB = np.copy(uB)
        uuA,uuB = bcf(uuA,uuB,bcp,dx,t)
        k1A,k1B = getRK4coef(x,uuA,uuB,f,dx,dt,nx)
        k1A = extend2GhostCells(k1A,ng)
        k1B = extend2GhostCells(k1B,ng)

        return uA + k1A, uB + k1B
# Compute first derivative
def get1d(u,dx,periodic,order):
    a = np.zeros_like(u)
    if order == 1:
        a[1:-1] = 1./(2.*dx)*(u[2:] - u[0:-2])
        a[0] = a[1]
        a[-1] = a[-2]
    elif order == 2:
        a[1:-1] = 1./(2.*dx)*(u[2:] - u[0:-2])
        a[0] = (-3*u[0] + 4.*u[1] - u[2])/(2.*dx)
        a[-1] = (3*u[-1] - 4.*u[-2] + u[-3])/(2.*dx)
    elif order == 4 :
        a[2:-2] = 1./(12.*dx)*(u[0:-4] - 8.* u[1:-3] + 8.*u[3:-1] - u[4:])
        #if (periodic) :
        #    a[0] = 1./(12.*dx)*(u[-3] - 8.* u[-2] + 8.*u[1] - u[2])
            ##### Todo Boundaries
            #a[1] = 1./(12.*dx)*(u[-2] - 8.* u[-1] + 8.*u[1] - u[2])
    return a

# Compute first derivative
def get2d(u,dx,periodic,order=2):
    a = np.zeros_like(u)
    a[1:-1] = 1./(dx*dx)*(u[2:] - 2.*u[1:-1] + u[0:-2])
    if order == 1:
        a[0] = a[1]
        a[-1] = a[-2]
    elif order == 2:
        if periodic :
            a[0] = 1./(dx*dx)*(u[1] - 2.*u[0] - u[-2])
            a[-1] = a[0]
        else :
            a[0] = 1./(dx*dx)*(2.*u[0] - 5.*u[1] + 4.*u[2] - u[3])
            a[-1] = 1./(dx*dx)*(2.*u[-1] - 5.*u[-2] + 4.*u[-3] - u[-4])
    elif order == 4 :
        a[2:-2] = 1./(12.*dx*dx)*(-u[0:-4] + 16.* u[1:-3] - 30*u[2:-2] + 16.*u[3:-1] - u[4:])
        #### TODO Boundaries
    return a

# Compute first derivative
def get3d(u,dx,periodic,order=2):
    a = np.zeros_like(u)
    a[2:-2] = 1./(2.*dx*dx)*(-u[0:-4] + 2.*u[1:-3] - 2.*u[3:-1] + u[4:])
    a[0] = 1./(dx*dx)*(-u[0] + 3.*u[1] - 3.*u[2] + u[3])
    a[1] = 1./(dx*dx)*(-u[1] + 3.*u[2] - 3.*u[3] + u[4])
    a[-1] = 1./(dx*dx)*(u[-1] - 3.*u[-2] + 3.*u[-3] - u[-4])
    a[-2] = 1./(dx*dx)*(u[-2] - 3.*u[-3] + 3.*u[-4] - u[-5])
    return a
def EFDSolver(h,u,dx,dt,t,order,BCfunction,BCparam=None,periodic=False,ng=2):
    
    """
    Finite Difference Solver for the second step of the splitted Serre equations.
    
    - Parameters
        * h,u (1D array) : solution
        * dx,dt,t (integers) : space step, time step, time
        * BCfunction (function) : function that modifies the linear system to impose the BCs
        * BCparam (1D array) : argument for BCfunction; contains the BCs in the form
             BC=[u(left),ux(left),uxx(left),alpha1*u(left) + beta1*ux(right) + gamma1*uxx(right),
                u(right),ux(right),uxx(right),alpha2*u(right) + beta2*ux(right) + gamma2*uxx(right),
                alpha1,beta1,gamma1,alpha2,beta2,gamma2,Fleft,Fright] 
        * periodic (boolean) : indicates if the function is periodic
        
    - Returns
        * u2 (1D array) : solution (velocity)
    """

    if periodic :
        for v in [u,h] :
            v = imposePeriodicity(v,ng)
    #h,u=periodicDomain(h,u,None,dx,t)

    ux = get1d(u,dx,periodic,order=2)
    uxx = get2d(u,dx,periodic,order=2)
    uxxx = get3d(u,dx,periodic,order=2)
    uux = u*ux
    uuxdx = get1d(uux,dx,periodic,order=2)
    hx = get1d(h,dx,periodic,order=2)
    h2x = get1d(h*h,dx,periodic,order=2)

    if periodic :
        for v in [ux,uux,uxx,uuxdx,h2x,hx] :
            v = imposePeriodicity(v,ng)
    
    g1 = h*h*h*(u*uxx - ux*ux)
    #g1 = h*h*h*(uuxdx - 2.*ux*ux)
    g1x = get1d(g1,dx,periodic,order=2)
    #g1x = 3.*h*h*hx*(u*uxx-ux*ux) + h*h*h*(-ux*uxx + u*uxxx)
    
    
    ################
    #g1x = 3.*h*h*hx*(u*uxx-ux*ux) + h*h*h*(ux*uxx + u*uxxx - 2.*ux*uxx)    
    #################
    
    g2 = u - 1./2.*h2x*ux - h*h*uxx/3.
       
    d0 = 1. + 2./(3.*dx*dx)*h*h
    dp1 = -h2x/(4.*dx) - h*h/(3.*dx*dx)
    dp1 = dp1[0:-1]
    dm1 = h2x/(4.*dx) - h*h/(3.*dx*dx)
    dm1 = dm1[1:]
    
    M = np.diag(d0) + np.diag(dp1,1) + np.diag(dm1,-1)

    M[0,:] = 0
    M[-1,:] = 0
    
    M[0,0] = 1. - 3./4.*h2x[0]/dx - 2./3.*h[0]*h[0]/(dx*dx)
    M[0,1] = h2x[0]/dx + 5./3.*h[0]*h[0]/(dx*dx)
    M[0,2] = -1./4.*h2x[0]/dx - 4./3.*h[0]*h[0]/(dx*dx)
    M[0,3] = 1./3.*h[0]*h[0]/(dx*dx)

    M[-1,-1] = 1. + 3./4.*h2x[-1]/dx - 2./3.*h[-1]*h[-1]/(dx*dx)
    M[-1,-2] = -h2x[-1]/dx + 5./3.*h[-1]*h[-1]/(dx*dx)
    M[-1,-3] = 1./4.*h2x[-1]/dx - 4./3.*h[-1]*h[-1]/(dx*dx)
    M[-1,-4] = 1./3.*h[-1]*h[-1]/(dx*dx)


    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(suppress=True)

    
    rhs = g2 + dt/(3.*h)*g1x
    
    if BCparam != None:
        if BCparam[14] == keepFirstSolL :
            BCparam[0] = u[0]
        if BCparam[15] == keepFirstSolL :
            BCparam[4] = u[-1]    
    
    M,rhs = BCfunction(M,rhs,t,dx,BCparam)
    
    u2 = np.linalg.solve(M,rhs)
        
    return u2
def EFDSolverFM4(h,u,dx,dt,t,order,BCfunction,BCparam=None,periodic=False,ng=2,eta=0.):
    
    """
    Finite Difference Solver for the second step of the splitted Serre equations, using the discretization derived
    in the paper of Fabien Marche
    
    - Parameters
        * h,u (1D array) : solution
        * dx,dt,t (integers) : space step, time step, time
        * BCfunction (function) : function that modifies the linear system to impose the BCs
        * BCparam (1D array) : argument for BCfunction; contains the BCs in the form
             BC=[u(left),ux(left),uxx(left),alpha1*u(left) + beta1*ux(right) + gamma1*uxx(right),
                u(right),ux(right),uxx(right),alpha2*u(right) + beta2*ux(right) + gamma2*uxx(right),
                alpha1,beta1,gamma1,alpha2,beta2,gamma2,Fleft,Fright] 
        * periodic (boolean) : indicates if the function is periodic
        
    - Returns
        * u2 (1D array) : solution (velocity)
    """
    
    gr = 9.81

    if periodic :
        for v in [u,h] :
            v = imposePeriodicity(v,ng)
    
    hu = h*u
    
    ux = get1d(u,dx,periodic,order=4)
    uxx = get2d(u,dx,periodic,order=4)
    uux = u*ux
    uuxdx = get1d(uux,dx,periodic,order=4)
    hx = get1d(h,dx,periodic,order=4)
    hxx = get2d(h,dx,periodic,order=4)
    h2x = get1d(h*h,dx,periodic,order=4)
    hhx = h*hx
    
    Q = 2.*h*hx*ux*ux + 4./3.*h*h*ux*uxx
    rhs = gr*h*hx + h*Q  

    if periodic :
        for v in [ux,uux,uxx,uuxdx,h2x,hx,hhx] :
            v = imposePeriodicity(v,ng)   
    
    d0 = 1. + hx*hx/3. + h*hxx/3. + 5.*h*h/(6.*dx*dx)
    dp1 = -2./3.*h*hx/(3.*dx) - 4./3.*h*h/(3.*dx*dx)
    dp1 = dp1[0:-1]
    dm1 = +2./3.*h*hx/(3.*dx) - 4./3.*h*h/(3.*dx*dx)
    dm1 = dm1[1:]
    dp2 = 1./3.*h*hx/(12.*dx) + 1./3.*h*h/(12.*dx*dx)
    dp2 = dp2[0:-2]
    dm2 = -1./3.*h*hx/(12.*dx) + 1./3.*h*h/(12.*dx*dx)
    dm2 = dm2[2:]
    
    M = np.diag(d0) + np.diag(dp1,1) + np.diag(dm1,-1) + np.diag(dp2,2) + np.diag(dm2,-2)

    M[0,:] = 0
    M[1,:] = 0
    M[-1,:] = 0
    M[-2,:] = 0

    ### Correct it (but in general these lines are replaced by the BC)
    M[0,0] = h[0]*(1. - 3./4.*h2x[0]/dx - 2./3.*h[0]*h[0]/(dx*dx))
    M[0,1] = h[0]*(h2x[0]/dx + 5./3.*h[0]*h[0]/(dx*dx))
    M[0,2] = h[0]*(-1./4.*h2x[0]/dx - 4./3.*h[0]*h[0]/(dx*dx))
    M[0,3] = h[0]*(1./3.*h[0]*h[0]/(dx*dx))

    M[1,1] = h[1]*(1. - 3./4.*h2x[1]/dx - 2./3.*h[1]*h[1]/(dx*dx))
    M[1,2] = h[1]*(h2x[1]/dx + 5./3.*h[1]*h[1]/(dx*dx))
    M[1,3] = h[1]*(-1./4.*h2x[1]/dx - 4./3.*h[1]*h[1]/(dx*dx))
    M[1,4] = h[1]*(1./3.*h[1]*h[1]/(dx*dx))    
    
    M[-1,-1] = h[-1]*(1. + 3./4.*h2x[-1]/dx - 2./3.*h[-1]*h[-1]/(dx*dx))
    M[-1,-2] = h[-1]*(-h2x[-1]/dx + 5./3.*h[-1]*h[-1]/(dx*dx))
    M[-1,-3] = h[-1]*(1./4.*h2x[-1]/dx - 4./3.*h[-1]*h[-1]/(dx*dx))
    M[-1,-4] = h[-1]*(1./3.*h[-1]*h[-1]/(dx*dx))
      
    M[-2,-2] = h[-2]*(1. + 3./4.*h2x[-2]/dx - 2./3.*h[-2]*h[-2]/(dx*dx))
    M[-2,-3] = h[-2]*(-h2x[-2]/dx + 5./3.*h[-2]*h[-2]/(dx*dx))
    M[-2,-4] = h[-2]*(1./4.*h2x[-2]/dx - 4./3.*h[-2]*h[-2]/(dx*dx))
    M[-2,-5] = h[-2]*(1./3.*h[-2]*h[-2]/(dx*dx))
    ######

    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(suppress=True)

    if BCparam != None:
        if BCparam[14] == keepFirstSolL :
            BCparam[0] = u[0]
        if BCparam[15] == keepFirstSolL :
            BCparam[4] = u[-1]    
    
    M,rhs = BCfunction(M,rhs,t,dx,BCparam)

    z = np.linalg.solve(M,rhs)
    hu2 = hu + dt*(gr*h*hx-z)

    return hu2/h
def EFDSolverFM4Bottom(h,u,dx,dt,t,order,BCfunction,BCparam=None,periodic=False,ng=2,eta=0.):
    
    """
    Finite Difference Solver for the second step of the splitted Serre equations,
    with a flart but non necessarily horizontal bottom, using the discretization derived
    in the paper of Fabien Marche
    
    - Parameters
        * h,u (1D array) : solution
        * dx,dt,t (integers) : space step, time step, time
        * BCfunction (function) : function that modifies the linear system to impose the BCs
        * BCparam (1D array) : argument for BCfunction; contains the BCs in the form
             BC=[u(left),ux(left),uxx(left),alpha1*u(left) + beta1*ux(right) + gamma1*uxx(right),
                u(right),ux(right),uxx(right),alpha2*u(right) + beta2*ux(right) + gamma2*uxx(right),
                alpha1,beta1,gamma1,alpha2,beta2,gamma2,Fleft,Fright] 
        * periodic (boolean) : indicates if the function is periodic
        * eta : slope of the bottom
        
    - Returns
        * u2 (1D array) : solution (velocity)
    """
    
    gr = 9.81

    if periodic :
        for v in [u,h] :
            v = imposePeriodicity(v,ng)
    
    hu = h*u
    
    ux = get1d(u,dx,periodic,order=4)
    uxx = get2d(u,dx,periodic,order=4)
    uux = u*ux
    uuxdx = get1d(uux,dx,periodic,order=4)
    hx = get1d(h,dx,periodic,order=4)
    hxx = get2d(h,dx,periodic,order=4)
    h2x = get1d(h*h,dx,periodic,order=4)
    hhx = h*hx
    
    Q = 2.*h*hx*ux*ux + 4./3.*h*h*ux*uxx + eta*eta*h*u*ux + eta*eta*(hx+eta)*u*u
    rhs = gr*h*hx + h*Q + gr*h*eta  

    if periodic :
        for v in [ux,uux,uxx,uuxdx,h2x,hx,hhx] :
            v = imposePeriodicity(v,ng)   
    
    d0 = 1. + hx*hx/3. + h*hxx/3. + 5.*h*h/(6.*dx*dx) + eta*(hx+eta)
    dp1 = -2./3.*h*hx/(3.*dx) - 4./3.*h*h/(3.*dx*dx)
    dp1 = dp1[0:-1]
    dm1 = +2./3.*h*hx/(3.*dx) - 4./3.*h*h/(3.*dx*dx)
    dm1 = dm1[1:]
    dp2 = 1./3.*h*hx/(12.*dx) + 1./3.*h*h/(12.*dx*dx)
    dp2 = dp2[0:-2]
    dm2 = -1./3.*h*hx/(12.*dx) + 1./3.*h*h/(12.*dx*dx)
    dm2 = dm2[2:]
    
    M = np.diag(d0) + np.diag(dp1,1) + np.diag(dm1,-1) + np.diag(dp2,2) + np.diag(dm2,-2)

    M[0,:] = 0
    M[1,:] = 0
    M[-1,:] = 0
    M[-2,:] = 0

    ### Correct it (but in general these lines are replaced by the BC)
    M[0,0] = h[0]*(1. - 3./4.*h2x[0]/dx - 2./3.*h[0]*h[0]/(dx*dx))
    M[0,1] = h[0]*(h2x[0]/dx + 5./3.*h[0]*h[0]/(dx*dx))
    M[0,2] = h[0]*(-1./4.*h2x[0]/dx - 4./3.*h[0]*h[0]/(dx*dx))
    M[0,3] = h[0]*(1./3.*h[0]*h[0]/(dx*dx))

    M[1,1] = h[1]*(1. - 3./4.*h2x[1]/dx - 2./3.*h[1]*h[1]/(dx*dx))
    M[1,2] = h[1]*(h2x[1]/dx + 5./3.*h[1]*h[1]/(dx*dx))
    M[1,3] = h[1]*(-1./4.*h2x[1]/dx - 4./3.*h[1]*h[1]/(dx*dx))
    M[1,4] = h[1]*(1./3.*h[1]*h[1]/(dx*dx))    
    
    M[-1,-1] = h[-1]*(1. + 3./4.*h2x[-1]/dx - 2./3.*h[-1]*h[-1]/(dx*dx))
    M[-1,-2] = h[-1]*(-h2x[-1]/dx + 5./3.*h[-1]*h[-1]/(dx*dx))
    M[-1,-3] = h[-1]*(1./4.*h2x[-1]/dx - 4./3.*h[-1]*h[-1]/(dx*dx))
    M[-1,-4] = h[-1]*(1./3.*h[-1]*h[-1]/(dx*dx))
      
    M[-2,-2] = h[-2]*(1. + 3./4.*h2x[-2]/dx - 2./3.*h[-2]*h[-2]/(dx*dx))
    M[-2,-3] = h[-2]*(-h2x[-2]/dx + 5./3.*h[-2]*h[-2]/(dx*dx))
    M[-2,-4] = h[-2]*(1./4.*h2x[-2]/dx - 4./3.*h[-2]*h[-2]/(dx*dx))
    M[-2,-5] = h[-2]*(1./3.*h[-2]*h[-2]/(dx*dx))
    ######

    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(suppress=True)

    if BCparam != None:
        if BCparam[14] == keepFirstSolL :
            BCparam[0] = u[0]
        if BCparam[15] == keepFirstSolL :
            BCparam[4] = u[-1]    
    
    M,rhs = BCfunction(M,rhs,t,dx,BCparam)

    z = np.linalg.solve(M,rhs)
    hu2 = hu + dt*(gr*h*(hx+eta)-z)

    return hu2/h
def EFDSolverFM(h,u,dx,dt,t,order,BCfunction,BCparam=None,periodic=False,ng=2):
    
    """
    Finite Difference Solver for the second step of the splitted Serre equations, using the discretization derived
    in the paper of Fabien Marche
    
    - Parameters
        * h,u (1D array) : solution
        * dx,dt,t (integers) : space step, time step, time
        * BCfunction (function) : function that modifies the linear system to impose the BCs
        * BCparam (1D array) : argument for BCfunction; contains the BCs in the form
             BC=[u(left),ux(left),uxx(left),alpha1*u(left) + beta1*ux(right) + gamma1*uxx(right),
                u(right),ux(right),uxx(right),alpha2*u(right) + beta2*ux(right) + gamma2*uxx(right),
                alpha1,beta1,gamma1,alpha2,beta2,gamma2,Fleft,Fright] 
        * periodic (boolean) : indicates if the function is periodic
        
    - Returns
        * u2 (1D array) : solution (velocity)
    """
    
    gr = 9.81

    if periodic :
        for v in [u,h] :
            v = imposePeriodicity(v,ng)
    
    hu = h*u
    
    ux = get1d(u,dx,periodic,order=2)
    uxx = get2d(u,dx,periodic,order=2)
    uux = u*ux
    uuxdx = get1d(uux,dx,periodic,order=2)
    hx = get1d(h,dx,periodic,order=2)
    hxx = get2d(h,dx,periodic,order=2)
    h2x = get1d(h*h,dx,periodic,order=2)
    hhx = h*hx
    
    Q = 2.*h*hx*ux*ux + 4./3.*h*h*ux*uxx
    rhs = gr*h*hx + h*Q  

    if periodic :
        for v in [ux,uux,uxx,uuxdx,h2x,hx,hhx] :
            v = imposePeriodicity(v,ng)   
    
    d0 = 1. + hx*hx/3. + h*hxx/3. + 2.*h*h/(3.*dx*dx)
    dp1 = -h*hx/(3.*2.*dx) - h*h/(3.*dx*dx)
    dp1 = dp1[0:-1]
    dm1 = h*hx/(3.*2.*dx) - h*h/(3.*dx*dx)
    dm1 = dm1[1:]
    
    M = np.diag(d0) + np.diag(dp1,1) + np.diag(dm1,-1)

    M[0,:] = 0
    M[-1,:] = 0

    ### Correct it (but in general these lines are replaced by the BC)
    M[0,0] = h[0]*(1. - 3./4.*h2x[0]/dx - 2./3.*h[0]*h[0]/(dx*dx))
    M[0,1] = h[0]*(h2x[0]/dx + 5./3.*h[0]*h[0]/(dx*dx))
    M[0,2] = h[0]*(-1./4.*h2x[0]/dx - 4./3.*h[0]*h[0]/(dx*dx))
    M[0,3] = h[0]*(1./3.*h[0]*h[0]/(dx*dx))

    M[-1,-1] = h[-1]*(1. + 3./4.*h2x[-1]/dx - 2./3.*h[-1]*h[-1]/(dx*dx))
    M[-1,-2] = h[-1]*(-h2x[-1]/dx + 5./3.*h[-1]*h[-1]/(dx*dx))
    M[-1,-3] = h[-1]*(1./4.*h2x[-1]/dx - 4./3.*h[-1]*h[-1]/(dx*dx))
    M[-1,-4] = h[-1]*(1./3.*h[-1]*h[-1]/(dx*dx))
    ######

    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(suppress=True)

    if BCparam != None:
        if BCparam[14] == keepFirstSolL :
            BCparam[0] = u[0]
        if BCparam[15] == keepFirstSolL :
            BCparam[4] = u[-1]    
    
    M,rhs = BCfunction(M,rhs,t,dx,BCparam)

    z = np.linalg.solve(M,rhs)
    hu2 = hu + dt*(gr*h*(hx+eta)-z)

    return hu2/h
# solve the Serre equation
def splitSerre(x,h,u,t0,tmax,bcfunction1,bcfunction2,bcparam1,bcparam2,dx,nx,vardt = True, dt = 0.05,
               splitSteps = 3, periodic=False,order=2,fvsolver=muscl2.fluxes2,fvTimesolver=RK4,fdsolver=EFDSolver,
              ghostcells = 2,eta=0.):
    t = t0
    grav = 9.8
    
    uall = u
    hall = h
    tall = np.ones(1)*t0

    print(r'CFL = %f' %(dt/(dx*dx*dx)))
    
    while t < tmax and dt > 1e-9:
        if vardt :
            if (np.amax(np.absolute(u)) + np.sqrt(grav*np.amax(h))) > 1.e-6:
                dt = dx/(np.amax(np.absolute(u)) + np.sqrt(grav*np.amax(h)))
            print(r'dt = %f; t = %f' %(dt,t))
        t = t+dt
        
        hu = h*u

        h,hu = bcfunction1(h,hu,bcparam1,dx,t)
        #F = nswe.fluxes(h,hu,nx)
        #h[1:-1] = h[1:-1] - dt/dx*(F[0,1:] - F[0,0:-1])
        #hu[1:-1] = hu[1:-1] - dt/dx*(F[1,1:] - F[1,0:-1])
        
        if splitSteps == 3: ## Adv Disp Adv
            h,hu = fvTimesolver(x,h,hu,fvsolver,bcfunction1,bcparam1,dx,dt/2.,nx,t,ng=ghostcells)
            u = np.where(h[:]>1e-5, hu[:]/h[:], 0.)
            u = fdsolver(h,u,dx,dt,t,order,bcfunction2,bcparam2,periodic=periodic,ng=ghostcells)
            hu = h*u
            h,hu = bcfunction1(h,hu,bcparam1,dx,t)  
            h,hu = fvTimesolver(x,h,hu,fvsolver,bcfunction1,bcparam1,dx,dt/2.,nx,t,ng=ghostcells)
            u = np.where(h[:]>1e-5, hu[:]/h[:], 0.)
        elif splitSteps == 2 : ## Adv Disp
            h,hu = fvTimesolver(x,h,hu,fvsolver,bcfunction1,bcparam1,dx,dt,nx,t,ng=ghostcells)
            u = np.where(h[:]>1e-5, hu[:]/h[:], 0.)
            u = fdsolver(h,u,dx,dt,t,order,bcfunction2,bcparam2,periodic,ng=ghostcells,eta=eta)
        elif splitSteps == -2 : ## Disp Adv
            u = np.where(h[:]>1e-5, hu[:]/h[:], 0.)
            u = fdsolver(h,u,dx,dt,t,order,bcfunction2,bcparam2,periodic,ng=ghostcells)            
            hu = h*u
            h,hu = bcfunction1(h,hu,bcparam1,dx,t)  
            h,hu = fvTimesolver(x,h,hu,fvsolver,bcfunction1,bcparam1,dx,dt/2.,nx,t,ng=ghostcells)
            u = np.where(h[:]>1e-5, hu[:]/h[:], 0.)
        elif splitSteps == -3 : ## Adv Disp
            u = np.where(h[:]>1e-5, hu[:]/h[:], 0.)
            u = fdsolver(h,u,dx,dt/2.,t,order,bcfunction2,bcparam2,periodic,ng=ghostcells)            
            hu = h*u
            h,hu = bcfunction1(h,hu,bcparam1,dx,t)  
            h,hu = fvTimesolver(x,h,hu,fvsolver,bcfunction1,bcparam1,dx,dt/2.,nx,t,ng=ghostcells)
            u = np.where(h[:]>1e-5, hu[:]/h[:], 0.)
            u = fdsolver(h,u,dx,dt/2.,t,order,bcfunction2,bcparam2,periodic,ng=ghostcells)

        hall = np.column_stack((hall,h))
        uall = np.column_stack((uall,u))
        tall = np.hstack((tall,t*np.ones(1)))
        
    return hall,uall,tall
## solve only the NSWE part (for comparison)
def NSWE(x,h,u,t0,tmax,bcfunction,bcparam,dx,nx,vardt = True, dt = 0.05, fvsolver=muscl2.fluxes2,fvTimesolver=RK4,
         ghostcells = 2):
    t = t0
    grav = 9.8
    
    uall = u
    hall = h
    tall = np.ones(1)*t0

    print(r'CFL = %f' %(dt/(dx*dx*dx)))
    
    while t < tmax and dt > 1e-9:
        if vardt:
            if (np.amax(np.absolute(u)) + np.sqrt(grav*np.amax(h))) > 1.e-6:
                dt = dx*dx*dx/(np.amax(np.absolute(u)) + np.sqrt(grav*np.amax(h)))
            print(r'dt = %f; t = %f' %(dt,t))
        t = t+dt
      
        hu = h*u

        h,hu = bcfunction(h,hu,bcparam,dx,t)
        #F = nswe.fluxes(h,hu,nx)
        #h[1:-1] = h[1:-1] - dt/dx*(F[0,1:] - F[0,0:-1])
        #hu[1:-1] = hu[1:-1] - dt/dx*(F[1,1:] - F[1,0:-1])
        #F = muscl2.fluxes2(h,hu,nx)
        #h[1:-1] = h[1:-1] - dt/dx*(F[0,1:] - F[0,0:-1])
        #hu[1:-1] = hu[1:-1] - dt/dx*(F[1,1:] - F[1,0:-1])
        #h[2:-2] = h[2:-2] -dt/dx*(F[0,1:] - F[0,:-1])
        #hu[2:-2] = hu[2:-2] -dt/dx*(F[1,1:] - F[1,:-1])
        
        h,hu = fvTimesolver(x,h,hu,fvsolver,bcfunction,bcparam,dx,dt,nx,t,ghostcells)
            
        u = np.where(h[:]>1e-5, hu[:]/h[:], 0.)
        
        #u = EFDSolver(h,u,dx,dt)

        hall = np.column_stack((hall,h))
        uall = np.column_stack((uall,u))
        tall = np.hstack((tall,t*np.ones(1)))
        
    return hall,uall,tall
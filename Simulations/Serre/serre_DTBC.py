## Numerical resolution of the Serre equations

import sys
sys.path.append('../')
sys.path.append('../nswe')

import numpy as np
import matplotlib.pyplot as plt
import math
import generalFunctions as gF
import cnoidal
import nswe
import muscl2

def imposePeriodicity(v,ng) :
    """
    Impose periodicity to ng ghost cells
    """
    
    for i in range(ng) :
        v[0+i] = v[-2*ng+i]
        v[-1-i] = v[2*ng-(i+1)]
    return v
def discretizeSpace(xmin,xmax,nx):
    """
    Create an array with nx points in [xmin,xmax)
    """
    
    dx = (xmax-xmin)/nx
    x = np.arange(xmin,xmax,dx)
    
    return x,dx

def discretizeSpaceFull(xmin,xmax,nx):
    """
    Create an array with nx points in [xmin,xmax]
    """
    
    x = np.linspace(xmin,xmax,nx)
    dx = np.diff(x)[0]
    
    return x,dx
def openDomain0GC(h,hu,BC,dx,t,ref=[],idx=[]):
    """
    BC for an open domain with 0 ghost cells
    """
    
    return h,hu
def periodicDomainTwoGC(h,hu,BC,dx,t):
    """
    Impose periodic BC for 2 ghost cells in the FV scheme
    """

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
import nswe_wbmuscl4 as wb4

def fluxes_periodic(h,hu,n,periodic,ng,u_refRK=[],h_refRK=[],idx=[]):
    """
    Creates the fluxes obtaines from a 4th order MUSCL reconstruction.
    Note that the MUSCL's stencil is such that we need 6 ghost cells, 3 on each side. 
    Hence we need to add these values. For a periodic scheme this is not too difficult, but for the 
    small domain, which is not periodic, we need to save the value of the reference solution 
    on the corresponding nodes. This is what is done here. with the arrays refRK_save.
    """
    
    if u_refRK == [] and h_refRK == []:
        u_refRK_save = np.zeros(6)
        h_refRK_save = np.zeros(6)
    else:
        u_refRK_save = []
        h_refRK_save = []
            
    ## first, we remove the ghost cells
    nx = h.shape[0]-2*ng
    ## then we add 3 cells on each side for the muscl scheme
    h0 = np.zeros(nx+6)
    u0 = np.zeros(nx+6)
    d0 = np.zeros(nx+6)
    h0[3:-3] = h[ng:len(h)-ng]
    u0[3:-3] = hu[ng:len(hu)-ng]
    u0 = np.where(h0>1e-10,u0/h0,0) #hu/h
    u = np.where(h>1e-10,hu/h,0)
    
    if idx != []:
        h0[:3] = h0[-6:-3]
        h0[-3:] = h0[3:6]
        u0[:3] = u0[-6:-3]
        u0[-3:] = u0[3:6]
        ## saving the reference
        idx1 = idx[0]
        idx2 = idx[1]
        u_refRK_save = np.append(u[idx1-3:idx1], u[idx2:idx2+3])
        h_refRK_save = np.append(h[idx1-3:idx1], h[idx2:idx2+3])
        
    elif u_refRK != [] and h_refRK != []:
        h0[:3]  = h_refRK[:3]
        h0[-3:] = h_refRK[-3:]
        u0[:3]  = u_refRK[:3]
        u0[-3:] = u_refRK[-3:]
        
    else:
        h0[:3]  = h0[3:6]
        h0[-3:] = h0[-6:-3]
        u0[:3]  = u0[3:6]
        u0[-3:] = u0[-6:-3]
    
    fp, fm, sc = wb4.fluxes_sources(d0,h0,u0)
    return fp, u_refRK_save, h_refRK_save
def getRK4coef(uA,uB,f,dx,dt,nx,periodic,ng,u_refRK=[],h_refRK=[],idx=[]):
    """
    compute any of the RK4 coefficients (k_i)
    """
    
    F, u_refRK_save, h_refRK_save = f(uA,uB,nx,periodic,ng,u_refRK,h_refRK,idx)
    return -dt/dx*(F[0,1:] - F[0,:-1]), -dt/dx*(F[1,1:] - F[1,:-1]), u_refRK_save, h_refRK_save

def extend2GhostCells(v,ng):
    """
    complete the vector of RK4 coefficients with zeros in the ghost cells (to perform the sum u  + k_i)
    """
    
    return np.concatenate((np.zeros(ng),v,np.zeros(ng)))

def RK4(uA,uB,f,bcf,bcp,dx,dt,nx,t,periodic,ng,u_refRK=[],h_refRK=[],idx=[]):
    """
    RK4 for one time step
    """
    
    #### for the small domain, we need to impose the value of the bigdomain functions 
    #### on the 6 cells used for the MUSCL scheme
    #### ---> we are storing them in refRK_save
    u_refRK_save = np.zeros((4,6))
    h_refRK_save = np.zeros((4,6))
        
    uuA = np.copy(uA)
    uuB = np.copy(uB)
    if u_refRK == [] and h_refRK == []:
        k1A,k1B,u_refRK_save[0],h_refRK_save[0] = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng,idx=idx)
    else:
        k1A,k1B,trash1,trash2 = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng,u_refRK[0],h_refRK[0])
    k1A = extend2GhostCells(k1A,ng)
    k1B = extend2GhostCells(k1B,ng)

    uuA = uA+k1A/2.
    uuB = uB+k1B/2.
    if u_refRK == [] and h_refRK == []:
        k2A,k2B,u_refRK_save[1],h_refRK_save[1] = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng,idx=idx)
    else:
        k2A,k2B,trash1,trash2 = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng,u_refRK[1],h_refRK[1])
    k2A = extend2GhostCells(k2A,ng)
    k2B = extend2GhostCells(k2B,ng)

    uuA = uA+k2A/2.
    uuB = uB+k2B/2.
    if u_refRK == [] and h_refRK == []:
        k3A,k3B,u_refRK_save[2],h_refRK_save[2] = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng,idx=idx)
    else:
        k3A,k3B,trash1,trash2 = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng,u_refRK[2],h_refRK[2])
    k3A = extend2GhostCells(k3A,ng)
    k3B = extend2GhostCells(k3B,ng)

    uuA = uA+k3A
    uuB = uB+k3B
    if u_refRK == [] and h_refRK == []:
        k4A,k4B,u_refRK_save[3],h_refRK_save[3] = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng,idx=idx)
    else:
        k4A,k4B,trash1,trash2 = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng,u_refRK[3],h_refRK[3])
    k4A = extend2GhostCells(k4A,ng)
    k4B = extend2GhostCells(k4B,ng)

    return uA + 1./6.*(k1A+2.*k2A+2.*k3A+k4A), uB + 1./6.*(k1B+2.*k2B+2.*k3B+k4B), u_refRK_save, h_refRK_save
def periodicDomain2TwoGC(M,rhs,t,dx,BC):
    """
    Impose periodic BC for 2 ghost cells in the FD scheme
    """
    
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
import convolution as cvl

def imposeBC_FV(M,rhs,BCs,h,u,hx,hu,dx,dt,nit,Y=[],uall=None):
    """
    Impose boundary conditions for the dispersive part
    
    - Inputs :
        * M : matrix of the FD scheme
        * rhs : right-hand side of the FD scheme
        * BCs : array containing one BC in each line, in the form
            [Position,Type,Value], where
                ::: Position (int) : indicates the point to be modified (0,1,...,-2,-1)
                ::: Type (str) : indicates the type of BC : "Dirichlet"/"Neumann"/"Robin"/"DTBC"
                ::: Value (float) : value of the BC
        * h,hx,hu : informations from the last computation
        * dt, dx : time and space steps
        
    - Outputs :
        * M, modified for the BC
        * rhs, modified for the BC
    """
    
    gr = 9.81
    
    ## first loop to compute DTBC convolution parameters only once
    convol = False
    for i in range(BCs.shape[0]):
        [pos,typ,val] = BCs[i,:3]
        if typ == 'DTBC_Y':
            convol = True
            Ct = cvl.convolution_exact(nit, Y, uall)
            # uu = (1./hp1)*(h*u + dt*gr*h*hx) 
            uu = u + dt*gr*hx
            break
        
    ## impose BCs
    for i in range(BCs.shape[0]) :
        [pos,typ,val] = BCs[i,:3]
        pos = int(pos)
        val = float(val)
        
        if typ == "Dirichlet" :
            M[pos,:] = 0.
            M[pos,pos] = 1.
            rhs[pos] = -(val*h[pos]-hu[pos] - dt*gr*h[pos]*hx[pos])/dt
            
        elif typ == "Neumann" :
            M[pos,:] = 0.
            if pos == 0:
                M[0,0] = -h[1]
                M[0,1] = h[0]
                rhs[0] = h[0]*h[1]/dt*(u[1]-u[0] + dt*gr*(hx[1]-hx[0]) - val*dx)
            else:
                M[pos,pos] = h[pos-1]
                M[pos,pos-1] = -h[pos]
                rhs[pos] = h[pos]*h[pos-1]/dt*(u[pos]-u[pos-1] + dt*gr*(hx[pos]-hx[pos-1]) - val*dx)
                
        elif typ == "Robin" :
            alpha = float(BCs[i,3])
            beta = float(BCs[i,4])
            M[pos,:] = 0.
            if pos == 0 or pos == -2 :
                M[pos,pos] = dt*h[pos+1]*(alpha*dx - beta)
                M[pos,pos+1] = beta*dt*h[pos]
                rhs[pos] = h[pos]*h[pos+1]*(\
                                    alpha*dx*(u[pos]+dt*gr*hx[pos]) + \
                                    beta*(u[pos+1] - u[pos] + dt*gr*(hx[pos+1]-hx[pos])) - dx*val)
            elif pos == 1 or pos == -1 :
                M[pos,pos] = dt*h[pos-1]*(alpha*dx + beta)
                M[pos,pos-1] = -beta*dt*h[pos]
                rhs[pos] = h[pos]*h[pos-1]*(\
                                            alpha*dx*(u[pos]+dt*gr*hx[pos]) + \
                                            beta*(u[pos] - u[pos-1] + dt*gr*(hx[pos]-hx[pos-1])) - dx*val)
                            
        elif typ == "DTBC_Y":
            
            M[pos,:] = 0.
            
            if pos == 0:
                # Left TBC 1 ==> unknown = U[0]
                M[pos,0]   =  1.
                M[pos,1]   = -   Y[4,0]*h[0]/h[1]
                M[pos,2]   =     Y[6,0]*h[0]/h[2]
                val        = Ct[4,1] - Ct[6,2]
                rhs[pos]   = -(h[0]/dt)*( val - uu[0] + Y[4,0]*uu[1] - Y[6,0]*uu[2] )                
            elif pos == 1:
                # Left TBC 2 ==> unknown = U[1]
                M[pos,0]   =  1.
                M[pos,2]   = -   Y[5,0]*h[0]/h[2]
                M[pos,3]   =  2.*Y[8,0]*h[0]/h[3]
                M[pos,4]   = -   Y[7,0]*h[0]/h[4]
                val        = Ct[5,2] - 2*Ct[8,3] + Ct[7,4]
                rhs[pos]   = -(h[0]/dt)*( val - uu[0] + Y[5,0]*uu[2] - 2*Y[8,0]*uu[3] + Y[7,0]*uu[4] ) 
            elif pos == -1:
                ## Right TBC 1 ==> unknown = U[J]
                M[pos,-1]  =  1.
                M[pos,-2]  = -   Y[0,0]*h[-1]/h[-2]
                M[pos,-3]  =     Y[2,0]*h[-1]/h[-3]
                val        =  Ct[0,-2] - Ct[2,-3]
                rhs[pos]   = -(h[-1]/dt)*( val - uu[-1] + Y[0,0]*uu[-2] - Y[2,0]*uu[-3] )
            elif pos == -2:
                ## Right TBC 2 ==> unknown = U[J-1]
                M[pos,-1] =  1.
                M[pos,-2] = -2.*Y[0,0]*h[-1]/h[-2]
                M[pos,-3] =     Y[1,0]*h[-1]/h[-3]
                M[pos,-5] = -   Y[3,0]*h[-1]/h[-5]
                val       = 2*Ct[0,-2] - Ct[1,-3] + Ct[3,-5]
                rhs[pos]  = -(h[-1]/dt)*( val - uu[-1] + 2*Y[0,0]*uu[-2] - Y[1,0]*uu[-3] + Y[3,0]*uu[-5] )    
                
        else :
            sys.exit("Wrong type of TBC!! Please use Dirichlet/Neumann/Robin/DTBC_Y")
        
    if convol:
        return M,rhs,Ct
    else:
        return M,rhs,[]
def get1d(u,dx,periodic,order):
    """
    Compute first derivative
    """
    
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

def get2d(u,dx,periodic,order=2):
    """
    Compute second derivative
    """
    
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

def get3d(u,dx,periodic,order=2):
    """
    Compute third derivative
    """
    
    a = np.zeros_like(u)
    a[2:-2] = 1./(2.*dx*dx)*(-u[0:-4] + 2.*u[1:-3] - 2.*u[3:-1] + u[4:])
    a[0] = 1./(dx*dx)*(-u[0] + 3.*u[1] - 3.*u[2] + u[3])
    a[1] = 1./(dx*dx)*(-u[1] + 3.*u[2] - 3.*u[3] + u[4])
    a[-1] = 1./(dx*dx)*(u[-1] - 3.*u[-2] + 3.*u[-3] - u[-4])
    a[-2] = 1./(dx*dx)*(u[-2] - 3.*u[-3] + 3.*u[-4] - u[-5])
    return a
def EFDSolverFM4(h,u,dx,dt,t,order,BCfunction,BCparam=None,periodic=False,ng=2,Y=[],nit=0,uall=None,debug=False):
    """
    Finite Difference Solver for the second step of the splitted Serre equations, using the discretization derived
    in the paper of Fabien Marche.
    
    - Parameters
        * h,u (1D array) : solution
        * dx,dt,t : space step, time step, time
        * BCfunction (function) : function that modifies the linear system to impose the BCs
        * BCparam (1D array) : argument for BCfunction
        * periodic (boolean) : indicates if the function is periodic
        * ng : number of ghostcells
        * Y : convolution coefficients
        * nit : current iteration
        * uall : solution at previous iterations to compute convolution
        * debug : if True, increase verbosity
        
    - Returns
        * hu2/h (1D array) : solution (velocity)
    """
    
    gr = 9.81

    if periodic :
        for v in [u,h] :
            v = imposePeriodicity(v,ng)
    
    hu = h*u
    
    order = 2
    
    ux = get1d(u,dx,periodic,order=order)
    uxx = get2d(u,dx,periodic,order=order)
    uux = u*ux
    uuxdx = get1d(uux,dx,periodic,order=order)
    hx = get1d(h,dx,periodic,order=order)
    hxx = get2d(h,dx,periodic,order=order)
    h2x = get1d(h*h,dx,periodic,order=order)
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

    np.set_printoptions(threshold=np.nan)
        
    if BCfunction != imposeBC_FV:
        M,rhs = BCfunction(M,rhs,t,dx,BCparam)
    else:
        M,rhs,Ct = BCfunction(M,rhs,BCparam,h,u,hx,hu,dx,dt,nit,Y,uall)
            
    z = np.linalg.solve(M,rhs)
    hu2 = hu + dt*(gr*h*hx-z)
    
    if Y != [] and debug:
        ## check the numerical validation of the DTBC
        u2 = hu2/h
        
        print " *  Left"
        print u2[0] - Y[4,0]*u2[1] - Ct[4,1] +   Y[6,0]*u2[2] +   Ct[6,2]
        print u2[0] - Y[5,0]*u2[2] - Ct[5,2] + 2*Y[8,0]*u2[3] + 2*Ct[8,3] - Y[7,0]*u2[4] - Ct[7,4] 
        
        print " *  Right"
        print u2[-1] -   Y[0,0]*u2[-2] -   Ct[0,-2] + Y[2,0]*u2[-3] + Ct[2,-3] 
        print u2[-1] - 2*Y[0,0]*u2[-2] - 2*Ct[0,-2] + Y[1,0]*u2[-3] + Ct[1,-3] - Y[3,0]*u2[-5] - Ct[3,-5]

    return hu2/h
def splitSerre(x,h,u,t0,tmax,bcfunction1,bcfunction2,bcparam1,bcparam2,dx,dt,nx,
               splitSteps = 3, periodic=False,order=2,
               fvsolver=fluxes_periodic,fvTimesolver=RK4,fdsolver=EFDSolverFM4,
               ghostcells = 2,Y=[],debug=False,
               u_refRK=[],h_refRK=[], idx=[]):
    """
    Solve the Serre equations with a splitting scheme.
    
    - Parameters 
        * x : spatial domain of resolution
        * h,u : initial solution
        * t0, tmax : initial and final times
        * bcfunction, bcparam : boundary conditions functions and parameters for the advection (1) 
                                and the dispersion (2)
        * dx, dt : space and time steps
        * nx : number of nodes
        * splitSteps : how to perform the splitting, 2 --> adv + disp, 3 --> 0.5*adv + disp + 0.5*adv (Strang)
        * periodic : has to be True if the bcfunctions are periodic BC
        * order : order of the fdsolver
        * fvsolver : fluxes builder fo the advection
        * fvTimesolver : time step of the advection
        * fdsovler : FD solver
        * ghostcells : number of ghostcells
        * Y : convolution coefficient for the DTBC
        * debug : True to increase verbosity of the DTBC
        * u_refRK, h_refRK, idx : reference values for the 4 steps of the RK4 time step, along with idx of the node 
                                  to save in the case of the reference solution
        
    - Returns
        * hall,uall,tall : h, u arrays over time and the time array
        * u_refRK_save,h_refRK_save : if not empty, value of the reference solution on the corresponding nodes to 
                                      impose boundary conditions in the advection step of the small domain solution
    """
    
    t = t0
    it = 0
    grav = 9.8
    ng = ghostcells
    
    uall = u
    hall = h
    tall = np.ones(1)*t0

    print(r'CFL = %f' %(dt/(dx*dx*dx)))
    
    u_refRK_save = []
    h_refRK_save = []
    
    while t < tmax:
            
        if u_refRK == [] and h_refRK == []:
            u_refRK_it = []
            h_refRK_it = []
        else:
            u_refRK_it = u_refRK[it]
            h_refRK_it = h_refRK[it]
        
        hu = h*u
        h,hu = bcfunction1(h,hu,bcparam1,dx,t)
        
        if splitSteps == 3: ## Adv Disp Adv
            ## 0.5*adv
            h,hu,u_refRK_temp,h_refRK_temp = fvTimesolver(h,hu,fvsolver,bcfunction1,bcparam1,dx,dt/2.,nx,t,periodic,
                                                          ng=ghostcells,u_refRK=u_refRK_it,h_refRK=h_refRK_it,idx=idx)
            h,hu = bcfunction1(h,hu,bcparam1,dx,t)
            u = np.where(h[:]>1e-10, hu[:]/h[:], 0.)
            
            ## disp
            u = fdsolver(h,u,dx,dt,t,order,bcfunction2,bcparam2,
                         periodic=periodic,ng=ghostcells,Y=Y,nit=it+1,uall=uall,debug=debug)
            hu = h*u
            h,hu = bcfunction1(h,hu,bcparam1,dx,t)
            
            ## 0.5*adv
            h,hu,u_refRK_temp,h_refRK_temp = fvTimesolver(h,hu,fvsolver,bcfunction1,bcparam1,dx,dt/2.,nx,t,periodic,
                                                          ng=ghostcells,u_refRK=u_refRK_it,h_refRK=h_refRK_it,idx=idx)
            h,hu = bcfunction1(h,hu,bcparam1,dx,t)
            u = np.where(h[:]>1e-10, hu[:]/h[:], 0.)
            
        elif splitSteps == 2 : ## Adv Disp
            ## adv
            h,hu,u_refRK_temp,h_refRK_temp = fvTimesolver(h,hu,fvsolver,bcfunction1,bcparam1,dx,dt,nx,t,periodic,
                                                          ng=ghostcells,u_refRK=u_refRK_it,h_refRK=h_refRK_it,idx=idx)
            h,hu = bcfunction1(h,hu,bcparam1,dx,t)
            u = np.where(h[:]>1e-10, hu[:]/h[:], 0.)
            
            ## disp
            u = fdsolver(h,u,dx,dt,t,order,bcfunction2,bcparam2,
                         periodic=periodic,ng=ghostcells,Y=Y,nit=it+1,uall=uall,debug=debug)

        ## saving references in the big domain case
        if u_refRK == [] and h_refRK == []:
            u_refRK_save.append(u_refRK_temp)
            h_refRK_save.append(h_refRK_temp)
            
        ## next time        
        t = t+dt
        it += 1

        ## saving for plotting
        hall = np.column_stack((hall,h))
        uall = np.column_stack((uall,u))
        tall = np.hstack((tall,t*np.ones(1)))
                    
    return hall,uall,tall,u_refRK_save,h_refRK_save
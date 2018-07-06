
import sys
sys.path.append('../')
sys.path.append('../nswe')

import numpy as np
import matplotlib.pyplot as plt
import serre
import cnoidal
import nswe_wbmuscl4 as wb4


nan = float("nan")
def periodicSubDomain_1_TwoGC(h,hu,BC,dx,t):
    """
    Boundary conditions for the left subdomain, with two ghostcells on the left for periodicity.
    """
    
    hb = 1.*h
    hub = 1.*hu
    
    hb[0] = BC[0,-2]
    hub[0] = BC[1,-2]   
    hb[1] = BC[0,-1]
    hub[1] = BC[1,-1]
    
    return hb,hub

def periodicSubDomain_2_TwoGC(h,hu,BC,dx,t):
    """
    Boundary conditions for the right subdomain, with two ghostcells on the right for periodicity.
    """
    
    hb = 1.*h
    hub = 1.*hu
    
    hb[-1] = BC[0,1]
    hub[-1] = BC[1,1]   
    hb[-2] = BC[0,0]
    hub[-2] = BC[1,0]
    
    return hb,hub

def impose_periodicity_2subdom(a1,b1,a2,b2):
    """
    Impose periodicity once the solution from the advection step have been computed in each domain 
    by the MUSCL scheme.
    """
    a1b = np.copy(a1)
    b1b = np.copy(b1)
    a2b = np.copy(a2)
    b2b = np.copy(b2)
    
    a1b[:2] = a2[-4:-2]
    b1b[:2] = b2[-4:-2]
    
    a2b[-2:] = a1[2:4]
    b2b[-2:] = b1[2:4]                  
                  
    return a1b,b1b,a2b,b2b
    
def extend2GhostCells(v,ng):
    """
    complete the vector of RK4 coefficients with zeros in the ghost cells 
    (to perform the sum u  + k_i)
    """
    return np.concatenate((np.zeros(ng),v,np.zeros(ng)))

def extend2GhostCells_right(v,ng):
    """
    complete the vector with ng ghost cells on the right
    (to perform the sum u  + k_i in RK4)
    """
    return np.concatenate((v,np.zeros(ng)))

def restrict2GhostCells_right(v,ng):
    """
    remove the ng ghost cells on the right
    """
    return v[:len(v)-ng]

def extend2GhostCells_left(v,ng):
    """
    complete the vector with ng ghost cells on the left
    (to perform the sum u  + k_i in RK4)
    """
    return np.concatenate((np.zeros(ng),v))

def restrict2GhostCells_left(v,ng):
    """
    remove the ng ghost cells on the left
    """
    return v[ng:]
import nswe_wbmuscl4 as wb4

def fluxes_periodic(h,hu,n,periodic,ng):
    """
    Finite volume solver for the monodomain. For the three ghost cells necessary to the MUSCL scheme,
    we use periodic conditions. Moreover, we save values at the interface for the debugging mode of the DDM. 
    """
            
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
    
    if periodic:
        h0[:3] = h0[-6:-3]
        h0[-3:] = h0[3:6]
        u0[:3] = u0[-6:-3]
        u0[-3:] = u0[3:6]

    else:
        h0[:3]  = h0[3:6]
        h0[-3:] = h0[-6:-3]
        u0[:3]  = u0[3:6]
        u0[-3:] = u0[-6:-3]
    
    fp, fm, sc = wb4.fluxes_sources(d0,h0,u0)
    
    return fp
# compute any of the RK4 coefficients (k_i)
def getRK4coef(uA,uB,f,dx,dt,nx,periodic,ng):
    F = f(uA,uB,nx,periodic,ng)
    return -dt/dx*(F[0,1:] - F[0,:-1]), -dt/dx*(F[1,1:] - F[1,:-1])

# RK4 for one time step
def RK4(uA,uB,f,bcf,bcp,dx,dt,nx,t,periodic,ng,u_refRK=[],h_refRK=[],idx=[]):
        
    uuA = np.copy(uA)
    uuB = np.copy(uB)
    k1A,k1B = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng)
    k1A = extend2GhostCells(k1A,ng)
    k1B = extend2GhostCells(k1B,ng)

    uuA = uA+k1A/2.
    uuB = uB+k1B/2.
    k2A,k2B = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng)
    k2A = extend2GhostCells(k2A,ng)
    k2B = extend2GhostCells(k2B,ng)

    uuA = uA+k2A/2.
    uuB = uB+k2B/2.
    k3A,k3B = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng)
    k3A = extend2GhostCells(k3A,ng)
    k3B = extend2GhostCells(k3B,ng)

    uuA = uA+k3A
    uuB = uB+k3B
    k4A,k4B = getRK4coef(uuA,uuB,f,dx,dt,nx,periodic,ng)
    k4A = extend2GhostCells(k4A,ng)
    k4B = extend2GhostCells(k4B,ng)

    uuA = uA + 1./6.*(k1A+2.*k2A+2.*k3A+k4A)
    uuB = uB + 1./6.*(k1B+2.*k2B+2.*k3B+k4B)
        
    ## [] are for serre.splitSerre, but we don't need them here
    return uuA, uuB, [], []
def imposeBCDispersive(M,rhs,BCs,h,u,hx,hu,dx,dt,Y=[],eta=0.,hp1=[],inter=None):
    
    """
    Impose three boundary conditions for the dispersive part
    
    - Inputs :
        * M : matrix of the FD scheme
        * rhs : right-hand side of the FD scheme
        * BCs : array of dimensions 3x3 containing one TBC in each line, in the form
            [Position,Type,Value,Opt], where
                ::: Position (int) : indicates the point to be modified (0,1,...,-2,-1)
                ::: Type (str) : indicates the type of BC : "Dirichlet"/"Neumann"/"TBC"
                ::: Value (float) : value of the BC
                ::: Opt [int,float,array] : optional coefficients for the TBC; depends on the Type 
        * h,hx,hu : informations from the last computation
        * dt
        * hp1 : h from the next iteration
        
    - Outputs :
        * M
        * rhs
    """
    gr = 9.81
    
    ### verif number of TBCs
    #if BCs.shape[0] != 3 :
    #    sys.exit("Wrong number of BCs")
        
    ## impose BCs
    for i in range(BCs.shape[0]) :
        [pos,typ,val] = BCs[i,:3]
        pos = int(pos)
        val = float(val)
        if typ == "Dirichlet" or typ == "periodic" :
            M[pos,:] = 0.
            M[pos,pos] = 1.
            rhs[pos] = -(val*h[pos]-hu[pos] - dt*gr*h[pos]*hx[pos])/dt
                            
        elif typ == "DTBC_Y":
            
            M[pos,:] = 0.
            
            if pos == 0:
                # Left TBC 1 ==> unknown = U[0]
                M[0,0]   =  1.
                M[0,1]   = -   Y[4,0]*h[0]/h[1]
                M[0,2]   =     Y[6,0]*h[0]/h[2]
                rhs[pos] = val           
            elif pos == 1:
                # Left TBC 2 ==> unknown = U[1]
                M[1,0]   =  1.
                M[1,2]   = -   Y[5,0]*h[0]/h[2]
                M[1,3]   =  2.*Y[8,0]*h[0]/h[3]
                M[1,4]   = -   Y[7,0]*h[0]/h[4]
                rhs[pos] = val
            elif pos == -1:
                ## Right TBC 1 ==> unkonwn = U[J]
                M[-1,-1] =  1.
                M[-1,-2] = -   Y[0,0]*h[-1]/h[-2]
                M[-1,-3] =     Y[2,0]*h[-1]/h[-3]
                rhs[pos] = val
            elif pos == -2:
                ## Right TBC 2 ==> unknown = U[J-1]
                M[-2,-1] =  1.
                M[-2,-2] = -2.*Y[0,0]*h[-1]/h[-2]
                M[-2,-3] =     Y[1,0]*h[-1]/h[-3]
                M[-2,-5] = -   Y[3,0]*h[-1]/h[-5]
                rhs[pos] = val
                    
        else :
            sys.exit("Wrong type of TBC!! Please use Dirichlet/Neumann/TBC")
        
    return M,rhs
def EFDSolverFM4(h,u,dx,dt,order,BCs,it,periodic=False,ng=2,side="left",href=None,uref=None,Y=[],
                 domain=0,ind=0,zref=None):
    
    """
    Finite Difference Solver for the second step of the splitted Serre equations, using the discretization derived
    in the paper of Fabien Marche
    MODIFICATION : imposition of BCs
    
    - Parameters
        * h,u (1D array) : solution
        * dx,dt,t (integers) : space step, time step, time
        * BCfunction (function) : function that modifies the linear system to impose the BCs
        * BCparam (1D array) : argument for BCfunction; contains the BCs in the form
             BC=[u(left),ux(left),uxx(left),alpha1*u(left) + beta1*ux(right) + gamma1*uxx(right),
                u(right),ux(right),uxx(right),alpha2*u(right) + beta2*ux(right) + gamma2*uxx(right),
                alpha1,beta1,gamma1,alpha2,beta2,gamma2,Fleft,Fright] 
        * periodic (boolean) : indicates if the function is periodic
        * ind : index to restrain to the given subdomain
        
    - Returns
        * u2 (1D array) : solution (velocity)
    """
    
    gr = 9.81
    
    if periodic :
        for v in [u,h] :
            v = serre.imposePeriodicity(v,ng)    

    hu = h*u
        
    order = 2
    
    ux = serre.get1d(u,dx,periodic,order=order)
    uxx = serre.get2d(u,dx,periodic,order=order)
    uux = u*ux
    uuxdx = serre.get1d(uux,dx,periodic,order=order)
    hx = serre.get1d(h,dx,periodic,order=order)
    hxx = serre.get2d(h,dx,periodic,order=order)
    h2x = serre.get1d(h*h,dx,periodic,order=order)
    hhx = h*hx
    
    Q = 2.*h*hx*ux*ux + 4./3.*h*h*ux*uxx
    rhs = gr*h*hx + h*Q   
    
    if periodic :
        for v in [ux,uux,uxx,uuxdx,h2x,hx,hhx,hxx] :
            v = serre.imposePeriodicity(v,ng)  
    
    if domain == 1:
        u = u[:ind]
        ux = ux[:ind]
        uxx = uxx[:ind]
        uux = uux[:ind]
        uuxdx = uuxdx[:ind]
        h = h[:ind]        
        hx = hx[:ind]
        hxx = hxx[:ind]
        h2x = h2x[:ind]
        hhx = hhx[:ind]
        hu = h*u
        Q = Q[:ind]
        rhs = rhs[:ind]
    elif domain == 2:
        u = u[ind:]
        ux = ux[ind:]
        uxx = uxx[ind:]
        uux = uux[ind:]
        uuxdx = uuxdx[ind:]
        h = h[ind:]        
        hx = hx[ind:]
        hxx = hxx[ind:]
        h2x = h2x[ind:]
        hhx = hhx[ind:]
        hu = h*u  
        Q = Q[ind:]
        rhs = rhs[ind:]
     
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
    
    M,rhs = imposeBCDispersive(M,rhs,BCs,h,u,hx,hu,dx,dt,Y=Y)    
    z = np.linalg.solve(M,rhs)
    
    hu2 = hu + dt*(gr*h*hx-z)
    
    return hu2/h, z
def norm2(u, dx):
  """
  Return the l^2 norm of an numpy array.
  """
  return np.sqrt(dx*np.sum(u**2))


def splitSerreDDM(x,u,h,t0,tmax,dt,dx,nx,cond_int_1,cond_int_2,cond_bound,periodic=True,
                  bcfunction_adv=serre.periodicDomainTwoGC,
                  uref=None,href=None,zref=None,debug_1=False,debug_2=False,Y=[],
                  ng=3,fvTimesolver=RK4):
    """
    If the DDM is overlapping : N1+N2 >= N+2, otherwise N1+N2 = N+1.

                   0 1 2                                                    N-2 N-1=J
    Monodomain   = [ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ]
                   0 1 2             |         N1-2 N1-1=J_1
    Left Domain  = [ - - - - - - - - - - - - - - - ]
                                     | 1 2         |                       N2-2 N2-1=J_2
    Right Domain =                   [ - - - - - - - - - - - - - - - - - - - - ]
                                     |             |
    Index on the monodomain   :     N-N2          N1-1
    Index on the left domain  :     N-N2          N1-1
    Index on the right domain :      0          N1+N2-N-1

                                    O12            J21
                                    
    Arguments:
    ----------
    - x : domain of computation
    - u,h : unknowns of the Serre equations
    - t0, tmax : starting and stopping times
    - dt, dx : time and space steps
    - nx : unknowns in the monodomain
    - cond_int_1, cond_int_2 : conditions at the interface between the two domains
    - cond_bound : conditions at the boundaries of the mono-domain
    - uref, href, zref : references values (mono-domain)
    - Y : convolution coefficients for the discrete TBC
    - debug_1, debug_2 : if True, we impose the monodomain solution on the boundaries of the subdomain i
    - ng : number of ghostcells (3 for the advection part)
    - fvTimesolver : solver for the advection part
    - u_refRK, h_refRK : references for the RK part (necessary for debug mode)
    """
    
    ## time steps
    t = t0
    it = 0     
    
    ## parameters for the DDM
    n = nx
    assert nx == len(u)
    j = n-1
    n1 = int((4./5.)*n)
    j1 = n1-1
    # minimal overlapping for our TBC : n = n1+n2-5
    n2 = n-n1+5
    # n2 = int((3./4.)*n)
    j2 = n2-1
    
    ## communication between domains
    # last node of the left domain in the right domain
    j21 = n1+n2-n-1
    # first node of the right domain in the left domain
    o12 = n-n2
    
    ## decomposition
    u1 = u[:n1]
    u2 = u[o12:]
    
    ## store solutions of all timesteps
    uall = np.copy(u)
    u1all = np.copy(u1)
    u2all = np.copy(u2)
    tall = np.ones(1)*t0
    
    ## precision
    nitermax = 300
    eps = 10**(-15)

    print "*** starting DDM resolution with {} - {} at the interface".format(cond_int_1, cond_int_2)
    print " * "
    print " *  precision = {:.3e}".format(eps)
    print " * "
    
    while abs(t-tmax) > 10**(-12):
        
        ## starting from the reference
        h = href[:,it]
        u = uref[:,it]
        hu = h*u
            
        ## advection on the mono domain
        # no need to decompose into domains as the scheme is explicit
        h,hu,trash1,trash2 = fvTimesolver(h,hu,fluxes_periodic,bcfunction_adv,None,dx,dt,nx,t,periodic,
                                          ng=ng)
        h,hu = bcfunction_adv(h,hu,None,dx,t)
            
        ## retrieving u and decomposing
        u = hu/h
        u1 = u[:n1]
        h1 = h[:n1]
        u2 = u[o12:]
        h2 = h[o12:]
        
        ## starting the Schwarz algorithm
        cvg = False
        niter = 0
        # for the z part of the dispersion 
        # (it is the variable that is going to vary during the DDM, cf. reports from Joao)
        z1 = np.zeros_like(u1)
        z2 = np.zeros_like(u2)
        
        ## monitoring error
        if it == 50:
            monitor = True
        else:
            monitor = False
        if monitor:
            err_tab = []
        
        print " *  --------------------------"
        print " *  t = {:.2f}".format(t+dt)
        print " *  Advection error for h :", np.sqrt(norm2(h1-href[:n1,it+1], dx)**2 + 
                                                     norm2(h2-href[o12:,it+1], dx)**2)
                    
        ## order for derivatives in the dispersive equation
        FDorder = 2
        
        ## DDM for the dispersive part
        while niter < nitermax and cvg == False:
                        
            ## \Omega_1 : left --> BC, right --> IBC
            if debug_1:
                cond_int_1 = "Dirichlet"
                val11 = uref[j1,it+1]
                val12 = uref[j1-1,it+1]
            elif cond_int_1 == "Dirichlet":
                val11 = u2[j21]
                val12 = u2[j21-1]
            elif cond_int_1 == "DTBC_Y":
                val11 = z2[j21] -   Y[0,0]*(h[j1]/h[j1-1])*z2[j21-1] \
                                +   Y[2,0]*(h[j1]/h[j1-2])*z2[j21-2]
                val12 = z2[j21] - 2*Y[0,0]*(h[j1]/h[j1-1])*z2[j21-1] \
                                +   Y[1,0]*(h[j1]/h[j1-2])*z2[j21-2] \
                                -   Y[3,0]*(h[j1]/h[j1-4])*z2[j21-4]
            else:
                val11 = 0.
                val12 = 0.

            if debug_1 or periodic:
                bc11 = uref[0,it+1]
                bc12 = uref[1,it+1]  
            else:
                bc11 = 0.
                bc12 = 0.
                
            BCconfig1 = np.array([[0,cond_bound,bc11,1.,0.,1.],
                                 [-1,cond_int_1,val11,1.,0.,1.],
                                 [1,cond_bound,bc12,1.,0.,1.],
                                 [-2,cond_int_1,val12,1.,0.,1.]], dtype=object)
            
            ## solving in the left domain
            u1_save = np.copy(u1)
            z1_save = np.copy(z1)
            u1,z1 = EFDSolverFM4(h,u,dx,dt,FDorder,BCconfig1,it,
                                 Y=Y,domain=1,ind=n1,periodic=periodic)
            assert(len(u1) == n1)
            
            ## \Omega_2 : left --> IBC, right --> BC
            if debug_2:
                cond_int_2 = "Dirichlet"
                val21 = uref[o12,it+1]
                val22 = uref[o12+1,it+1]
            elif cond_int_2 == "Dirichlet":
                val21 = u1_save[o12]
                val22 = u1_save[o12+1]
            elif cond_int_2 == "DTBC_Y":
                val21 = z1_save[o12] -   Y[4,0]*(h[o12]/h[o12+1])*z1_save[o12+1] \
                                     +   Y[6,0]*(h[o12]/h[o12+2])*z1_save[o12+2]
                val22 = z1_save[o12] -   Y[5,0]*(h[o12]/h[o12+2])*z1_save[o12+2] \
                                     + 2*Y[8,0]*(h[o12]/h[o12+3])*z1_save[o12+3] \
                                     -   Y[7,0]*(h[o12]/h[o12+4])*z1_save[o12+4]
            else:
                val21 = 0.
                val22 = 0.
                
            if debug_2 or periodic:
                bc21 = uref[-1,it+1]
                bc22 = uref[-2,it+1]    
            else:
                bc21 = 0.
                bc22 = 0.
                
            BCconfig2 = np.array([[0,cond_int_2,val21,1.,0.,1.],
                                 [-1,cond_bound,bc21,1.,0.,1.],
                                 [1,cond_int_2,val22,1.,0.,1.],
                                 [-2,cond_bound,bc22,1.,0.,1.]], dtype=object)      
            
            ## solving in the right domain
            u2_save = np.copy(u2)
            z2_save = np.copy(z2)
            u2,z2 = EFDSolverFM4(h,u,dx,dt,FDorder,BCconfig2,it,
                                 Y=Y,domain=2,ind=o12,periodic=periodic)
            assert(len(u2) == n2)
            
            ## periodicity
            h1u1 = h1*u1
            h2u2 = h2*u2
            if periodic:
                h1,h1u1,h2,h2u2 = impose_periodicity_2subdom(h1,h1u1,h2,h2u2)
            
            ## test convergence with reference to uref
            ## convergence in u
            err_norm_ref = np.sqrt(norm2(u1-uref[:n1,it+1], dx)**2 + norm2(u2-uref[o12:,it+1], dx)**2)
            ## convergence in z
            # err_norm_ref = np.sqrt(norm2(z1-zref[:n1,it], dx)**2 + norm2(z2-zref[o12:,it], dx)**2)
            ## convergence error instead of reference (for when we don't know the reference solution)
            err_norm_cvg = np.sqrt(norm2(u1-u1_save, dx)**2 + norm2(u2-u2_save, dx)**2)
            
            ## choose which error to consider
            err_norm = err_norm_ref
            
            ## monitoring error
            if monitor:
                err_tab.append(err_norm)
            
            niter += 1
            
            ## if convergence
            if err_norm < eps:
                print " *  DDM cvg reached in {:4d} iterations, error = {:.3e}".format(niter, err_norm)
                print " *  left domain interface  : {:.3e}".format(np.sqrt((u1[j1]-uref[j1,it+1])**2 +
                                                                           (u1[j1-1]-uref[j1-1,it+1])**2))
                print " *  right domain interface : {:.3e}".format(np.sqrt((u2[0]-uref[o12,it+1])**2 +
                                                                           (u2[1]-uref[o12+1,it+1])**2))
                print " * "
                cvg = True
                        
            ## if not convergence after nitermax iterations
            if niter == nitermax:
                print " *  DDM cvg not reached after {:4d} iterations, error = {:.3e}".format(niter,err_norm)
                print " *  left domain interface  : {:.3e}".format(np.sqrt((u1[j1]-uref[j1,it+1])**2 + 
                                                                          (u1[j1-1]-uref[j1-1,it+1])**2))
                print " *  right domain interface : {:.3e}".format(np.sqrt((u2[0]-uref[o12,it+1])**2 + 
                                                                          (u2[1]-uref[o12+1,it+1])**2))

                print " * "
        
        ## building ddm solution for plotting
        u[:o12] = u1[:o12]
        u[o12:n1] = .5*(u1[o12:] + u2[:j21+1])
        u[n1:] = u2[j21+1:]
                
        ## monitoring error
        err1 = u1-uref[:n1,it+1]
        err2 = u2-uref[o12:,it+1]
        err1 = np.append(err1,np.zeros(n-n1))
        err2 = np.append(np.zeros(n-n2),err2)
        if monitor:
            plt.plot(err_tab)
            plt.yscale('log')
            plt.savefig('error_{}-{}.pdf'.format(cond_int_1, cond_int_2))
            plt.clf()
            
        ## stacking after convergence
        try:
            z1all = np.column_stack((z1all,z1))
            z2all = np.column_stack((z2all,z2))
        except:
            z1all = z1
            z2all = z2
        u1all = np.column_stack((u1all,u1))
        u2all = np.column_stack((u2all,u2))
        uall = np.column_stack((uall,u))
        tall = np.hstack((tall,t*np.ones(1)))
        try:
            err1all = np.column_stack((err1all,err1))
            err2all = np.column_stack((err2all,err2))
        except:
            err1all = np.copy(err1)
            err2all = np.copy(err2)
        
        ## next time step
        t  += dt
        it += 1
        
    print "*** DDM over"
    
    # saving interfaces for plotting
    ddm = [x[o12],x[j1]]
        
    return uall,u1all,u2all,z1all,z2all,tall,ddm,err1all,err2all
def computeErrorTBC(u,uref,idxlims,dx,dt):
    lim1 = idxlims[0]
    lim2 = idxlims[1]
    uwind = uref[lim1:lim2+1,:]
    errDom = np.linalg.norm(u-uwind)*np.sqrt(dx*dt)
    errInt1 = np.linalg.norm(u[0,:]-uwind[0,:])*np.sqrt(dt)
    errInt2 = np.linalg.norm(u[-1,:]-uwind[-1,:])*np.sqrt(dt)
    
    return errDom,errInt1,errInt2
    
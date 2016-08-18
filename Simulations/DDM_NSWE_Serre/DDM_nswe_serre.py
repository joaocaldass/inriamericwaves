
import sys
sys.path.append('../Serre')
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import generalFunctions as gF
import muscl2
import nswe_wbmuscl4 as wb4
import serre
import serreTBC
import cnoidal
def openDomainThreeGC(h,hu,BC,dx,t):
    """
    *Function to be passed as "bcfunction" argument for DDM_NSWE
    *Impose classical open boundary conditions in a domain with three ghost cells in each side (necessary for O4).
    *The domain is supposed to be [-L-3*dx, L+3*dx]
    
    *Inputs
        * h, hu : variables to impose the BCs (can be any variable)
        * BC, dx, t : unused in this function
        
    * Outputs :
        * hb,hub : h,hu with BCs
    """
    
    hb = 1.*h
    hub = 1.*hu
    
    hb[0] = h[3]
    hub[0] = hu[3]
    hb[1] = h[3]
    hub[1] = hu[3]
    hb[2] = h[3]
    hub[2] = hu[3]
    
    hb[-1] = h[-4]
    hub[-1] = hu[-4]    
    hb[-2] = h[-4]
    hub[-2] = hu[-4]
    hb[-3] = h[-4]
    hub[-3] = hu[-4]
    
    return hb,hub
def DDM_BCs(h,hu,BC,dx,t):
    """
    *Function to be passed as "bcfunction" argument for DDM_NSWE
    *Impose user-defined Dirichlet boundary conditions for NSWE solver
    
    *Inputs
        * h, hu : variables to impose the BCs (can be any variable)
        * BC : array with lines in the form [pos,valh, valhu] :
            * pos : position for imposing the boundary condition (0,1,2,...,-3,-2,-1)
            * valh,valhu : values imposed to each variable
        * dx, t : unused in this function
        
    * Outputs :
        * hb,hub : h,hu with BCs
    """
    
    ### BC = [[pos,val h, val hu],]
    hb = 1.*h
    hub = 1.*hu

    for i in range(BC.shape[0]):
        [pos,valh,valhu] = BC[i,:]
        hb[pos] = valh
        hub[pos] = valhu
    
    return hb,hub
def fluxes(h,hu,n):
    """
    *Function to be passed as "fvsolver" argument for DDM_NSWE
    *Impose periodicty to ghost cells and compute the fluxes
    *The domain is supposed to be [-L-3*dx, L+3*dx]
    
    *Inputs
        * h, hu
        * n : unused in this function
        
    * Outputs :
        * fp : fluxes on the cells' interfaces
    """
    
    h0 = np.copy(h)
    u0 = np.copy(hu)
    d0 = np.zeros(n)
    u0 = np.where(h0>1e-10,u0/h0,h0)#hu/h
    
    fp, fm, sc = wb4.fluxes_sources(d0,h0,u0)
    return fp
def DDM_NSWE(x1,h1,u1,x2,h2,u2,t0,tmax,bcfunction,dx,nx,
             dt = 0.05, fvsolver=muscl2.fluxes2,fvTimesolver=serre.RK4,
             ghostcells = 3,externalBC="open",coupleSerre=False,serreDomain=1,nitermax=10,dispersiveBC = None,
             configDDM = 1, ov = 0,href=None,uref=None,xref=None, debug = False, modifyIBC = False,eta=0.):
    
    """
    *Applies a DDM to solve the NSWE in two subdomains;
    *Possibly solve the dispersive part of the Serre equation in one of the subdomains
    
    * Inputs:
        - x1,x2,xref : subdomains/monodomain
        - h1,h2,u1,u2,href,uref : solutions in the subdomains/in the monodomain
        - t0,tmax,dt
        - dx
        - nx: number of points in monodomain
        - bcfunction : function to impose BCs in the NSWE solver (ex: function DDM_BCs defined above)
        - fvsolver : function to compute the fluxes in the NSWE part (ex: fucntion fluxes defined above)
        - fvTimesolver : function to integrate the NSWE in time
        - ghost cells : number of ghost cells in each side
        - externalBC: "open"/"periodic" (only the "open" case is correctly defined)
        - coupleSerre : True/False : compute the dispersive part
        - serreDomain (1/2): subdomain with the dispersive equation
        - nitermax
        - dispersiveBC : 
        - configDDM : 1 (no overlap) / 2 (overlap)
        - ov : size of overlap (Number of overlapped cells = 1 + 2*ov, ov>=0)
        - debug : impose referential solution in ghost cells
    """
    
    t = t0
    it = 0
    grav = 9.81/10.
    
    uall1 = u1
    hall1 = h1
    tall1 = np.ones(1)*t0
    uall2 = u2
    hall2 = h2
    tall2 = np.ones(1)*t0
    huref = href*uref
    
    while t < tmax and dt > 1e-9:
        it = it+1
        t = t+dt
        #print("t = ",t)
        
        h1prev = np.copy(h1)
        u1prev = np.copy(u1)
        h2prev = np.copy(h2)
        u2prev = np.copy(u2)
        hu1prev = h1prev*u1prev
        hu2prev = h2prev*u2prev
        
        niter = 0
        converg = False
        
        while niter < nitermax and converg == False:
            
            niter = niter+1
            
            h1mm = np.copy(h1)
            u1mm = np.copy(u1)
            h2mm = np.copy(h2)
            u2mm = np.copy(u2)
            
            hu1 = h1*u1
            hu2 = h2*u2

            Nddm = nx/2

            if externalBC == "periodic":
                bcparam1 = np.array([[0,h2[-6],hu2[-6]], #external
                                     [1,h2[-5],hu2[-5]], #external
                                     [2,h2[-4],hu2[-4]], #external
                                     [-3,h2[3],hu2[3]],  #interface
                                     [-2,h2[4],hu2[4]],  #interface
                                     [-1,h2[5],hu2[5]]]) #interface
                bcparam2 = np.array([[0,h1[-6],hu1[-6]], #interface
                                     [1,h1[-5],hu1[-5]], #interface
                                     [2,h1[-4],hu1[-4]], #interface
                                     [-3,h1[3],hu1[3]],  #external
                                     [-2,h1[4],hu1[4]],  #external 
                                     [-1,h1[5],hu1[5]]]) #external
            elif externalBC == "open":
                if configDDM == 1:
                    bcparam1 = np.array([[0,h1prev[3],hu1prev[3]],
                                         [1,h1prev[3],hu1prev[3]],
                                         [2,h1prev[3],hu1prev[3]],
                                         [-3,h2[3],hu2[3]],
                                         [-2,h2[4],hu2[4]],
                                         [-1,h2[5],hu2[5]]])
                    bcparam2 = np.array([[0,h1[-6],hu1[-6]],
                                         [1,h1[-5],hu1[-5]],
                                         [2,h1[-4],hu1[-4]],
                                         [-3,h2prev[-4],hu2prev[-4]],
                                         [-2,h2prev[-4],hu2prev[-4]],
                                         [-1,h2prev[-4],hu2prev[-4]]])
                elif configDDM == 2:
                    if not debug :
                        bcparam1 = np.array([[0,h1prev[3],hu1prev[3]],
                                             [1,h1prev[3],hu1prev[3]],
                                             [2,h1prev[3],hu1prev[3]],
                                             [-3,h2[4+2*ov],hu2[4+2*ov]],
                                             [-2,h2[5+2*ov],hu2[5+2*ov]],
                                             [-1,h2[6+2*ov],hu2[6+2*ov]]])
                        bcparam2 = np.array([[0,h1[-7-2*ov],hu1[-7-2*ov]],
                                             [1,h1[-6-2*ov],hu1[-6-2*ov]],
                                             [2,h1[-5-2*ov],hu1[-5-2*ov]],
                                             [-3,h2prev[-4],hu2prev[-4]],
                                             [-2,h2prev[-4],hu2prev[-4]],
                                             [-1,h2prev[-4],hu2prev[-4]]])

                    else:
                        Nddm = nx/2
                        print(Nddm)
                        bcparam1 = np.array([[0,h1prev[3],hu1prev[3]],
                                             [1,h1prev[3],hu1prev[3]],
                                             [2,h1prev[3],hu1prev[3]],
                                             [-3,href[Nddm+1,it],huref[Nddm+1,it]],
                                             [-2,href[Nddm+2,it],huref[Nddm+2,it]],
                                             [-1,href[Nddm+3,it],huref[Nddm+3,it]]])
                        bcparam2 = np.array([[0,href[Nddm-3,it],huref[Nddm-3,it]],
                                             [1,href[Nddm-2,it],huref[Nddm-2,it]],
                                             [2,href[Nddm-1,it],huref[Nddm-1,it]],
                                             [-3,h2prev[-4],hu2prev[-4]],
                                             [-2,h2prev[-4],hu2prev[-4]],
                                             [-1,h2prev[-4],hu2prev[-4]]])

            h1,hu1 = bcfunction(h1,hu1,bcparam1,dx,t)
            h2,hu2 = bcfunction(h2,hu2,bcparam2,dx,t)

            if debug :
                print("Stencils around the interface point (href,h1,h2):")
                print("g-3","g-2","g-1","Int","g1","g2","g3")
                print(href[Nddm-3:Nddm+4,it])
                print(h1[-7:])
                print(h2[:7])
            #print("Stencils hu before:")
            #print(hu1[-7:])
            #print(hu2[:7])
            h1,hu1 = fvTimesolver(x1,h1prev,hu1prev,fvsolver,bcfunction,bcparam1,dx,dt,x1.size,t,ghostcells)
            h2,hu2 = fvTimesolver(x2,h2prev,hu2prev,fvsolver,bcfunction,bcparam2,dx,dt,x2.size,t,ghostcells)   

            if debug :
                print("")
                print("**********")
                print("Debugging")
                print("**********")
                print("")
                print("Verification of the interface position [M,Om1,Om2]:")
                print(xref[Nddm],x1[-4],x2[3])
                print("")
                print("Verification of the solution in ghost cells (h1,href):")
                print("Int","g1","g2","g3")
                print(h1[-4:])
                print(href[Nddm+ov:Nddm+4+ov,it])
                print("")
                print("Verification of the solution in the first cells (h1,href):")
                print("-3","-2","-1","Int")
                print(h1[-7:-3])
                print(href[Nddm-3+ov:Nddm+1+ov,it])
                print("")
                print("Verification of the solution in ghost cells (h2,href):")
                print("g-3","g-2","g-1","Int")
                print(h2[:4])
                print(href[Nddm-3-ov:Nddm+1-ov,it])
                print("")
                print("Verification of the solution in the first cells (h2,href):")
                print("Int","1","2","3")
                print(h2[3:7])
                print(href[Nddm-ov:Nddm+4-ov,it])
                print("")
                print("**///***///***///***")
                print("")
                print("")
            #u1 = np.where(h1[:]>1e-5, hu1[:]/h1[:], 0.)
            #u2 = np.where(h2[:]>1e-5, hu2[:]/h2[:], 0.)   
            

            ## correct ghost cells on the interface (unnecessary?)
            if externalBC == "open":   
                if configDDM == 1:
                    bcparam1 = np.array([[-3,h2[3],hu2[3]],
                                         [-2,h2[4],hu2[4]],
                                         [-1,h2[5],hu2[5]]])
                    bcparam2 = np.array([[0,h1[-6],hu1[-6]],
                                         [1,h1[-5],hu1[-5]],
                                         [2,h1[-4],hu1[-4]]])
                elif configDDM == 2:
                    if not debug :
                        bcparam1 = np.array([[-3,h2[4+2*ov],hu2[4+2*ov]],
                                             [-2,h2[5+2*ov],hu2[5+2*ov]],
                                             [-1,h2[6+2*ov],hu2[6+2*ov]]])
                        bcparam2 = np.array([[0,h1[-7-2*ov],hu1[-7-2*ov]],
                                             [1,h1[-6-2*ov],hu1[-6-2*ov]],
                                             [2,h1[-5-2*ov],hu1[-5-2*ov]]])
                    else :
                        bcparam1 = np.array([[-3,href[Nddm+1,it],huref[Nddm+1,it]],
                                             [-2,href[Nddm+2,it],huref[Nddm+2,it]],
                                             [-1,href[Nddm+3,it],huref[Nddm+3,it]]])
                        bcparam2 = np.array([[0,href[Nddm-3,it],huref[Nddm-3,it]],
                                             [1,href[Nddm-2,it],huref[Nddm-3,it]],
                                             [2,href[Nddm-1,it],huref[Nddm-1,it]]])
            h1,hu1 = bcfunction(h1,hu1,bcparam1,dx,t)
            h2,hu2 = bcfunction(h2,hu2,bcparam2,dx,t)
            u1 = np.where(h1[:]>1e-5, hu1[:]/h1[:], 0.)
            u2 = np.where(h2[:]>1e-5, hu2[:]/h2[:], 0.)   
            
            ## verify convergence
            eps = 1e-9
            #print(niter,np.linalg.norm(h1-h1mm),np.linalg.norm(h2-h2mm),np.linalg.norm(u1-u1mm),np.linalg.norm(u2-u2mm))
            if np.linalg.norm(h1-h1mm) < eps and np.linalg.norm(h2-h2mm) < eps and \
               np.linalg.norm(u1-u1mm) < eps and np.linalg.norm(u2-u2mm) < eps :
                    converg = True
            
            a1 = u1[-4]-np.sqrt(grav*h1[-4])
            b1 = u1[-4]+np.sqrt(grav*h1[-4])
            a2 = u2[3+2*ov]-np.sqrt(grav*h2[3+2*ov])
            b2 = u2[3+2*ov]+np.sqrt(grav*h2[3+2*ov])
            
            #print("Int 1: ",niter,np.absolute(u1[-4]-u2[3+2*ov]),np.absolute(h1[-4]-h2[3+2*ov]),
                #  np.absolute(a1-a2),np.absolute(b1-b2))
            
            a1 = u1[-4-2*ov]-np.sqrt(grav*h1[-4-2*ov])
            b1 = u1[-4-2*ov]+np.sqrt(grav*h1[-4-2*ov])
            a2 = u2[3]-np.sqrt(grav*h2[3])
            b2 = u2[3]+np.sqrt(grav*h2[3])
            
            #print("Int 2: ", niter,np.absolute(u1[-4-2*ov]-u2[3]),np.absolute(h1[-4-2*ov]-h2[3]),
            #np.absolute(a1-a2),np.absolute(b1-b2))
                    

            nxx = uref.shape[0]
            err1 = np.linalg.norm(u1[3:-3]-uref[3:u1.size-3,it])
            err2 = np.linalg.norm(u2[3:-3]-uref[nxx-u2.size+3:-3,it])
            
            if not coupleSerre :
                print("")
                print("t = %f"%t)
                print("Error wr to monodomain solution (Om1,Om2): ",err1,err2)
                print("")
            
        #solve Dispersion for one of the domains
        FDorder = 4  ## order of FD scheme
        
        if coupleSerre:
            if serreDomain == 1:
                    ## define boundary conditions
                
                if modifyIBC :  ## Dirichlet condition
                    dispersiveBC = np.array([[0,"Robin",0.,1.,0.], 
                                     [-2,"Robin",u1[-5],1.,0.], ## external boundaries
                                     [-1,"Robin",u1[-4],1.,0.]]) ## interface : Dirichlet 
                u1aux = serreTBC.modifiedEFDSolverFM(h1[3:-3],u1[3:-3],dx,dt,FDorder,dispersiveBC)
                #u1aux = serreTBC.linearEFDSolverFM(h1[3:-3],u1[3:-3],dx,dt,FDorder,dispersiveBC,h0=h1[-1],u0=u1[-1])
                u1 = np.hstack((u1[:3],u1aux,u1[-3:]))
                
                ## copy to overlapped area
                u2[3:4+2*ov] = u1[-4-2*ov:-3]
            elif serreDomain == 2:
                if modifyIBC :
                    dispersiveBC = np.array([[0,"Robin",u2[3],1.,0.],
                                     [1,"Robin",u2[4],1.,0.],
                                     [-1,"Robin",0.,1.,0.]])
                u2aux = serreTBC.modifiedEFDSolverFM(h2[3:-3],u2[3:-3],dx,dt,FDorder,dispersiveBC)
                #u2aux = serreTBC.linearEFDSolverFM(h2[3:-3],u2[3:-3],dx,dt,FDorder,dispersiveBC,h0=h2[0],u0=u2[0])
                u2 = np.hstack((u2[:3],u2aux,u2[-3:]))
                ## copy to overlapped area
                u1[-4-2*ov:-3] = u2[3:4+2*ov]


        hall1 = np.column_stack((hall1,h1))
        uall1 = np.column_stack((uall1,u1))
        tall1 = np.hstack((tall1,t*np.ones(1)))

        hall2 = np.column_stack((hall2,h2))
        uall2 = np.column_stack((uall2,u2))
        tall2 = np.hstack((tall2,t*np.ones(1)))
    return hall1,uall1,tall1,hall2,uall2,tall2

import numpy as np
import kdv
import besseTBC
import matplotlib.pyplot as plt
import pickle
import generalFunctions as gF
def showRankingSpeed(tests,nb,criteria="speed") :
    if criteria == "speed":
        idxcrt = 1
    elif criteria == "error":
        idxcrt = 2
    else:
        print("Wrong criteria")
  
    dt = np.dtype('str,int,float')
    testsList = np.array([(eval(key),int(tests[key][0]), float(tests[key][1]))for key in tests.keys()])

    idxdlt = np.zeros(1)
    for i in range(testsList.shape[0]) :
        if float(testsList[i,2]) > 1e6 or np.isnan(testsList[i,2]):
            idxdlt = np.append(idxdlt,i)
    idxdlt = np.delete(idxdlt,0)
    
    testsList = np.delete(testsList,idxdlt,axis=0)
        
    print("Best results")
    idxbest = np.argsort(testsList[:,idxcrt])
    for i in range(nb):
        coefs = testsList[idxbest[i],0]
        print(r"(%.3f,%.3f)" %(coefs[0],coefs[1]),testsList[idxbest[i],1],testsList[idxbest[i],2])
## Returns TBC computed in \Omega_j for the resolution in \Omega_i

def computeExteriorTBC(u,dx,cL,cR,cond,order=2) :
    
    if cond == 1:
        if order == 2:
            return u[-1] - cL/dx*(u[-1] - u[-2]) + cL*cL/(dx*dx)*(u[-1] - 2.*u[-2] + u[-3])
            #return u[-1] - cL/dx*(3./2.*u[-1] - 2.*u[-2] + 1./2.*u[-3]) + cL*cL/(dx*dx)*(u[-1] - 2.*u[-2] + u[-3])
    elif cond == 2:
        if order == 2:
            return u[0] - cR*cR/(dx*dx)*(u[0] - 2.*u[1] + u[2])
    elif cond == 3:
        if order == 2:
            return 1./dx*(u[1]-u[0]) + cR/(dx*dx)*(u[0] - 2.*u[1] + u[2])
def computeCorrectionsTBC(u1,u2,uprev1,uprev2,dx,dt,cL,cR,pointR) :
    
    corr = np.zeros(3)
    
    ### TBC for the point N in Omega2
    if pointR == 0:
        corr[0] = cL/dx*(u1[-1] - u1[-2]) - dx/dt*cL*cL*(u1[-1] - uprev1[-1] - uprev2[0])
        corr[1] = -dx/dt*cR*cR*(u2[0] - uprev2[0] - uprev1[-1])
        corr[2] = -2.*dx/dt*(dx*uprev1[-2] + cR*uprev1[-1])
    
    ### TBC for the point N+1 in Omega2
    else:
        corr[0] = 2.*cL*dx/dt*(cL*uprev2[0] + dx*uprev2[1])
        corr[1] = -dx/dt*cR*cR*(u2[0] - uprev2[0] - uprev1[-1])
        corr[2] = -2.*dx/dt*(dx*uprev1[-2] + cR*uprev1[-1])
    
    return corr
    
def verifyConvergence(u1,u1prev,u2,u2prev,uref,it,dx,eps):
    conv = False
    diff1 = np.linalg.norm(u1-u1prev)
    diff2 = np.linalg.norm(u2-u2prev)
    diff = np.sqrt(dx)*np.amax(np.array([diff1,diff2]))

    
    #### Alternative
    #nx = u1.size
    #diff1 = np.linalg.norm(u1-uref[:nx,it])
    #diff2 = np.linalg.norm(u2-uref[nx-1:,it])
    #diff = np.sqrt(dx)*np.amax(np.array([diff1,diff2]))
    
    #diff = np.absolute(u1[-1] - u2[0])
    
    if (diff<eps):
        conv = True
                           
    return conv,diff
## Additive Schwarz method (for solving one time step)


#### Debugging levels
## debug = 0 : no debug ->                 u_i^{n+1,0} = u^{n,infty};   TBC(u_i) = TBC(u_j)
## debug = 1 : exact previous solution     u_i^{n+1,0} = u_ref^n;       TBC(u_i) = TBC(u_j)
## debug = 2 : exact prev. sol. and TBC    u_i^{n+1,0} = u_ref^n;       TBC(u_i) = TBC(u_ref)
## debug = 3 : exact. prev. sol and Dirichlet in interface

def ASM(x1,x2,u1,u2,t,dx,dt,order,cL,cR,maxiter,eps,uref,it,debug,corrTBC,verbose,fourConditions,pointR,modifyDiscret):
    
    converg = False
    niter = 0
    nx = u1.size
    
    if debug == 3: ### impose Dirichlet on the interface
        cL = 0.
        cR = 0.
    
    if not debug :
        u1prev = np.copy(u1)
        u2prev = np.copy(u2)
    else:
        u1prev = np.copy(uref[0:nx,it-1])
        u2prev = np.copy(uref[nx-1:,it-1])       
    diff = np.zeros(1)
    err = np.zeros(1)
    errInterfaceL = np.zeros(1)
    errInterfaceR = np.zeros(1)
    
    while converg == False and niter <= maxiter :
        niter = niter + 1
 
        if verbose :
            print("++++++++++++++++++++++++++++++++++++++++")
            print(" ")
            print(r"niter = %d"%niter)
        if debug <=1:
            uleft = np.copy(u1)
            uright = np.copy(u2)
        else :
            uleft = np.copy(uref[0:nx,it])
            uright = np.copy(uref[nx-1:,it])  
            
        corr = computeCorrectionsTBC(uleft,uright,u1prev,u2prev,dx,dt,cL,cR,pointR)
                    
        ## BC for Omega_1
        BC1 = np.zeros(3) ############
        if debug <=2 :
            BC1[1] = computeExteriorTBC(uright,dx,cL,cR,2) + corrTBC*corr[1]
            BC1[2] = computeExteriorTBC(uright,dx,cL,cR,3) + corrTBC*corr[2]
        else : ### impose Dirichlet on the interface
            BC1[1] = uref[nx-1,it]
            BC1[2] = uref[nx-1,it]/dx - uref[nx-2,it]/dx 
                   
        ## BC for Omega_2
        BC2 = np.zeros(4)
        if debug <=2 :
            BC2[0] = computeExteriorTBC(uleft,dx,cL,cR,1) +  corrTBC*corr[0]
        else : ### impose Dirichlet on the interface
            BC2[0] = uref[nx-1,it]

        #### Store solutions of the previous iteration
        u1m = np.copy(u1)
        u2m = np.copy(u2)

        coef = np.zeros((1,2))

        coef[0,0] = cL
        coef[0,1] = cR
 
        ### modify uncentered -> centered ("4th TBC")
        ###### in N
        if pointR == 1:
            BC2[3] = u1prev[-1] - dt/(dx*dx*dx)*(-1./2.*uleft[-3] + uleft[-2]) ####################
        ###### in N+1
        else :
            BC2[3] = u2prev[1] - dt/(dx*dx*dx)*(-1./2.*uleft[-2]) ####################
        #BC2[3] = uleft[-1]
        
        if verbose :
            print(" ")
            print(" Before computation :")
            print("Norm(u^n - u_ref^n)")
            print(np.linalg.norm(u1-uref[0:nx,it-1]),np.linalg.norm(u2-uref[nx-1:,it-1]))
        
        u1 = besseTBC.IFDBesse(u1prev,None,None,t,dt,dx,1.,0,coef,0,corrTBC,BC1,
                               fourConditions = fourConditions,pointR = pointR, modifyDiscret = modifyDiscret)
        u2 = besseTBC.IFDBesse(u2prev,None,None,t,dt,dx,1.,0,coef,corrTBC,0,BC2,
                               fourConditions = fourConditions,pointR = pointR, modifyDiscret = modifyDiscret)
        
        if verbose :
            print(" ")
            print(" After computation :")
            print("Norm(u^{n+1} - u_ref^{n+1})")
            print(np.linalg.norm(u1-uref[0:nx,it]),np.linalg.norm(u2-uref[nx-1:,it]))

            print(" ")
            print("Debugging : Residuals computed using : ")
            print("*** Computed solution    : u^{n+1} = DDM(u_ref^{n}) ")
            print("*** Referential solution : u^{n+1} = u_ref^{n+1} ")
        
        ### residuals TBCleft - TBCright
        resTBC1 = -cL*(u2[1] - u2[0])/dx + cL*cL*(-2.*u2[1] + u2[2])/(dx*dx) + \
                    cL*(u2[1]-2.*u2[2]+u2[3])/dx + 2.*cL*dx/dt*(cL*(u2[0] - u2prev[0]) + dx*(u2[1]-u2prev[1])) \
                    + cL*(u1[-1] - u1[-2])/dx - cL*cL*(u1[-3] - 2.*u1[-2])/(dx*dx)
        resTBC2 = -cR*cR/(dx*dx)*(u1[-3] -2.*u1[-2]) + dx/dt*cR*cR*(u1[-1] - u1prev[-1]) + \
                    cR*cR/(dx*dx)*(u2[2] -2.*u2[1]) + dx/dt*cR*cR*(u2[0] - u2prev[0])
        resTBC3 = (u1[-1] - u1[-2])/dx + cR/(dx*dx)*(u1[-3] - 2.*u1[-2]) \
                    - 2.*dx/dt*(dx*(u1[-2]-u1prev[-2]) + cR*(u1[-1] - u1prev[-1])) + (u1[-4] - 2.*u1[-3] + u1[-2])/dx \
                    -(u2[1] - u2[0])/dx - cR/(dx*dx)*(-2.*u2[1] + u2[2])
            
        ### residuals ut + uxxx
        resEq1 = (u1[-2] - u1prev[-2])/dt + 1./(2.*dx*dx*dx)*(-u1[-4] + 2.*u1[-3] - 2.*u1[-1] + u2[1] ) 
        resEq2 = (u1[-1] - u1prev[-1])/dt + 1./(2.*dx*dx*dx)*(-u1[-3] + 2.*u1[-2] - 2.*u2[1] + u2[2] ) 
        resEq3 = (u2[1] - u2prev[1])/dt + 1./(2.*dx*dx*dx)*(-u1[-2] + 2.*u2[0] - 2.*u2[2] + u2[3] ) 
        
        uaux1 = np.copy(u1)
        uaux2 = np.copy(u2)
        u1 = np.copy(uref[0:nx,it])
        u2 = np.copy(uref[nx-1:,it]) 

        ### same residuals computed for the referential solution (must be zero)
        resTBC1Ref = -cL*(u2[1] - u2[0])/dx + cL*cL*(-2.*u2[1] + u2[2])/(dx*dx) + \
                    cL*(u2[1]-2.*u2[2]+u2[3])/dx + 2.*cL*dx/dt*(cL*(u2[0] - u2prev[0]) + dx*(u2[1]-u2prev[1])) \
                    + cL*(u1[-1] - u1[-2])/dx - cL*cL*(u1[-3] - 2.*u1[-2])/(dx*dx)
        resTBC2Ref = -cR*cR/(dx*dx)*(u1[-3] -2.*u1[-2]) + dx/dt*cR*cR*(u1[-1] - u1prev[-1]) + \
                    cR*cR/(dx*dx)*(u2[2] -2.*u2[1]) + dx/dt*cR*cR*(u2[0] - u2prev[0])
        resTBC3Ref = (u1[-1] - u1[-2])/dx + cR/(dx*dx)*(u1[-3] - 2.*u1[-2]) \
                    - 2.*dx/dt*(dx*(u1[-2]-u1prev[-2]) + cR*(u1[-1] - u1prev[-1])) + (u1[-4] - 2.*u1[-3] + u1[-2])/dx \
                    -(u2[1] - u2[0])/dx - cR/(dx*dx)*(-2.*u2[1] + u2[2])

        ### same residuals computed for the referential solution (must be zero)
        resEq1Ref = (u1[-2] - u1prev[-2])/dt + 1./(2.*dx*dx*dx)*(-u1[-4] + 2.*u1[-3] - 2.*u1[-1] + u2[1] ) 
        resEq2Ref = (u1[-1] - u1prev[-1])/dt + 1./(2.*dx*dx*dx)*(-u1[-3] + 2.*u1[-2] - 2.*u2[1] + u2[2] ) 
        resEq3Ref = (u2[1] - u2prev[1])/dt + 1./(2.*dx*dx*dx)*(-u1[-2] + 2.*u2[0] - 2.*u2[2] + u2[3] ) 
        
        u1 = np.copy(uaux1)
        u2 = np.copy(uaux2)
        
        if verbose :
            print("Res TBC computed solution    : ",resTBC1,resTBC2,resTBC3)
            print("Res Eq computed solution    : ",resEq1,resEq2,resEq3)
            print("Res TBC referential solution : ",resTBC1Ref,resTBC2Ref,resTBC3Ref)
            print("Res Eq referential solution : ",resEq1Ref,resEq2Ref,resEq3Ref)
            print(" ")
        
        converg, diffit = verifyConvergence(u1,u1m,u2,u2m,uref,it,dx,eps)

        diff = np.hstack((diff,diffit*np.ones(1))) 
        if uref != None:
            nx = u1.size
            errit = np.sqrt(dx)*np.linalg.norm(np.concatenate((u1,u2))-np.concatenate((uref[0:nx,it],uref[nx-1:,it])))
            err = np.hstack((err,errit*np.ones(1)))
            errInterfaceL = np.hstack((errInterfaceL,np.absolute(u1[-1] - uref[nx-1,it])*np.ones(1)))
            errInterfaceR = np.hstack((errInterfaceR,np.absolute(u2[0] - uref[nx-1,it])*np.ones(1)))
                    
    return u1,u2,niter, np.sqrt(dx)*np.linalg.norm(u1-u1m), np.sqrt(dx)*np.linalg.norm(u2-u2m),diff,err,errInterfaceL,errInterfaceR
## number of iterations for each time step
def plotIter(niter,tall,title="Number of iterations") :
    plt.figure()
    plt.plot(tall[1:],niter[1:])
    plt.title(title)
    plt.xlabel("t")

## convergence behaviour for each iteration in a timestep
def plotSnapshotCV(diffit,tall,tsnaps,niterall,loc=0) :
    plt.figure()
    plt.title("Convergence of the Schwarz method")
    plt.xlabel("Number of iterations (k)")
    plt.ylabel("$log(max(||u_1^{k+1}-u_1^{k}||, ||u_2^{k+1}-u_2^{k}||))$")
    xmax = 0
    
    for t in tsnaps:
        it = np.argmin(np.absolute(tall-t))
        niter = niterall[it]
        realt = tall[it]
        if niter > xmax:
            xmax = niter
        plt.plot(np.arange(1,niter+1,1),np.log10(diffit[1:niter+1,it]),label=r"t = %.4f" %realt)
        
    plt.xlim(1,xmax)
    plt.legend(loc=loc)
    
## error behaviour for each iteration in a timestep
def plotSnapshotError(errit,tall,tsnaps,niterall,loc=0) :
    plt.figure()
    plt.title("Error (computed x reference solution)")
    plt.xlabel("Number of iterations (k)")
    plt.ylabel("$||u-u_{ref}||$")
    xmax = 0
    
    for t in tsnaps:
        it = np.argmin(np.absolute(tall-t))
        niter = niterall[it]
        realt = tall[it]
        if niter > xmax:
            xmax = niter
        plt.plot(np.arange(1,niter+1,1),(errit[1:niter+1,it]),label=r"t = %.4f" %realt)

    plt.xlim(1,xmax)
    plt.legend(loc=loc)


## Error in convergence for each time step    
def plotConvergenceErrors(errall,niterall,tall):
    plt.figure()
    plt.title("Error in converged solution")
    plt.xlabel("$t$")
    plt.ylabel("$||u-u_{ref}||$")
    
    ## Error in convergence
    for i in range(errall.shape[-1]-1) :
        niter = niterall[i]
        errall[-1,i] = errall[niter,i]
        
    plt.plot(tall[:-1],errall[-1,1:-1])
    
    return errall
    
## Plot exact + computed solutions in the timesteps in tsnaps    
def plotSnapshotDDM(xs,us,tall,tsnaps,saveSnap = False, savePaths = None, ext = "png", location = 0) :

    cntSnap = 0
    for t in tsnaps :
        plt.figure()
        it = np.argmin(np.absolute(tall-t))
        plt.plot(xs[1],us[1][:,it],label="$\Omega_1$")
        plt.plot(xs[2],us[2][:,it],label="$\Omega_2$")
        plt.plot(xs[0],us[0][:,it],label="Exact",marker="+",linestyle="None",markevery=10)
        plt.title(r't = %f'%tall[it])
        plt.xlabel("$x$")
        plt.ylabel("$u$")
        plt.legend(loc = location)
        if saveSnap:
            plt.savefig(savePaths[cntSnap]+"."+ext)
        cntSnap = cntSnap + 1
            
def computeErrorDDM(us,dt) :
    ucomp = np.vstack((us[1],us[2]))
    e,ErrTm,ErrL2 = besseTBC.computeError(ucomp,us[0],dt)
        
    return  e,ErrTm,ErrL2
def runSimulation(x1,x2,u1,u2,t0,tmax,dx,dt,cL,cR,
                  uref = None, maxiter = 10,eps = 1e-6,printstep=10,debug=0,corrTBC=0,verbose=0,
                  fourConditions = 0, pointR = 0, modifyDiscret = 0) :

    print("")
    print("*** Computing solution ...")
    
    t = t0
    uall1 = np.copy(u1)
    uall2 = np.copy(u2)
    tall = np.ones(1)*t0
    niterall = int(np.zeros(1))
    diff1all = np.zeros(1)
    diff2all = np.zeros(1)
    diffitall = np.zeros((maxiter,int((tmax-t0)/dt)+2))
    erritall = np.zeros((maxiter,int((tmax-t0)/dt)+2))
    errIntLitall = np.zeros((maxiter,int((tmax-t0)/dt)+2))
    errIntRitall = np.zeros((maxiter,int((tmax-t0)/dt)+2))
    
    nsteps = 0
    
    while t < tmax :
        
        nsteps = nsteps + 1
        t = t+dt
        

        u1,u2,niter,diff1,diff2,diff,err,errIntL,errIntR = ASM(x1,x2,u1,u2,t,dx,dt,0,cL,cR,
                                                           maxiter,eps,uref,nsteps,debug,corrTBC,
                                                           verbose,fourConditions,pointR,modifyDiscret)
        if niter > maxiter-1:
            niter = maxiter-1
        
        uall1 = np.column_stack((uall1,u1))
        uall2 = np.column_stack((uall2,u2))
        tall = np.hstack((tall,t*np.ones(1)))
        niterall = np.hstack((niterall,int(niter*np.ones(1))))
        diff1all = np.hstack((diff1all,diff1*np.ones(1)))
        diff2all = np.hstack((diff2all,diff2*np.ones(1)))
        diffitall[0:niter+1,nsteps] = diff[0:niter+1]
        erritall[0:niter+1,nsteps] = err[0:niter+1]
        erritall[-1,nsteps] = erritall[niter,nsteps]
        errIntLitall[0:niter+1,nsteps] = errIntL[0:niter+1]
        errIntLitall[-1,nsteps] = errIntLitall[niter,nsteps]
        errIntRitall[0:niter+1,nsteps] = errIntR[0:niter+1]
        errIntRitall[-1,nsteps] = errIntRitall[niter,nsteps]        
        
        if nsteps%printstep == 0:
            print(nsteps,t,niter)
            
    print("*** End of computation ***")        
    return uall1,uall2,tall,niterall,diff1all,diff2all,diffitall,erritall,errIntLitall,errIntRitall
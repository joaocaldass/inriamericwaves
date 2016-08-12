
import numpy as np
import matplotlib.pyplot as plt
import kdv
import generalFunctions as gF
from scipy import special
import sys
import math
import json
import yaml
import marshal

nan = float('nan')
## supposing x.size = y.size  (???)
def convolution(x,y) :
    c = np.zeros_like(x)
    N = x.size
    for i in range(N) :
        #j = 0
        #while j<=N and i-j >= 0:
        #    c[i] = c[i] + x[i-j]*y[j]
        #    j = j+1
        c[i] = np.sum(np.flipud(x[0:i+1])*y[0:i+1])
    return c
#Error functions defined by Besse
def computeError(u,uexact,dt) :
    e = np.linalg.norm(uexact - u,axis=0)/np.linalg.norm(uexact,axis=0)
    ErrTm = np.amax(e)
    ErrL2 = np.sqrt(dt)*np.linalg.norm(e)
    
    return e,ErrTm,ErrL2
def imposeTBC(M,rhs,um,umm,U2,dx,dt,order,coef,BCs,correctTBCL,correctTBCR,fourConditions,pointR,useTBCL,useTBCR) :
    
    if order == 0:
        
        cL = coef[0,0]
        cR = coef[0,1]
        
        M[-1,:] = 0
        M[-2,:] = 0

        if useTBCL:
        ## Eq. in N
            if pointR == 0:
                M[0,:] = 0
                ####M[0,0] = 1. + cL*U2/dx + cL*cL*U2/(dx*dx) + correctTBCL*(-cL/dx + dx/dt*cL*cL) 
                ####M[0,1] = -cL*U2/dx - 2.*cL*cL*U2/(dx*dx) + correctTBCL*(cL/dx)
                ####M[0,2] = cL*cL*U2/(dx*dx)
                M[0,0] = 1. + cL*U2/dx + cL*cL*U2/(dx*dx) + correctTBCL*(-2.*cL/dx + dx/dt*cL*cL) 
                M[0,1] = -cL*U2/dx - 2.*cL*cL*U2/(dx*dx) + correctTBCL*(cL/dx)
                M[0,2] = cL*cL*U2/(dx*dx)
                rhs[0] = BCs[0]
            ## Eq in N+1
            else :
                M[correctTBCL,:] = 0
                M[correctTBCL,0] = 1. + cL*U2/dx + cL*cL*U2/(dx*dx) + correctTBCL*(2.*dx/dt*cL*cL) #+ correctTBCL*(dx/dt*cL*cL)
                M[correctTBCL,1] = -cL*U2/dx - 2.*cL*cL*U2/(dx*dx) + correctTBCL*(cL/dx + 2.*cL*dx*dx/dt)
                M[correctTBCL,2] = cL*cL*U2/(dx*dx) + correctTBCL*(-2.*cL/dx)
                M[correctTBCL,3] = correctTBCL*(cL/dx)
                rhs[correctTBCL] = BCs[0]
        else : ## Neumann
            M[0,:] = 0.
            #M[0,0] = 1.
            #M[0,1] = -1.
            M[0,0] = 1. + 1./dx + 1./(dx*dx)
            M[0,1] = -1./dx -  2./(dx*dx)
            M[0,2] = 1./(dx*dx)
            rhs[0] = 0.
            
        if useTBCR :
            M[-1,-1] = 1. - cR*cR/(dx*dx)  + correctTBCR*(dx/dt*cR*cR) #+ correctTBCR*(dx/dt*cR*cR)
            M[-1,-2] =   2.*cR*cR/(dx*dx)
            M[-1,-3] = - cR*cR/(dx*dx)
            rhs[-1] = BCs[1] 

            M[-2,-1] = 1./(dx) + cR/(dx*dx)  + correctTBCR*(-2.*dx/dt*cR)
            M[-2,-2] = -1./(dx) - 2.*cR/(dx*dx) + correctTBCR*(-2.*dx*dx/dt + 1./dx)
            M[-2,-3] =   cR/(dx*dx) + correctTBCR*(-2./dx)
            M[-2,-4] = correctTBCR*(1./dx)
            rhs[-2] = BCs[2]
        else: ## Neumman and Dirichlet
            M[-1,-1] = 1.
            rhs[-1] = 0.
            
            M[-2,-1] = 1.
            M[-2,-2] = -1.
            rhs[-2] = 0.
            
            
        #rhs[correctTBCL] = BCs[0]
        ####if pointR == 0:
        ####    rhs[0] = BCs[0]
        ####else :
        ####    rhs[correctTBCL] = BCs[0]
        ####rhs[-2] = BCs[2]
        ####rhs[-1] = BCs[1] 
        
        
        
        ### Modify uncentered -> centered
        if correctTBCL and fourConditions:
        ####### in N
            if pointR == 1:
                M[0,:] = 0.
                M[0,0] = 1.
                M[0,1] = -dt/(dx*dx*dx)
                M[0,2] = dt/(2.*dx*dx*dx)
                rhs[0] = BCs[3]
        ####### in N+1
            else :
                M[1,:] = 0.
                M[1,0] = dt/(dx*dx*dx)
                M[1,1] = 1.
                M[1,2] = -dt/(dx*dx*dx)
                M[1,3] = dt/(2.*dx*dx*dx)
                rhs[1] = BCs[3]
        
    elif order == 1 or order == .5:
        
        ## order = 0.5 : P_1^2 truncated in degree 1
        
        simplif = 1.
        if order == .5:
            simplif = 0.
            
        dx2 = dx*dx
        dt2 = dt*dt
        
        cL = coef[0,0]
        cR = coef[0,1]
        dL = coef[1,0]
        dR = coef[1,1]
        
        M[0,:] = 0
        M[-1,:] = 0
        M[-2,:] = 0
        
#        M[0,0] = 1. - U2*(cL/d2 - dL/d1) - U2*(cL*cL/d4 - 2.*cL*dL/d3 + dL*dL/d2)
#        M[0,1] = -U2*(-2.*cL/d2 + dL/d1) - U2*(-cL*cL*4./d4 + 2.*cL*dL*3./d3 - 2.*dL*dL/d2)
#        M[0,2] = -U2*(cL/d2) - U2*(cL*cL*6./d4 - 2.*cL*dL*3./d3 - 2.*dL*dL/d2)
#        M[0,3] = -U2*(-cL*cL*4./d4 + 2.*cL*dL/d3)
#        M[0,4] = -U2*cL*cL/d4
        
#        M[-1,-1] = 1. + cR/d3 + dR/d2
#        M[-1,-2] = -3.*cR/d3 -2.*dR/d2
#        M[-1,-3] = 3.*cR/d3 + dR/d2
#        M[-1,-3] = -cR/d3
        
#        M[-2,-1] = 1./d1 + cR*cR/d4 + 2.*cR*dR/d3 + dR*dR/d2
#        M[-2,-2] = -1./d1 - 4.*cR*cR/d4 - 2.*cR*dR*3./d3 - 2.*dR*dR/d2
#        M[-2,-3] = 6.*cR*cR/d4 + 2.*cR*cR*3./d3 + dR*dR/d2
#        M[-2,-4] = -4.*cR*cR/d4 - 2.*cR*dR/d3
#        M[-2,-5] = cR*cR/d4
        
#        rhs[0] = 0.
#        rhs[-1] = 0.
#        rhs[-2] = 0
                
        M[0,0] = 1. + U2/dx*(cL/dt + dL) + U2/dx2*(simplif*cL*cL/dt2 + 2.*cL*dL/dt + dL*dL)
        M[0,1] = - U2/dx*(cL/dt + dL) - 2.*U2/dx2*(simplif*cL*cL/dt2 + 2.*cL*dL/dt + dL*dL)
        M[0,2] = + U2/dx2*(simplif*cL*cL/dt2 + 2.*cL*dL/dt + dL*dL)

        M[-1,-1] = 1. - (simplif*cR*cR/dt2 + 2.*cR*dR/dt + dR*dR)/dx2
        M[-1,-2] = 2.*(simplif*cR*cR/dt2 + 2.*cR*dR/dt + dR*dR)/dx2
        M[-1,-3] = -(simplif*cR*cR/dt2 + 2.*cR*dR/dt + dR*dR)/dx2 

        M[-2,-1] = 1./dx + (cR/dt + dR)/dx2
        M[-2,-2] = -1./dx -2.*(cR/dt + dR)/dx2
        M[-2,-3] = (cR/dt + dR)/dx2
        
        rhs[0] = - U2*cL/(dt*dx)*(um[1] - um[0]) + \
                U2/dx2*(2.*simplif*cL*cL/dt2 + 2.*cL*dL/dt)*(um[0] - 2.*um[1] + um[2]) - \
                U2*simplif*cL*cL/(dt2*dx2)*(umm[0] - 2.*umm[1] + umm[2])
        rhs[-1] = -(2.*simplif*cR*cR/dt2 + 2.*cR*dR/dt)*(um[-1] - 2.*um[-2] + um[-3])/dx2 + \
                simplif*cR*cR/(dt2*dx2)*(umm[-1] - 2.*umm[-2] + umm[-3])      
        rhs[-2] = cR/(dt*dx2)*(um[-1] - 2.*um[-2] + um[-3])
    
    return M,rhs
# Our scheme for the dispersion equation + Besse's TBCs
def IFDBesse2(u,um,umm,t,dt,dx,U2,order,coef,correctTBCL,correctTBCR,BCs=np.zeros(3)):
    k = dt/(dx*dx*dx)

    nx = u.size - 1

    d0  = 1.*np.ones(nx+1)
    d1 = -k*13/8.*np.ones(nx)
    d2 = +k*np.ones(nx-1)
    d3 = -k*1./8.*np.ones(nx-2)
    
    M =  np.diag(d0) + np.diag(d1,1) + np.diag(d2,2) + np.diag(d3,3) - np.diag(d1,-1) - np.diag(d2,-2) - np.diag(d3,-3)
       
    vvv = np.zeros(nx+1)
    vvv[0] = 1. - 1.*k
    vvv[1] = 3.*k
    vvv[2] = -3.*k
    vvv[3] = 1.*k 
    
    vvv[0] = 1. -5./2.*k #######
    vvv[1] = 9.*k #######
    vvv[2] = -12.*k #######
    vvv[3] = 7.*k #######
    vvv[4] = -3./2*k #######
    
    zzz = -np.flipud(vvv)
    zzz[-1] = 1. + 1.*k
    zzz[-1] = 1. + 5./2.*k ######

    M[0,:] = vvv
    M[1,:] = np.roll(vvv,1)
    M[2,:] = np.roll(vvv,2)
    
    M[nx,:] = zzz
    M[nx-1,:] = np.roll(zzz,-1)
    M[nx-2,:] = np.roll(zzz,-2)
    
    rhs = np.copy(u)
    
    M,rhs = imposeTBC(M,rhs,um,umm,U2,dx,dt,order,coef,BCs,correctTBCL,correctTBCR)
    
    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(suppress=True)
    
    u2 = np.linalg.solve(M,rhs)
    
    return u2
# Our scheme for the dispersion equation + Besse's TBCs
def IFDBesse(u,um,umm,t,dt,dx,U2,order,coef,correctTBCL,correctTBCR,useTBCL, useTBCR, BCs=np.zeros(3),
             fourConditions=0,pointR = 0,modifyDiscret=0, middlePoint = 0, uNeig = None):
    k = dt/(dx*dx*dx)

    nx = u.size - 1

    d0  = 1.*np.ones(nx+1)
    d1 = -k*np.ones(nx)
    d2 = +k*1./2.*np.ones(nx-1)
    
    M =  np.diag(d0) + np.diag(d1,1) + np.diag(d2,2)  - np.diag(d1,-1) - np.diag(d2,-2)
     
    ### Increase order of uncentered !!!!!    
    vvv = np.zeros(nx+1)
    ### order 2
    vvv[0] = 1. - 5./2.*k
    vvv[1] = 9.*k
    vvv[2] = -12.*k
    vvv[3] = 7.*k
    vvv[4] = -3./2.*k
    
    zzz = -np.flipud(vvv)
    zzz[-1] = 1. + 5./2.*k
    
    M[0,:] = vvv
    M[1,:] = np.roll(vvv,1)
    
    M[nx,:] = zzz
    M[nx-1,:] = np.roll(zzz,-1)
    
    if modifyDiscret == 1: ### uncentered discretization for the point near the interface
            if middlePoint == 0:
                mp = nx/2 + 1
            else:
                mp = middlePoint
            M[mp,:] = 0.
            M[mp,mp] = 1. - 5./2.*k
            M[mp,mp+1] = 9.*k
            M[mp,mp+2] = -12.*k
            M[mp,mp+3] = 7.*k
            M[mp,mp+4] = -3./2.*k
    
    if modifyDiscret == 2 :  #### centered discret for point N+1 in Omega2
        M[1,:] = 0
        M[1,0] = k
        M[1,1] = 1.
        M[1,2] = -k
        M[1,3] = 1./2.*k
        
    if modifyDiscret == 3 :   #### centered discret for point N+1 in Omega2 (1/2 + 1/2)
        M[1,:] = 0.
        M[1,0] = k/2.
        M[1,1] = 1.
        M[1,2] = -k
        M[1,3] = k/2.        
        
        
    rhs = np.copy(u)
    
    M,rhs = imposeTBC(M,rhs,um,umm,U2,dx,dt,order,coef,BCs,correctTBCL,correctTBCR,fourConditions,pointR,
                      useTBCL, useTBCR)
    
    if modifyDiscret == 2 :  #### centered discret. for point N+1 in Omega2
        rhs[1] = rhs[1] + uNeig*k/2.
    if modifyDiscret == 3 :  #### centered discret for point N+1 in Omega2 (1/2 + 1/2)
        rhs[1] = rhs[1] - k/2.*(uNeig[-1] - uNeig[-2])
    
    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(suppress=True)
    
    u2 = np.linalg.solve(M,rhs)
    
    return u2
## Only dispersive part of KdV
def runDispKdV(x,u,t0,tmax,U2,coef, periodic=1, vardt = True, dt = 0.01, verbose = True, order = 0,
               correctTBC=0, modifyDiscret = 0, middlePoint = 0, useTBCL = True, useTBCR = True):
    
    print("")
    print("*** Computing solution ...")
    t = t0
    tall = np.ones(1)*t0
    u0 = u
    uall = u
    u0min = np.amin(u)
    u0max = np.amax(u)
    dx = np.diff(x)[0]
    iter = 0
    eps = 1e-6

    ##### Parameters
    printstep = 5
    
    
    orderOrig = order
    um = np.copy(u)
    umm = np.copy(u)
    while t<tmax:
        iter = iter + 1

        if iter>2:
            umm = np.copy(um)
        if iter>1:
            um = np.copy(u)
        
        order = orderOrig
        if iter<=2:
            order = 0
    
        if vardt:
            umax = np.amax(np.absolute(u))
            dt = dx/(1.+2*umax) - eps     # CFL CONDITION

        t = t+dt
        
        if periodic :
            u = kdv.FourierSolver(u,t,dt,dx)
        else :
            u = IFDBesse(u,um,umm,t,dt,dx,U2,order,coef,0,0,useTBCL,useTBCR,
                         modifyDiscret = modifyDiscret, middlePoint = middlePoint)

        uall = np.column_stack((uall,u))
        tall = np.hstack((tall,t*np.ones(1)))
        
        if iter%100 == 0 and verbose:
            print(iter,t)
            
    print("*** End of computation ***")
    return uall,tall

### Optimize parameters of the polynomial
def optimizeParamO0(x,u,uallexact,t0,tmax,U2,cLs,cRs, N, dt, prevTests, verbose = False,LeD = False) :

    order = 0

    dx = x[1] - x[0]
        
    print(cLs,cRs)

    uallexactOrig = np.copy(uallexact)
    
    tests = {}
    testsLight = prevTests

    cntTests = 0

    coefTBC = np.zeros((1,2))

    uall = 1
    tall = 1
    
    for cL in cLs:
        cntcR = 0 ## to avoid repeated tests if cL=cR
        for cR in cRs:
            if (str((cL,cR)) not in testsLight.keys()) : ## if a new test
                cntcR = cntcR + 1

                coefTBC[0,0] = cL
                if not LeD:
                    coefTBC[0,1] = cR
                else : coefTBC[0,1] = cL

                if not LeD or cntcR == 1 :
                    cntTests = cntTests+1
                    uall,tall = runDispKdV(x,u,t0,tmax,U2, coefTBC ,periodic=0, vardt = False, dt = dt, verbose = False,
                                              order = order, useTBCL = True, useTBCR = True)    
                    print(cL,cR)

                    if cntTests > 0:
                        coef = np.amax(uallexact[:,1])/np.amax(uall[:,1])
                        coef = 1.
                        uallexact[:,1:] = uallexact[:,1:]/coef
                        
                    en,ErrTm,ErrL2 = computeError(uall,uallexact,dt)

                    tests[str((cL,cR))] = (uall,tall,en,ErrTm,ErrL2)
                    testsLight[str((cL,cR))] = (ErrTm,ErrL2)
                    print(ErrTm,ErrL2)


                    
    ## Errors

    errorsall = np.array([(key,float(testsLight[key][0]), float(testsLight[key][1]))for key in testsLight.keys()])
    
    
    cnt = 0
    ## ignore explosed solutions
    for i in range(errorsall.shape[0]) :
        if float(errorsall[i,1]) < 10 :
            if cnt == 0:
                errors = errorsall[i,:]
            else:
                errors = np.vstack((errors,errorsall[i,:]))
            cnt = cnt+1

    print(errors.shape)
    if (errors.ndim == 2):
        testTmmax = errors[np.argmax(errors[:,2]),0]
        testTmmin = errors[np.argmin(errors[:,2]),0]
        testL2max = errors[np.argmax(errors[:,1]),0]
        testL2min = errors[np.argmin(errors[:,1]),0]
    else :
        testTmmax = errors[0]
        testTmmin = errors[0]
        testL2max = errors[0]
        testL2min = errors[0]
        
    if verbose :
        print('ErrTm Max = ',testsLight[testTmmax][0], r"(cL,cR) = ", testTmmax)
        print('ErrTm Min = ',testsLight[testTmmin][0], r"(cL,cR) = ", testTmmin)
        print('ErrL2 Max = ',testsLight[testL2max][1], r"(cL,cR) = ", testL2max)
        print('ErrL2 Min = ',testsLight[testL2min][1], r"(cL,cR) = ", testL2min)

    return uall,uallexact,tall,tests,testsLight,errors,testTmmax,testTmmin,testL2max,testL2min
## Fundamental solution of the equation (for any initial condition)
def fundamentalSolution(x,t) :
    a = np.power(3.*t,-1./3.)
    Ai,Aip,Bi,Bip = special.airy(x*a)
    return a*Ai
## Exact solution
def exactSolution(x,t,initCond,app=1000.) :
    dx = x[1] - x[0]
    left = np.arange(x[0]-app,x[0],dx)
    sizeL = np.size(left)
    right = np.arange(x[-1] + dx,x[-1] + app, dx)
    sizeR = np.size(right)
    x2 = np.concatenate((left,x,right))

    uf = fundamentalSolution(x2,t)
    u0 = initCond(x2)
    
    b = np.convolve(uf/np.sum(uf),u0,'same')
    c = b[sizeL:sizeL+x.size]
    
    return c
## Initial condition
def initGauss(x) :
    return np.exp(-x*x)
## Initial condition
def initCosinus(x) :
    return np.cos(x)
## Exact solution
def exactSolution2(x,t) :
    dx = x[1] - x[0]
    left = np.arange(x[0]-1000.,x[0],dx)
    sizeL = np.size(left)
    right = np.arange(x[-1] + dx,x[-1] + 1000., dx)
    sizeR = np.size(right)
    x2 = np.concatenate((left,x,right))
    a = np.power(3.*t,-1./3.)
    Ai,Aip,Bi,Bip = special.airy(x2*a)
    e = np.exp(-x2*x2)
    b = np.convolve(a*Ai,e,'same')/np.sum(np.absolute(e))
    c = b[sizeL:sizeL+x.size]
    print(t)
    return c
## Load previous results from file and put in the library (or create a new one)
def loadTests(filename,module="marshal") :

    # load from file:
    try :
        with open(filename, 'r') as f:
            try:
                if module == "yaml":
                    tests = yaml.safe_load(f)
                elif module == "marshal":
                    tests = marshal.load(f)
            # if the file is empty the ValueError will be thrown
            except ValueError:
                tests= {}
    except : tests = {}
        
    return tests

def saveTests(tests,filename,module="marshal"):
    # save to file:
    with open(filename, 'w') as f:
        if module == "marshal":
            marshal.dump(tests, f)
        elif module == "json" :
            json.dump(tests, f)
    
def showRanking(tests,nb,criteria="L2") :
    if criteria == "Tm":
        idxcrt = 1
    elif criteria == "L2":
        idxcrt = 2
    else:
        print("Wrong criteria")
    
    testsList = np.array([(key,float(tests[key][0]), float(tests[key][1]))for key in tests.keys()])

    print("Best results")
    idxbest = np.argsort(testsList[:,idxcrt])
    for i in range(nb):
        coefs = eval(testsList[idxbest[i],0])
        #print(testsList[idxbest[i],:])
        print(r"(%.3f,%.3f)" %(coefs[0],coefs[1]),testsList[idxbest[i],1],testsList[idxbest[i],2])
def animateBestSolution(x,u,uallexact,U2,t0,tmax,dt,tests,xmin=None,xmax=None,criteria="L2"):

    coefs = np.zeros((1,2))
    
    if criteria == "Tm":
        idxcrt = 1
    elif criteria == "L2":
        idxcrt = 2
    else:
        print("Wrong criteria")
        
    testsList = np.array([(key,float(tests[key][0]), float(tests[key][1]))for key in tests.keys()])
    tup = testsList[np.argsort(testsList[:,idxcrt])[0]][0]
    coefsBest = np.array(eval(tup))

    coefs[0,:] = coefsBest
    uall,tall = runDispKdV(x,u,t0,tmax,U2, coefs,periodic=0, vardt = False, dt = dt, verbose = False,
                                              order = 0)

    if xmin == None:
        xmin = x[0]
    if xmax == None:
        xmax = x[-1]
    ymin = np.amin(np.concatenate(uall))
    ymax = np.amax(np.concatenate(uall))

    anim = gF.plotAnimationNSolutions(2,x,np.array([uall,uallexact]),
                               tall,xmin,xmax,ymin,ymax+.2,["best sol : " + tup,"exact"],r'$u$',location=(.7,.7))
    
    return anim
def plotBestWorstSolution(x,u,uallexact,U2,t0,tmax,dt,tall,tsnaps,tests,criteria="L2",
                          savePath = None, ext = "png",legloc=0):

    coefs = np.zeros((1,2))
    
    if criteria == "Tm":
        idxcrt = 1
    elif criteria == "L2":
        idxcrt = 2
    else:
        print("Wrong criteria")
        
    testsList = np.array([(key,float(tests[key][0]), float(tests[key][1]))for key in tests.keys()])

    tupBest = testsList[np.argsort(testsList[:,idxcrt])[0]][0]
    coefsBest = np.array(eval(tupBest))
    tupWorst = testsList[np.argsort(testsList[:,idxcrt])[-1]][0]
    coefsWorst = np.array(eval(tupWorst))
    
    coefs[0,:] = coefsBest
    uallBest,tallBest = runDispKdV(x,u,t0,tmax,U2, coefs,periodic=0, vardt = False, dt = dt, verbose = False,
                                              order = 0)
    coefs[0,:] = coefsWorst
    uallWorst,tallWorst = runDispKdV(x,u,t0,tmax,U2, coefs,periodic=0, vardt = False, dt = dt, verbose = False,
                                              order = 0)

    ymin = np.amin(np.concatenate(uallBest))
    ymax = np.amax(np.concatenate(uallBest))

    cnt=0
    for t in tsnaps :
        plt.figure()
        it = np.argmin(np.absolute(tallBest-t))
        plt.plot(x,uallBest[:,it],label="best sol (%.3f,%.3f) " %(coefsBest[0],coefsBest[1]))
        plt.plot(x,uallWorst[:,it],label="worst sol (%.3f,%.3f)" %(coefsWorst[0],coefsWorst[1]),linestyle="--")
        plt.plot(x,uallexact[:,it],marker='+',markevery=10,linestyle='None', label='Exact sol')
        plt.title(r't = %f'%tallBest[it])
        plt.legend(loc=legloc)
        plt.xlabel("$x$",fontsize="x-large")
        plt.ylabel("$u$",fontsize="x-large")
        if savePath != None :
            plt.savefig(savePath+"Snap" + str(cnt) + "."+ext)
        cnt = cnt+1
from itertools import cycle

def plotCoefError(testsLight,fixedValues,fixedCoef = "right",minCoef=None,maxCoef=None,
                  savePath=None,ext="png",plotErrors="both",legloc=0):

    lines = ["--","-.",":","-"]
    linecycler = cycle(lines)
    markers = ["None","+","o"]
    markers = np.tile(markers,(2,1)).transpose().flatten()
    markers = np.ndarray.tolist(markers)
    markercycler = cycle(markers)

    
    plt.figure()
    
    if fixedCoef == "right" :
        fixedIndex = 1
        leg = r"$c_R = $"
        title = "$c_R$ fixed"
        xlbl = "$c_L$"
    elif fixedCoef == "left" :
        fixedIndex = 0
        leg = r"$c_L = $"
        title = "$c_L$ fixed"
        xlbl = "$c_R$"
        
    ## string -> (cL,cR)
    testsList = np.array([(key,float(testsLight[key][0]), float(testsLight[key][1]))for key in testsLight.keys()])
    testsCoefsList = np.zeros((testsList.shape[0],4))
    
    for i in range(testsCoefsList.shape[0]) :
        tup = eval(testsList[i,0])
        testsCoefsList[i,0] = tup[0]
        testsCoefsList[i,1] = tup[1]
    testsCoefsList[:,2:4] = testsList[:,1:3] ## cL,cR,Tm,L2


    errormin = 100000000
    errormax = -100000000
    for fixed in fixedValues :
        filtredList = np.zeros((1,3))
        line = np.zeros((1,3))
        for i in range(testsCoefsList.shape[0]):
            if np.absolute(testsCoefsList[i,fixedIndex] - fixed) < 1e-6 :
                line[0,0] = testsCoefsList[i,1-fixedIndex] ## cL,Tm,L2
                line[0,1:3] = testsCoefsList[i,2:4]
                filtredList = np.concatenate((filtredList,line))
        filtredList = np.delete(filtredList,0,0)
        filtredList = filtredList[np.argsort(filtredList[:,0])]
        if plotErrors == "both" : 
            plt.plot(filtredList[:,0],filtredList[:,1],label=leg + r"%.3f"%fixed + " - Tm",
                     linestyle = next(linecycler), marker = next(markercycler) )
            plt.plot(filtredList[:,0],filtredList[:,2],label=leg + r"%.3f"%fixed + " - L2",
                     linestyle = next(linecycler), marker = next(markercycler) )
        elif plotErrors == "L2" :
            plt.plot(filtredList[:,0],filtredList[:,2],label=leg + r"%.3f"%fixed,
                     linestyle = next(linecycler), marker = next(markercycler))
        elif plotErrors == "Tm" :
            plt.plot(filtredList[:,0],filtredList[:,1],label=leg + r"%.3f"%fixed,
                     linestyle = next(linecycler), marker = next(markercycler))


        
        print(minCoef,maxCoef,filtredList[0,0],filtredList[-1,0])
        if minCoef == None:
            minCoef = 100000000
            if (filtredList[0,0] < minCoef):
                minCoef = filtredList[0,0]
        if maxCoef == None :
            maxCoef = -100000000
            if (filtredList[-1,0] > maxCoef):
                maxCoef = filtredList[-1,0]

        print(minCoef,maxCoef,errormin,errormax)
        idxmin = np.argmin(np.absolute(filtredList[:,0]-minCoef))
        idxmax = np.argmin(np.absolute(filtredList[:,0]-maxCoef))
        if np.amin([filtredList[idxmin:idxmax+1,1:]]) < errormin:
            errormin = np.amin([filtredList[idxmin:idxmax+1,1:]])
        if np.amax([filtredList[idxmin:idxmax+1,1:]]) > errormax:
            errormax = np.amax([filtredList[idxmin:idxmax+1,1:]])

    plt.xlim((minCoef,maxCoef))
    plt.ylim((errormin,errormax))
    plt.xlabel(xlbl,fontsize="x-large")
    
    if plotErrors == "both" :
        plt.ylabel("Error",fontsize="x-large")
    elif plotErrors == "L2":
        plt.ylabel(r"$e_{L2}$",fontsize="x-large")
    elif plotErrors == "Tm":
        plt.ylabel(r"$e_{Tm}$",fontsize="x-large")
    #plt.title(title)
    plt.legend(loc = legloc)
    
    if savePath != None:
        plt.savefig(savePath+"."+ext)
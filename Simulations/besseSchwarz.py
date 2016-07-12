
# -*- coding: utf-8 -*-

import numpy as np
import kdv
import besseTBC
import matplotlib.pyplot as plt
import cPickle as pickle
import generalFunctions as gF
def showRankingSpeed(tests,nb,criteria="speed") :
    if criteria == "speed":
        idxcrt = 1
    elif criteria == "error":
        idxcrt = 2
    else:
        print("Wrong criteria")
  
    testsList = np.array([(eval(key),int(tests[key][0]), float(tests[key][1]))for key in tests.keys()])

    idxdlt = np.zeros(1)
    for i in range(testsList.shape[0]) :
        if float(testsList[i,2]) > 1e6 or np.isnan(testsList[i,2]):
            idxdlt = np.append(idxdlt,i)
    idxdlt = np.delete(idxdlt,0)
    
    testsList = np.delete(testsList,idxdlt,axis=0)
     
    if nb > 0:
        print("Best results")
    idxbest = np.argsort(testsList[:,idxcrt])
    for i in range(nb):
        coefs = testsList[idxbest[i],0]
        print(r"(%.3f,%.3f)" %(coefs[0],coefs[1]),testsList[idxbest[i],1],testsList[idxbest[i],2])
        
    return testsList[idxbest,:]
def getTestResult(tests,cL,cR) :
    
    eps = 1e-6
    
    for key in tests.keys() :
        coefs = eval(key)
        if np.absolute(coefs[0] - cL) < eps and np.absolute(coefs[1] - cR) < eps:
            return tests[key]
    
    print("Coefs not found!!")
    
    return None
def getTestResultVariableT0(tests,t0) :
    
    if isinstance(t0,tuple):
        return tests[t0]
    else :
        return tests[str(t0)]
def getTestResumeVariableT0(tests,barrierN=0,barrierP=0) :
    
    resumeT0 = {}
    resumeCoef = {}
    for t0 in tests.keys():
        testT0 = getTestResultVariableT0(tests,t0)

        resultsRanking =  showRankingSpeed(testT0,0,criteria="speed")
        resultsRankingN = np.copy(resultsRanking) ## only negative cL
        resultsRankingP = np.copy(resultsRanking) ## only positive cL
        for i in range(resultsRanking.shape[0]) :
            resultsRanking[i,0] = resultsRanking[i,0][0]  ### only cL
            resultsRankingP[i,0] = resultsRankingP[i,0][0]
            resultsRankingN[i,0] = resultsRankingN[i,0][0]
            if resultsRanking[i,0] < barrierN:
                resultsRankingP[i,1] = 999
            if resultsRanking[i,0] > barrierP:
                resultsRankingN[i,1] = 999
        
        orderedIndex = np.argsort(resultsRanking[:,0])  ### growing cL
        resultsOrdered = resultsRanking[orderedIndex,:]
        resultsOrderedN = resultsRankingN[orderedIndex,:]
        resultsOrderedP = resultsRankingP[orderedIndex,:]
        
        itmin = np.argmin(resultsOrdered[:,1])
        itmax = np.argmax(resultsOrdered[:,1])
        itminN = np.argmin(resultsOrderedN[:,1])
        itmaxN = np.argmax(resultsOrderedN[:,1])
        itminP = np.argmin(resultsOrderedP[:,1])
        itmaxP = np.argmax(resultsOrderedP[:,1])
        errmin = np.argmin(resultsOrdered[:,2])
        errmax = np.argmax(resultsOrdered[:,2])       

        
        resumeT0[t0] = [resultsOrdered,resultsRanking,
                        np.array([resultsOrdered[itmin,0],resultsOrdered[itmin,1]]),
                        np.array([resultsOrdered[itmax,0],resultsOrdered[itmax,1]]),
                        np.array([resultsOrdered[errmin,0],resultsOrdered[errmin,2]]),
                        np.array([resultsOrdered[errmax,0],resultsOrdered[errmax,2]]),
                        np.array([resultsOrderedN[itminN,0],resultsOrderedN[itminN,1]]),
                        np.array([resultsOrderedP[itminP,0],resultsOrderedP[itminP,1]])]
        
        
    return resumeT0
        
from itertools import cycle

def plotErrorEvolution(tests,nb,legloc=0,savePath = None, ext = "png",titleCompl = "", errorLR = False,
                       rangeCoefs="all") : 

    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)
    
    markers = ["None","+","o"]
    markers = np.tile(markers,(3,1)).transpose().flatten()
    markers = np.ndarray.tolist(markers)
    markercycler = cycle(markers)
    
    print(markers)
    
    listNiter = np.zeros((1,2))
    line = np.zeros((1,2))
    for coefs in tests.keys():
        c = eval(coefs)[0]
        niter = tests[coefs][0]
        line[0,0] = c
        line[0,1] = niter
        listNiter = np.vstack((listNiter,line))
    listNiter = np.delete(listNiter,0,0)
    
    idxFaster = np.argsort(listNiter[:,1])
    orderedList = listNiter[idxFaster]    

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    cnt = 0
    i = 0
    while cnt < nb:
        c = orderedList[i,0]
        if ((rangeCoefs=="all" or rangeCoefs =="positive") and c >= 0.) or \
           ((rangeCoefs=="all" or rangeCoefs =="negative") and c <= 0.):

            error = getTestResult(tests,c,c)[2]

            ax.plot(np.log10(error),label="$c_L = %.3f$"%c,linestyle=next(linecycler), marker = next(markercycler))
            plt.xlabel("Iteration")
            plt.ylabel("log(Error)")
            #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
            legend = plt.legend(loc=legloc)
            # Set the fontsize
            for label in legend.get_texts():
                label.set_fontsize('small')

            if titleCompl == "":
                plt.title("Error evolution - %d faster cases (for the complete domain)"%nb)
            else :
                #plt.title("Error evolution - %d faster cases (for the complete domain) - "%nb + titleCompl)
                plt.title(titleCompl)
            cnt = cnt+1
        i = i+1

            
    if savePath != None:
        plt.savefig(savePath + "." + ext)            

    if errorLR :
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        for i in range(nb):
            c = orderedList[i,0]
            error1 = getTestResult(tests,c,c)[4]
            error2 = getTestResult(tests,c,c)[6]

            plt.plot(np.log10(error1),label="$c_L = %.3f$; L"%c,linestyle=next(linecycler), marker = next(markercycler))
            plt.plot(np.log10(error2),label="$c_L = %.3f$; R"%c,linestyle=next(linecycler), marker = next(markercycler))
            plt.xlabel("Iteration")
            plt.ylabel("log(Error)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)       
            if titleCompl == "":
                plt.title("Error evolution - %d faster cases (for each domain)"%nb)
            else :
                plt.title("Error evolution - %d faster cases (for each domain) - "%nb + titleCompl)

        if savePath != None:
            plt.savefig(savePath + "LR." + ext)
from itertools import cycle

def plotIterationsxCoef(tests,legLabel,titleCompl = "", legloc=0,savePath = None, ext = "png", txtFmt="%.4f",
                        xmin = None, xmax = None, ymin= None, ymax = None, differentLines = False, markevery = 10) : 

    if differentLines :
        lines = ["-","--","-.",":"]
        linecycler = cycle(lines)

        markers = ["None","+","o"]
        markers = np.tile(markers,(3,1)).transpose().flatten()
        markers = np.ndarray.tolist(markers)
        markercycler = cycle(markers)
    
    fig = plt.figure()
    
    a = getTestResumeVariableT0(tests)

    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    
    t0s = np.array(a.keys())
    t0sb = []
    print(t0s)
    for i in range(t0s.size):
        t0s[i] = eval(t0s[i])
        t0sb.append(eval(t0s[i]))
    orderedt0s = np.sort(t0sb)
        
    for t0float in orderedt0s :
    #for t0 in a.keys():   
        t0 = str(t0float)
        niterOrdered = a[t0][0]
        if differentLines :
            ax.plot(niterOrdered[:,0],niterOrdered[:,1],label=r'%s = ' %legLabel +txtFmt %(float(t0)),
                    linestyle = next(linecycler), marker = next(markercycler), markevery = markevery)
        else :
            ax.plot(niterOrdered[:,0],niterOrdered[:,1],label=r'%s = ' %legLabel +txtFmt %(float(t0)))
        niterMin = a[t0][2][1]
        cLiterMin = a[t0][2][0]

        ## find smaller niter
        for i in range(len(niterOrdered[:,0])) :
            cL = niterOrdered[i,0]
            if np.absolute(cL - cLiterMin) < 1e-9:
                errIterMin = niterOrdered[i,2]
                break

        print("t0 = %f --> min it = %d for cL = cR = %f and error = %e" % (float(t0),niterMin,cLiterMin,errIterMin))

    if xmin == None:
        xmin = niterOrdered[0,0]
    if xmax == None :
        xmax = niterOrdered[-1,0]
    if ymin == None :
        ymin = np.amin(niterOrdered[:,1]) - 1
    if ymax == None :
        ymax = np.amax(niterOrdered[:,1]) + 1
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xlabel("$c$",fontsize="x-large")
    plt.ylabel("Number of iterations",fontsize="large")
    #plt.title("Nb. of iter. until the convergence - " + titleCompl)
        
    
    
    if savePath != None:
        plt.savefig(savePath + "." + ext)
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
        #corr[0] = cL/dx*(u1[-1] - u1[-2]) - dx/dt*cL*cL*(u1[-1] - uprev1[-1] - uprev2[0])
        corr[0] = -cL/dx*(u1[-2]) - dx/dt*cL*cL*(u1[-1] - uprev1[-1] - uprev2[0])
        corr[1] = -dx/dt*cR*cR*(u2[0] - uprev2[0] - uprev1[-1])
        corr[2] = -2.*dx/dt*(dx*uprev1[-2] + cR*uprev1[-1])
    
    ### TBC for the point N+1 in Omega2
    else:
        corr[0] = 2.*cL*dx/dt*(cL*uprev2[0] + dx*uprev2[1])
        corr[1] = -dx/dt*cR*cR*(u2[0] - uprev2[0] - uprev1[-1])
        corr[2] = -2.*dx/dt*(dx*uprev1[-2] + cR*uprev1[-1])
    
    return corr
    
def verifyConvergence(u1,u1prev,u2,u2prev,uref,it,dx,eps, error, criteria = "diffIteration"):
    conv = False
    
    if criteria == "diffIteration":
        diff1 = np.linalg.norm(u1-u1prev)
        diff2 = np.linalg.norm(u2-u2prev)
        diff = np.sqrt(dx)*np.amax(np.array([diff1,diff2]))
    elif criteria == "diffInterface":
        diff = np.absolute(u1[-1] - u2[0])
    elif criteria == "error" :
        diff = error
    else :
        print("Wrong stopping criteria!!!!")
    
    if (diff<eps):
        conv = True
                           
    return conv,diff
## Additive Schwarz method (for solving one time step)


#### Debugging levels
## debug = 0 : no debug ->                 u_i^{n+1,0} = u^{n,infty};   TBC(u_i) = TBC(u_j)
## debug = 1 : exact previous solution     u_i^{n+1,0} = u_ref^n;       TBC(u_i) = TBC(u_j)
## debug = 2 : exact prev. sol. and TBC    u_i^{n+1,0} = u_ref^n;       TBC(u_i) = TBC(u_ref)
## debug = 3 : exact. prev. sol and Dirichlet in interface

def ASM(x1,x2,u1,u2,t,dx,dt,order,cL,cR,maxiter,eps,uref,it,debug,corrTBC,verbose,
        fourConditions,pointR,modifyDiscret,middlePoint, criteria):
    
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
    err1 = np.zeros(1)
    err2 = np.zeros(1)
    errInterfaceL = np.zeros(1)
    errInterfaceR = np.zeros(1)
    
    while converg == False and niter < maxiter :
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
        BC1 = np.zeros(3)
        if debug <=2 :
            BC1[1] = computeExteriorTBC(uright,dx,cL,cR,2) + corrTBC*corr[1]
            BC1[2] = computeExteriorTBC(uright,dx,cL,cR,3) + corrTBC*corr[2]
        else : ### impose Dirichlet on the interface
            BC1[1] = uref[nx-1,it]
            BC1[2] = uref[nx-1,it]/dx - uref[nx-2,it]/dx 
                   
        ## BC for Omega_2
        BC2 = np.zeros(4) ### possible computation of "4th TBC"
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
        ###### on N
        if pointR == 1: ### because the 3rd TBC is imposed on N+1
            BC2[3] = u1prev[-1] - dt/(dx*dx*dx)*(-1./2.*uleft[-3] + uleft[-2])
        ###### on N+1
        else : ### because the 3rd TBC is imposed on N
            BC2[3] = u2prev[1] - dt/(dx*dx*dx)*(-1./2.*uleft[-2])
        
        if verbose :
            print(" ")
            print(" Before computation :")
            print("Norm(u^n - u_ref^n)")
            print(np.linalg.norm(u1-uref[0:nx,it-1]),np.linalg.norm(u2-uref[nx-1:,it-1]))
        
        useTBCL = False
        useTBCR = True
        u1 = besseTBC.IFDBesse(u1prev,None,None,t,dt,dx,1.,0,coef,0,corrTBC,useTBCL,useTBCR,BC1,
                               fourConditions = fourConditions,pointR = pointR, modifyDiscret = 0,
                               middlePoint = middlePoint)
        useTBCL = True
        useTBCR = False    
        u2 = besseTBC.IFDBesse(u2prev,None,None,t,dt,dx,1.,0,coef,corrTBC,0,useTBCL,useTBCR,BC2,
                               fourConditions = fourConditions,pointR = pointR, modifyDiscret = modifyDiscret,
                               middlePoint = middlePoint, uNeig = uleft)
        
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
        resTBC1 =   cL*cL*(-2.*u2[1] + u2[2])/(dx*dx) + cL*cL*dx/dt*(u2[0] - u2prev[0]) \
                    - cL*cL*(u1[-3] - 2.*u1[-2])/(dx*dx) + cL*cL*dx/dt*(u1[-1] - u1prev[-1])
        resTBC2 = -cR*cR/(dx*dx)*(u1[-3] -2.*u1[-2]) + dx/dt*cR*cR*(u1[-1] - u1prev[-1]) + \
                    cR*cR/(dx*dx)*(u2[2] -2.*u2[1]) + dx/dt*cR*cR*(u2[0] - u2prev[0])
        resTBC3 = (u1[-1] - u1[-2])/dx + cR/(dx*dx)*(u1[-3] - 2.*u1[-2]) \
                    - 2.*dx/dt*(dx*(u1[-2]-u1prev[-2]) + cR*(u1[-1] - u1prev[-1])) + (u1[-4] - 2.*u1[-3] + u1[-2])/dx \
                    -(u2[1] - u2[0])/dx - cR/(dx*dx)*(-2.*u2[1] + u2[2])
            
        ### residuals ut + uxxx
        resEq1 = (u1[-2] - u1prev[-2])/dt + 1./(2.*dx*dx*dx)*(-u1[-4] + 2.*u1[-3] - 2.*u1[-1] + u2[1] ) 
        resEq2 = (u1[-1] - u1prev[-1])/dt + 1./(2.*dx*dx*dx)*(-u1[-3] + 2.*u1[-2] - 2.*u2[1] + u2[2] ) 
        resEq3 = (u2[0] - u2prev[0])/dt + 1./(2.*dx*dx*dx)*(-u1[-3] + 2.*u1[-2] - 2.*u2[1] + u2[2] ) 
        resEq4 = (u2[1] - u2prev[1])/dt + 1./(2.*dx*dx*dx)*(-u1[-2] + 2.*u2[0] - 2.*u2[2] + u2[3] ) 
        
        uaux1 = np.copy(u1)
        uaux2 = np.copy(u2)
        u1 = np.copy(uref[0:nx,it])
        u2 = np.copy(uref[nx-1:,it]) 

        ### same residuals computed for the referential solution (must be zero)
        resTBC1Ref =   cL*cL*(-2.*u2[1] + u2[2])/(dx*dx) + cL*cL*dx/dt*(u2[0] - u2prev[0]) \
                    - cL*cL*(u1[-3] - 2.*u1[-2])/(dx*dx) + cL*cL*dx/dt*(u1[-1] - u1prev[-1])
        resTBC2Ref = -cR*cR/(dx*dx)*(u1[-3] -2.*u1[-2]) + dx/dt*cR*cR*(u1[-1] - u1prev[-1]) + \
                    cR*cR/(dx*dx)*(u2[2] -2.*u2[1]) + dx/dt*cR*cR*(u2[0] - u2prev[0])
        resTBC3Ref = (u1[-1] - u1[-2])/dx + cR/(dx*dx)*(u1[-3] - 2.*u1[-2]) \
                    - 2.*dx/dt*(dx*(u1[-2]-u1prev[-2]) + cR*(u1[-1] - u1prev[-1])) + (u1[-4] - 2.*u1[-3] + u1[-2])/dx \
                    -(u2[1] - u2[0])/dx - cR/(dx*dx)*(-2.*u2[1] + u2[2])

        ### same residuals computed for the referential solution (must be zero)
        resEq1Ref = (u1[-2] - u1prev[-2])/dt + 1./(2.*dx*dx*dx)*(-u1[-4] + 2.*u1[-3] - 2.*u1[-1] + u2[1] ) 
        resEq2Ref = (u1[-1] - u1prev[-1])/dt + 1./(2.*dx*dx*dx)*(-u1[-3] + 2.*u1[-2] - 2.*u2[1] + u2[2] ) 
        resEq3Ref = (u2[0] - u2prev[0])/dt + 1./(2.*dx*dx*dx)*(-u1[-3] + 2.*u1[-2] - 2.*u2[1] + u2[2] ) 
        resEq4Ref = (u2[1] - u2prev[1])/dt + 1./(2.*dx*dx*dx)*(-u1[-2] + 2.*u2[0] - 2.*u2[2] + u2[3] ) 
        
        u1 = np.copy(uaux1)
        u2 = np.copy(uaux2)
        
        if verbose :
            print("Res TBC computed solution    : ",resTBC1,resTBC2,resTBC3)
            print("Res Eq computed solution    : ",resEq1,resEq2,resEq3,resEq4)
            print("Res TBC referential solution : ",resTBC1Ref,resTBC2Ref,resTBC3Ref)
            print("Res Eq referential solution : ",resEq1Ref,resEq2Ref,resEq3Ref,resEq4Ref)
            print(" ")
       
        if uref != None:
            nx = u1.size
            errit = np.sqrt(dx)*np.linalg.norm(np.concatenate((u1,u2))-np.concatenate((uref[0:nx,it],uref[nx-1:,it])))
            errit1 = np.sqrt(dx)*np.linalg.norm(u1 - uref[0:nx,it])
            errit2 = np.sqrt(dx)*np.linalg.norm(u2 - uref[nx-1:,it])
            err = np.hstack((err,errit*np.ones(1)))
            err1 = np.hstack((err1,errit1*np.ones(1)))
            err2 = np.hstack((err2,errit2*np.ones(1)))
            errInterfaceL = np.hstack((errInterfaceL,np.absolute(u1[-1] - uref[nx-1,it])*np.ones(1)))
            errInterfaceR = np.hstack((errInterfaceR,np.absolute(u2[0] - uref[nx-1,it])*np.ones(1)))
            
        converg, diffit = verifyConvergence(u1,u1m,u2,u2m,uref,it,dx,eps,errit,criteria=criteria)
        diff = np.hstack((diff,diffit*np.ones(1)))
        
    return u1,u2,niter, np.sqrt(dx)*np.linalg.norm(u1-u1m), np.sqrt(dx)*np.linalg.norm(u2-u2m),\
           diff,err,err1,err2,errInterfaceL,errInterfaceR
## Additive Schwarz method (for solving one time step)


#### Debugging levels
## debug = 0 : no debug ->                 u_i^{n+1,0} = u^{n,infty};   TBC(u_i) = TBC(u_j)
## debug = 1 : exact previous solution     u_i^{n+1,0} = u_ref^n;       TBC(u_i) = TBC(u_j)
## debug = 2 : exact prev. sol. and TBC    u_i^{n+1,0} = u_ref^n;       TBC(u_i) = TBC(u_ref)
## debug = 3 : exact. prev. sol and Dirichlet in interface

def MSM(x1,x2,u1,u2,t,dx,dt,order,cL,cR,maxiter,eps,uref,it,debug,corrTBC,verbose,
        fourConditions,pointR,modifyDiscret,middlePoint, criteria):
    
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
    err1 = np.zeros(1)
    err2 = np.zeros(1)
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
        
        if verbose :
            print(" ")
            print(" Before computation :")
            print("Norm(u^n - u_ref^n)")
            print(np.linalg.norm(u1-uref[0:nx,it-1]),np.linalg.norm(u2-uref[nx-1:,it-1]))
        
        useTBCL = False
        useTBCR = True
        u1 = besseTBC.IFDBesse(u1prev,None,None,t,dt,dx,1.,0,coef,0,corrTBC,useTBCL,useTBCR,BC1,
                               fourConditions = fourConditions,pointR = pointR, modifyDiscret = 0,
                               middlePoint = middlePoint)
        
        corr = computeCorrectionsTBC(u1,uright,u1prev,u2prev,dx,dt,cL,cR,pointR)
        ## BC for Omega_2
        BC2 = np.zeros(4)
        if debug <=2 :
            BC2[0] = computeExteriorTBC(uleft,dx,cL,cR,1) +  corrTBC*corr[0]
        else : ### impose Dirichlet on the interface
            BC2[0] = uref[nx-1,it]
        
        useTBCL = True
        useTBCR = False
   
        u2 = besseTBC.IFDBesse(u2prev,None,None,t,dt,dx,1.,0,coef,corrTBC,0,useTBCL,useTBCR,BC2,
                               fourConditions = fourConditions,pointR = pointR, modifyDiscret = modifyDiscret,
                               middlePoint = middlePoint, uNeig = u1)
       
        if uref != None:
            nx = u1.size
            errit = np.sqrt(dx)*np.linalg.norm(np.concatenate((u1,u2))-np.concatenate((uref[0:nx,it],uref[nx-1:,it])))
            errit1 = np.sqrt(dx)*np.linalg.norm(u1 - uref[0:nx,it])
            errit2 = np.sqrt(dx)*np.linalg.norm(u2 - uref[nx-1:,it])
            err = np.hstack((err,errit*np.ones(1)))
            err1 = np.hstack((err1,errit1*np.ones(1)))
            err2 = np.hstack((err2,errit2*np.ones(1)))
            errInterfaceL = np.hstack((errInterfaceL,np.absolute(u1[-1] - uref[nx-1,it])*np.ones(1)))
            errInterfaceR = np.hstack((errInterfaceR,np.absolute(u2[0] - uref[nx-1,it])*np.ones(1)))
            
        converg, diffit = verifyConvergence(u1,u1m,u2,u2m,uref,it,dx,eps,errit,criteria=criteria)
        diff = np.hstack((diff,diffit*np.ones(1)))
        
    return u1,u2,niter, np.sqrt(dx)*np.linalg.norm(u1-u1m), np.sqrt(dx)*np.linalg.norm(u2-u2m),\
           diff,err,err1,err2,errInterfaceL,errInterfaceR
## Additive Schwarz method (for solving one time step)


#### Debugging levels
## debug = 0 : no debug ->                 u_i^{n+1,0} = u^{n,infty};   TBC(u_i) = TBC(u_j)
## debug = 1 : exact previous solution     u_i^{n+1,0} = u_ref^n;       TBC(u_i) = TBC(u_j)
## debug = 2 : exact prev. sol. and TBC    u_i^{n+1,0} = u_ref^n;       TBC(u_i) = TBC(u_ref)
## debug = 3 : exact. prev. sol and Dirichlet in interface

def MSMi(x1,x2,u1,u2,t,dx,dt,order,cL,cR,maxiter,eps,uref,it,debug,corrTBC,verbose,
        fourConditions,pointR,modifyDiscret,middlePoint, criteria):
    
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
    err1 = np.zeros(1)
    err2 = np.zeros(1)
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
        
        if verbose :
            print(" ")
            print(" Before computation :")
            print("Norm(u^n - u_ref^n)")
            print(np.linalg.norm(u1-uref[0:nx,it-1]),np.linalg.norm(u2-uref[nx-1:,it-1]))
        
        useTBCL = True
        useTBCR = False
   
        u2 = besseTBC.IFDBesse(u2prev,None,None,t,dt,dx,1.,0,coef,corrTBC,0,useTBCL,useTBCR,BC2,
                               fourConditions = fourConditions,pointR = pointR, modifyDiscret = modifyDiscret,
                               middlePoint = middlePoint, uNeig = u1)
        
        corr = computeCorrectionsTBC(uleft,u2,u1prev,u2prev,dx,dt,cL,cR,pointR)
        ## BC for Omega_1
        BC1 = np.zeros(3) ############
        if debug <=2 :
            BC1[1] = computeExteriorTBC(uright,dx,cL,cR,2) + corrTBC*corr[1]
            BC1[2] = computeExteriorTBC(uright,dx,cL,cR,3) + corrTBC*corr[2]
        else : ### impose Dirichlet on the interface
            BC1[1] = uref[nx-1,it]
            BC1[2] = uref[nx-1,it]/dx - uref[nx-2,it]/dx 
        

        useTBCL = False
        useTBCR = True
        u1 = besseTBC.IFDBesse(u1prev,None,None,t,dt,dx,1.,0,coef,0,corrTBC,useTBCL,useTBCR,BC1,
                               fourConditions = fourConditions,pointR = pointR, modifyDiscret = 0,
                               middlePoint = middlePoint)
        
        if uref != None:
            nx = u1.size
            errit = np.sqrt(dx)*np.linalg.norm(np.concatenate((u1,u2))-np.concatenate((uref[0:nx,it],uref[nx-1:,it])))
            errit1 = np.sqrt(dx)*np.linalg.norm(u1 - uref[0:nx,it])
            errit2 = np.sqrt(dx)*np.linalg.norm(u2 - uref[nx-1:,it])
            err = np.hstack((err,errit*np.ones(1)))
            err1 = np.hstack((err1,errit1*np.ones(1)))
            err2 = np.hstack((err2,errit2*np.ones(1)))
            errInterfaceL = np.hstack((errInterfaceL,np.absolute(u1[-1] - uref[nx-1,it])*np.ones(1)))
            errInterfaceR = np.hstack((errInterfaceR,np.absolute(u2[0] - uref[nx-1,it])*np.ones(1)))
            
        converg, diffit = verifyConvergence(u1,u1m,u2,u2m,uref,it,dx,eps,errit,criteria=criteria)
        diff = np.hstack((diff,diffit*np.ones(1)))
        
    return u1,u2,niter, np.sqrt(dx)*np.linalg.norm(u1-u1m), np.sqrt(dx)*np.linalg.norm(u2-u2m),\
           diff,err,err1,err2,errInterfaceL,errInterfaceR
def optimizeSpeed(x,x1,x2,u,uallref,cLs,cRs,prevTests,t0,tmax,dx,dt,maxiter = 100, equalCoef = False,
                  middlePoint = 0, criteria = "diffIteration", epsCV = 1e-7, DDMmethod = ASM,
                  modifyDiscret = 3) :
  
    testsSpeed = prevTests

    keys = []
    for i in range(len(testsSpeed.keys())):
        keys.append(eval(testsSpeed.keys()[i]))
        
    for cL in cLs:
        for cR in cRs:
            
            if equalCoef :
                cR = cL
            
            iskey = False
            eps = 1e-9
            for key in testsSpeed.keys():
                coefs = eval(key)
                if np.absolute(coefs[0]-cL) < eps and np.absolute(coefs[1]-cR) < eps :
                    iskey = True
                    break
            if not iskey :
            #if (cL,cR) not in keys:       
            #if (str((cL,cR)) not in testsSpeed.keys()) :

                coefTBC = np.zeros((1,2))
                coefTBC[0,0] = cL
                coefTBC[0,1] = cR

                
                #uallref,tallref = besseTBC.runDispKdV(x,u,t0,tmax,1., coefTBC , periodic = 0, vardt = False, dt = dt,
                #                                          order = 0, modifyDiscret = 1, middlePoint = middlePoint,
                #                                         useTBCL = False, useTBCR = False)
                             
                nx = x1.size
                u1 = uallref[:nx,0]
                u2 = uallref[nx-1:,0]
                
    
                uall1Speed,uall2Speed,tall,niterallSpeed,diff1allSpeed,\
                diff2allSpeed,diffitallSpeed,errallSpeed,errall1Speed,errall2Speed,\
                errIntLSpeed,errIntRSpeed = runSimulation(x1,x2,u1,u2,t0,tmax,dx,dt,cL,cR,
                                                          uref=uallref,maxiter = maxiter,eps = epsCV,
                                                          printstep=100, debug=0,corrTBC=1, verbose=0,
                                                          fourConditions = 0, pointR = 0,
                                                          middlePoint = middlePoint,
                                                          criteria = criteria,
                                                          DDMmethod = DDMmethod,
                                                          modifyDiscret = modifyDiscret)
                print(cL,cR,niterallSpeed[1],errallSpeed[-1,1])
                #if errallSpeed[-1,1] != 'nan' and  errallSpeed[-1,1] != 'inf' and errallSpeed[-1,1]  < 1e6:
                testsSpeed[str((cL,cR))] = (int(niterallSpeed[1]),float(errallSpeed[-1,1]),
                                            np.ndarray.tolist(errallSpeed[:niterallSpeed[1]+1,1]),
                                            float(errall1Speed[-1,1]),
                                            np.ndarray.tolist(errall1Speed[:niterallSpeed[1]+1,1]),
                                            float(errall2Speed[-1,1]),
                                            np.ndarray.tolist(errall2Speed[:niterallSpeed[1]+1,1]),)
                
            if equalCoef :
                break

    return testsSpeed
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
                  fourConditions = 0, pointR = 0, modifyDiscret = 0, middlePoint = 0,
                  criteria = "diffIteration", DDMmethod = ASM) :

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
    erritall1 = np.zeros((maxiter,int((tmax-t0)/dt)+2))
    erritall2 = np.zeros((maxiter,int((tmax-t0)/dt)+2))    
    errIntLitall = np.zeros((maxiter,int((tmax-t0)/dt)+2))
    errIntRitall = np.zeros((maxiter,int((tmax-t0)/dt)+2))
    
    nsteps = 0
    
    while t < tmax :
        
        nsteps = nsteps + 1
        t = t+dt
        

        u1,u2,niter,diff1,diff2,diff,err,err1,err2,errIntL,errIntR = DDMmethod(x1,x2,u1,u2,t,dx,dt,0,cL,cR,
                                                           maxiter,eps,uref,nsteps,debug,corrTBC,
                                                           verbose,fourConditions,pointR,modifyDiscret,
                                                           middlePoint, criteria)
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
        erritall1[0:niter+1,nsteps] = err1[0:niter+1]
        erritall1[-1,nsteps] = erritall1[niter,nsteps]
        erritall2[0:niter+1,nsteps] = err2[0:niter+1]
        erritall2[-1,nsteps] = erritall2[niter,nsteps]
        errIntLitall[0:niter+1,nsteps] = errIntL[0:niter+1]
        errIntLitall[-1,nsteps] = errIntLitall[niter,nsteps]
        errIntRitall[0:niter+1,nsteps] = errIntR[0:niter+1]
        errIntRitall[-1,nsteps] = errIntRitall[niter,nsteps]        
        
        if nsteps%printstep == 0:
            print(nsteps,t,niter)
            
    print("*** End of computation ***")        
    return uall1,uall2,tall,niterall,diff1all,diff2all,diffitall,erritall,erritall1,erritall2,errIntLitall,errIntRitall
def compareDDMs(errallASM,errall1ASM,errall2ASM,niterallASM,
                errallMSM,errall1MSM,errall2MSM,niterallMSM,
                errallMSMi,errall1MSMi,errall2MSMi,niterallMSMi,
                dt) :

    errASM = np.sqrt(dt)*np.linalg.norm(errallASM[-1,:])
    err1ASM = np.sqrt(dt)*np.linalg.norm(errall1ASM[-1,:])
    err2ASM = np.sqrt(dt)*np.linalg.norm(errall2ASM[-1,:])
    maxIterASM = np.amax(niterallASM)
    errMSM = np.sqrt(dt)*np.linalg.norm(errallMSM[-1,:])
    err1MSM = np.sqrt(dt)*np.linalg.norm(errall1MSM[-1,:])
    err2MSM = np.sqrt(dt)*np.linalg.norm(errall2MSM[-1,:])
    maxIterMSM = np.amax(niterallMSM)
    errMSMi = np.sqrt(dt)*np.linalg.norm(errallMSMi[-1,:])
    err1MSMi = np.sqrt(dt)*np.linalg.norm(errall1MSMi[-1,:])
    err2MSMi = np.sqrt(dt)*np.linalg.norm(errall2MSMi[-1,:])
    maxIterMSMi = np.amax(niterallMSMi)

    print("ASM")
    print(r"Entire Simulation --> Error in Omega   : %.3e" %errASM)
    print(r"Entire Simulation --> Error in Omega_1 : %.3e" %err1ASM)
    print(r"Entire Simulation --> Error in Omega_2 : %.3e" %err2ASM)
    print(r"First time step   --> Error in Omega   : %.3e" %errallASM[-1,1])
    print(r"First time step   --> Error in Omega_1 : %.3e" %errall1ASM[-1,1])
    print(r"First time step   --> Error in Omega_2 : %.3e" %errall2ASM[-1,1])
    print(r"Max nb of iter : %d" %maxIterASM)
    print("")

    print("MSM")
    print(r"Entire Simulation --> Error in Omega   : %.3e" %errMSM)
    print(r"Entire Simulation --> Error in Omega_1 : %.3e" %err1MSM)
    print(r"Entire Simulation --> Error in Omega_2 : %.3e" %err2MSM)
    print(r"First time step   --> Error in Omega   : %.3e" %errallMSM[-1,1])
    print(r"First time step   --> Error in Omega_1 : %.3e" %errall1MSM[-1,1])
    print(r"First time step   --> Error in Omega_2 : %.3e" %errall2MSM[-1,1])
    print(r"Max nb of iter : %d" %maxIterMSM)
    print("")

    print("MSMi")
    print(r"Entire Simulation --> Error in Omega   : %.3e" %errMSMi)
    print(r"Entire Simulation --> Error in Omega_1 : %.3e" %err1MSMi)
    print(r"Entire Simulation --> Error in Omega_2 : %.3e" %err2MSMi)
    print(r"First time step   --> Error in Omega   : %.3e" %errallMSMi[-1,1])
    print(r"First time step   --> Error in Omega_1 : %.3e" %errall1MSMi[-1,1])
    print(r"First time step   --> Error in Omega_2 : %.3e" %errall2MSMi[-1,1])
    print(r"Max nb of iter : %d" %maxIterMSMi)
    print("")
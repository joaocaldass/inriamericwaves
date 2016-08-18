
import numpy as np
def Flux(u,a,b):
    return a*u + b*u*u
def Fluxder(u,a,b):
    return a + 2.0*b*u
def Fluxderinv(u,a,b):
    return (u-1.*a)/(2.*b)
def Riemann(u,x,t,a,b):
    uint = np.zeros_like(x)
    for  i in range(0,x.size-1):
        #incl = x[i]/t
        incl = 0
        um = u[i]
        up = u[i+1]
        if um == up :
            uint[i] = um
        elif  um > up:
            sigma = (Flux(um,a,b) - Flux(up,a,b))/(um-up)
            if incl < sigma:
                uint[i] = um
            else:
                uint[i] = up
        else:
            if incl < Fluxder(um,a,b) :
                uint[i] = um
            elif incl > Fluxder(up,a,b) :
                uint[i] = up
            else:
                uint[i] = Fluxderinv(incl,a,b)
    #uint[0] = u[0]
    uint[x.size-1] = u[x.size-1]
    return uint
def Euler(u,dx,dt,periodic):
    umm = np.roll(u,1)
    if periodic :
        umm[0] = u[u.size-1]
    else :
        umm[0] = u[0]
    
    f = Flux(u)
    fmm = Flux(umm)
    
    u2 = u - dt/dx*(f-fmm)
    
    return u2
def getRKCoef(u,x,t,dx,dt,a,b,periodic):
    uint = Riemann(u,x,t,a,b)
    uintmm = np.roll(uint,1)
    if periodic :
        uintmm[0] = uintmm[u.size-1]
    else :
        uintmm[0] = u[0]
        uint[u.size-1] = u[u.size-1]
        
    f = Flux(uint,a,b)
    fmm = Flux(uintmm,a,b)
    return dt*(fmm-f)/dx
        
def RK4(u,x,t,dx,dt,a,b,periodic):
        
    k1 = getRKCoef(u,x,t,dx,dt,a,b,periodic)
    k2 = getRKCoef(u+k1/2,x,t,dx,dt,a,b,periodic)
    k3 = getRKCoef(u+k2/2,x,t,dx,dt,a,b,periodic)
    k4 = getRKCoef(u+k3,x,t,dx,dt,a,b,periodic)
    
    u2 = u + 1./6.*(k1+2*k2+2*k3+k4)
    
    return u2
def FourierSolver(u,t,dt,dx):
    nx = u.size-1
    #f0 = 1/dx
    #freq = np.linspace(0,nx,nx+1)*f0/(nx+1)
    #freq[(nx+1)/2+1:nx+1] = -np.flipud( freq[1:(nx+1)/2+1])
    uhat = np.fft.fft(u)
    uhat = uhat*np.exp(dt*1.j*np.power(np.fft.fftfreq(uhat.size,d=dt),3))
    #uhat = uhat*np.exp(dt*1.j*np.power(freq,3))
    u2 = np.fft.ifft(uhat)
#    if not (np.all(np.isreal(u2))):
#        print(u2)
#        sys.exit("Error in Fourier method")
    return np.real(u2)
import numpy.polynomial.chebyshev as cheb

def computeChebMatrix(x,u):
    N = x.size-1
    a = x[0]
    b = x[N]
    lamb = 2./(b-a)
    roots = (2.*x-(b+a))/(b-a)

    T = np.diag(np.ones(N+1))
    Troots = cheb.chebval(roots,T)

    Txroots = np.zeros(Troots.shape)
    Txxroots = np.zeros(Troots.shape)
    
    A = np.zeros(Troots.shape)
    
    cc = np.ones(N+1)
    cc[0] = .5
    cc[N] = .5
    
    #for i in range(N+1):
    #    for j in range(N+1):
    #        for n in range(j):
    #            if (n+j)%2 == 1:
    #                Txroots[j,i] = Txroots[j,i] + cc[n]*Troots[n,i]
    #        Txroots[j,i] = Txroots[j,i]*2.*j*lamb
    
    ccM = np.tile(cc.transpose(),(N+1,1)).transpose()
    cT = ccM*Troots
    
    
    Z1 = np.array([0,1])
    Z1 = np.tile(Z1,N/2)
    Z2 = np.roll(Z1,-1)
    Z2[0] = 0
    Z3 = np.vstack((Z1,Z2)).transpose()
    Z3 = np.tile(Z3,(1,N/2))

    col = np.zeros((N,1))
    col[:,0] = Z3[:,N-2]
    Z3 = np.concatenate((Z3,col),axis=1)

    lig = np.zeros((1,N+1))
    lig[0,:] = Z3[N-2,:]
    Z3 = np.concatenate((Z3,lig),axis=0)

    Z = np.tril(Z3)

    ZcT = np.dot(Z,cT)
    
    D = np.linspace(0,N,N+1)
    D = np.tile(D,(N+1,1)).transpose()
    
    Txroots = 2.*lamb*D*ZcT
    
    #print(np.linalg.norm(Txroots-Txroots2))
    
    
    cTxroots = ccM*Txroots
    A = ccM.transpose()*2./N*np.dot(cTxroots.transpose(),Troots)
    
      
#    for i in range(N+1):
#        for n in range(N+1):
#            for j in range(N+1):
#                if j > 0 and j < N:
#                    A[i,n] = A[i,n] + Txroots[j,i]*Troots[j,n]
#                else:
#                    A[i,n] = A[i,n] + .5*Txroots[j,i]*Troots[j,n]
#            A[i,n] = A[i,n]*(2.*cc[n])/N
    
    
    B =  np.dot(A,A)
    Bb = np.dot(A[:,1:N],A[1:N,:])
    #C = np.dot(A,Bb)
    C = np.dot(A,B)
    
    return A,B,C


def ChebyshevSolver(x,u,dt,dx,A,B,C,cl) :
    
    N = x.size-1
    
    M = dt*C + np.diag(np.ones(N+1))
    M[0,:] = A[0,:]
    M[N,:] = B[0,:]
    M[1,:] = 0
    M[1,0] = 1.
    
    rhs = np.copy(u)
    
    rhs[0] = cl[2]
    rhs[N] = cl[3]
    rhs[1] = cl[0]
    
    u2 = np.linalg.solve(M,rhs)
    
    return u2

def ChebyshevSolver2(x,u,dt,dx,A,B,C,cl) :
    
    N = x.size-1
    
    M = dt*C + np.diag(np.ones(N+1))
    M[0,:] = 0
    M[N,:] = 0
    M[:,0] = 0
    M[:,N] = 0
    M[0,0] = 1
    M[N,N] = 1
    
    M[N-1,1:N] = A[N-1,1:N]
    #M[1,1] = -1.
    #M[N,:] = np.zeros(N+1)
    #M[N,N] = 1.
    #M[N,N-1] = -1.
    
    cl[0] = u[0]
    cl[1] = u[N]
    rhs = np.copy(u) - dt*(B[:,0]*cl[0] + B[:,N]*cl[1] + C[:,0]*cl[2] + C[:,N]*cl[3])
    rhs[0] = u[0]
    rhs[N] = u[N]
    
    rhs[N-1] = cl[2]
    
    u2 = np.linalg.solve(M,rhs)
    
    return u2
import sys
import math


def IFD4Solver(u,t,dt,dx,BC):
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
    ######vvv[0] = 1. - 49./8.*k
    ######vvv[1] = 29.*k
    ######vvv[2] = -461./8.*k
    ######vvv[3] = 62.*k
    ######vvv[4] = -307./8.*k
    ######vvv[5] = 13.*k
    ######vvv[6] = -15./8.*k
    
    zzz = -np.flipud(vvv)
    zzz[nx] = 1. + 1.*k
    ######zzz[nx] = 1. + 49./8.*k

    M[0,:] = vvv
    M[1,:] = np.roll(vvv,1)
    M[2,:] = np.roll(vvv,2)
    
    M[nx,:] = zzz
    M[nx-1,:] = np.roll(zzz,-1)
    M[nx-2,:] = np.roll(zzz,-2)
    

    rhs = np.copy(u)
    
    ### Boundary conditions
    ## Structure : BC=[u(left),ux(left),uxx(left),alpha*u(left) + beta*ux(right),
    ##                 u(right),ux(right),uxx(right),alpha*u(right) + beta*ux(right),
    ##                 alpha,beta] 
    cntBC = 0
    if not math.isnan(BC[0]) : # u L
        cntBC = cntBC+1
        M[0,:] = 0.
        M[0,0] = 1.
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
        M[nx,:] = 0.
        M[nx,nx] = 1.
        rhs[nx] = BC[4]
    if not math.isnan(BC[5]) : # ux R
        cntBC = cntBC+1
        row = nx
        if not math.isnan(BC[4]):
            row = row - 1
        M[row,:] = 0.
        M[row,nx] = 1./dx
        M[row,nx-1] = -1./dx
        rhs[row] = BC[5]
    if not math.isnan(BC[6]) : # uxx R
        cntBC = cntBC+1
        row = nx
        if not math.isnan(BC[4]):
            row = row - 1
        if not math.isnan(BC[5]):
            row = row - 1
        M[row,:] = 0.
        M[row,nx] = 1./(dx*dx)
        M[row,nx-1] = -2. /(dx*dx)  
        M[row,nx-2] = 1./(dx*dx)
        rhs[row] = BC[6]
    if not math.isnan(BC[7]) : # Robin R
        cntBC = cntBC+1
        if not (math.isnan(BC[4]) and math.isnan(BC[5]) and math.isnan(BC[6])) :
            sys.exit("Error in right BC : Robin defined with Dirichlet and/or Neumann")
        M[nx,:] = 0.
        M[nx,nx] = BC[8] + BC[9]/dx + BC[10]/(dx*dx)
        M[nx,nx-1] = -BC[9]/dx - 2.*BC[10]/(dx*dx)
        M[nx,nx-2] = BC[10]/(dx*dx)
        rhs[nx] = BC[7]

     
    if cntBC != 3 :
        print(cntBC)
        print(BC)
        sys.exit("Wrong number of BC")
        
    
    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(suppress=True)
    
    u2 = np.linalg.solve(M,rhs)
    
    return u2
def IFD4SolverMain(u,t,dt,dx,BC):
    k1 = IFD4Solver(u,t,dt,dx,BC) - u
    k2 = IFD4Solver(u+dt*k1/2.,t,dt,dx,BC) - (u+k1/2.)
    k3 = IFD4Solver(u+dt*k2/2.,t,dt,dx,BC) - (u+k2/2.)
    k4 = IFD4Solver(u+dt*k3,t,dt,dx,BC) - (u+k3)
    
    u2 = u + dt/6.*(k1 + 2.*k2 + 2.*k3 + k4)
    
    return u2
def computeError(x,u,uref):
    
    nx = x.size-1
    niter = u.shape[1]
    
    ## Error in every point
    uerr = u - uref
    
    ## Norm of the error for each time step
    errNorm = np.linalg.norm(uerr,axis=0)
    
    ## Error in the right boundary
    errBoundary = uerr[nx,:]
    
    maxErrNorm = np.amax(np.absolute(errNorm))
    maxErrBoundary = np.amax(np.absolute(errBoundary))
        
    return uerr, errNorm, errBoundary, maxErrNorm, maxErrBoundary
def runRk4FVFourier(x,u,t0,tmax,a,b,periodic=1, bc=None, constantBC = True, vardt = True):
    
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
    #a = 1
    #b = 1


    #if not periodic:
     #   A,B,C = computeChebMatrix(x,u)
    
    if not vardt :
        umax = np.amax(np.absolute(u))
        dt = dx/(1.*a+2*b*umax) - eps
    
    while t<tmax:
        iter = iter + 1
        
        if vardt:
            umax = np.amax(np.absolute(u))
            dt = dx/(1.*a+2*b*umax) - eps     # CFL CONDITION

        t = t+dt
        u = RK4(u,x,t,dx,dt,a,b,periodic)
        
        if periodic :
            u = FourierSolver(u,t,dt,dx)
        else :
            if constantBC :
                BC = bc
            else :
                BC = bc[iter-1,:]
            u = IFD4Solver(u,t,dt,dx,BC)
            ###u = ChebyshevSolver(x,u,dt,dx,A,B,C,cl)

        uall = np.column_stack((uall,u))
        tall = np.hstack((tall,t*np.ones(1)))
        
        if iter%100 == 0:
            print(iter,t)
            
    print("*** End of computation ***")
    return uall,tall
def discretizeSpace(xmin,xmax,dx) :
    nx = int((xmax-xmin)/dx)
    x = np.linspace(xmin, xmax,nx+1)    
    return x
import matplotlib.pyplot as plt
from matplotlib import animation
from JSAnimation import IPython_display
import kdv

def plotSolutionError(x,u,uerr,errNorm,errBoundary,tall,xmin,xmax,ymin,ymax):
    
    plt.plot(tall,errNorm,label=r'$e_1$')
    plt.legend(loc=(1,0))
    plt.plot(tall,errBoundary,label=r'$e_2$')
    plt.legend(loc=(1,0))
    plt.xlabel("t")
    
    print("*** Plotting animation ...")
    
    fig = plt.figure()
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
    line1, = ax.plot([], [], lw=2, label="Solution")
    line2, = ax.plot([], [], lw=2, label=r'$u-u_{ref}$')
    ax.set_ylabel(r'$u$')
    title = ax.set_title(r'$t=0.0 s$')
    plt.legend()

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def animate(i):
        line1.set_data(x, u[:,i])
        title.set_text('t=%.3f'%(tall[i]))
        line2.set_data(x, uerr[:,i])
        return line1,line2

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                        frames=u.shape[-1], interval=300)
    
    #anim = kdv.plotAnimation(x,uerr,tall,xmin,xmax,ymin,ymax)
    
    return anim
def showWaveInfo(A,wvl,B,eps,xmin,xmax,N) :
    
    h0min = 1.5*A/eps
    h0max = B*wvl/(2.*np.pi)
    h0 = 0.5*(h0min+h0max)
    alpha2 = h0*h0/6.

    print(r'*** Wave info : ***')
    print(r'Amplitude : %f' %A)
    print(r'Wavelength : %f' %wvl)
    print(r'1/(2.*Wavelength^2) : %f' %(1./(2.*wvl*wvl)))
    print(r'h0min : %f' %h0min)
    print(r'h0max : %f' %h0max)
    print(r'eps : %f' %eps)
    print(r'B : %f' %B)
    print(r'alpha^2 : %f' %alpha2)
    print(r'Space steps : %d' %N)
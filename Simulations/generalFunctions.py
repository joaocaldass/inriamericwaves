
import matplotlib.pyplot as plt
from matplotlib import animation
from JSAnimation import IPython_display

def plotAnimation(x,u,t,xmin,xmax,ymin,ymax,ylabel,save = False, path = "") :
    
    print("*** Plotting animation ...")
    
    fig = plt.figure()
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
    line, = ax.plot([], [], lw=2)
    ax.set_ylabel(ylabel)
    title = ax.set_title(r'$t=0.0 s$')

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(x, u[:,i])
        title.set_text('t=%.3f'%(t[i]))
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                        frames=u.shape[-1], interval=300)
    
    if save:
        anim.save(path, writer='ffmpeg', fps=30)
    
    return anim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from JSAnimation import IPython_display

def plotAnimationNSolutions(N,x,u,t,xmin,xmax,ymin,ymax,lb,ylabel,location=0,savePath=None,ddm=[]) :
    """
    Animate N solutions, all of them defined in the same spatial domain x and the spatial domain t
    
    *Inputs :
        - N : number of solutions
        - x : array of spatial points
        - t : array of instants
        - u = np.array([u1,u2,...]) : array of arrays containing the solutions. Each solution ui must have the shape MxT,
            where M = x.size and T = t.size
        - xmin,xmax : x interval for plotting
        - ymin,ymax : y interval for plotting
        - lb = ["lb1",...,"lbN"] : labels for the legend
        - ylabel = labelf for y axis
        - location (optional) : position of the legend (location = 0 as default gives an optimal position)
        - savePath (optional) : if not None, save the animation in a video specified by savePath
        - ddm : array with the interfaces for the plot
    """
    
    print("*** Plotting animation ...")
    
    ## nb of interfaces
    nb_int = len(ddm)
    for i in range(nb_int):
        lb.append('interface {}'.format(i))
    
    fig = plt.figure()
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
    line = np.array([])
    for i in range(N+nb_int):
        line = np.append(line,ax.plot([], [], lw=2, label=lb[i])) 
    ax.set_ylabel(ylabel)
    title = ax.set_title(r'$t=0.0 s$')
    plt.legend(loc=location)

    def init():
        for i in range(N+nb_int):
            line[i].set_data([], [])
        return line

    def animate(i):
        for j in range(N):
            line[j].set_data(x, u[j,:,i])
        for j in range(N, N+nb_int):
            line[j].set_data([ddm[j-N], ddm[j-N]], [ymin, ymax])
        title.set_text('t=%.3f'%(t[i]))
        return line

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                        frames=u.shape[-1], interval=300)
    
    if savePath != None:
        anim.save(savePath, writer="ffmpeg", fps=30)
    
    return anim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from JSAnimation import IPython_display

def plotAnimationNSolutionsDiffDomain(N,x,u,t,xmin,xmax,ymin,ymax,lb,ylabel,location=0,savePath=None) :
    """
    Animate N solutions, defined on different spatial domains x but in the same spatial domain t.
    Nevertelhess, all of the xi must have the same size and all the ui must have the same shape
    (if necessary, complete xi with values outside of [xmin,xmax] and ui with zeros).
    
    *Inputs :
        - N : number of solutions
        - x = np.array([x1,x2,...])
        - t : array of instants
        - u = np.array([u1,u2,...]) : array of arrays containing the solutions. Each solution ui must have the shape MxT,
            where M = x.size and T = t.size
        - xmin,xmax : x interval for plotting
        - ymin,ymax : y interval for plotting
        - lb = ["lb1",...,"lbN"] : labels for the legend
        - ylabel = labelf for y axis
        - location (optional) : position of the legend (location = 0 as default gives an optimal position)
        - savePath (optional) : if not None, save the animation in a video specified by savePath
    """
    
    print("*** Plotting animation ...")
    
    fig = plt.figure()
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
    line = np.array([])
    for i in range(N):
        line = np.append(line,ax.plot([], [], lw=2, label=lb[i])) 
    ax.set_ylabel(ylabel)
    title = ax.set_title(r'$t=0.0 s$')
    plt.legend(loc=location)

    def init():
        for i in range(N):
            line[i].set_data([], [])
        return line

    def animate(i):
        for j in range(N):
            line[j].set_data(x[j], u[j][:,i])
        title.set_text('t=%.3f'%(t[i]))
        return line

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                        frames=u[0].shape[-1], interval=300)

    if savePath != None:
        anim.save(savePath, writer="ffmpeg", fps=30)
    
    return anim
# save plot of N solutions in the instant t_n to an image file
def saveSnapshotNsolutions(N,n,x,u,t,lbl,xlbl,ylbl,path,ext="png",
                           xmin=None,xmax=None,ymin=None,ymax=None,legloc=0) :
    plt.figure()
    if xmin == None:
        xmin = x[0]
    if xmax == None:
        xmax = x[-1]
    if ymin == None :
        ymin = np.amin(u)
    if ymax == None : 
        ymax = np.amax(u)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    for i in range(N) :
        plt.plot(x,u[i,:,n],label=lbl[i])
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend(loc=legloc)
    plt.title(r't = %.3f'%t[n])
    plt.savefig(path+"."+ext)
# save plot of N solutions in the instant t_n to an image file
def saveSnapshotNsolutionsDiffDomain(N,n,x,u,t,lbl,xlbl,ylbl,path,
                           xmin,xmax,ymin,ymax,ext="png",legloc=0) :
    plt.figure()
    ##if xmin == None:
    ##    xmin = x[0]
    ##if xmax == None:
    ##    xmax = x[-1]
    ##if ymin == None :
    ##    ymin = np.amin(u)
    ##if ymax == None : 
    ##    ymax = np.amax(u)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    for i in range(N) :
        plt.plot(x[i],u[i][:,n],label=lbl[i])
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    lgd = plt.legend(loc=legloc)
    plt.title(r't = %.3f'%t[n])
    plt.savefig(path+"."+ext,bbox_extra_artists=(lgd,),bbox_inches='tight')
# save plot of one solution in instants t_n, n in ns, to an image file
def saveSnapshots(ns,x,u,t,xlbl,ylbl,title,path,ext="png") :
    plt.figure()
    for n in ns :
        plt.plot(x,u[:,n],label=r't = %.3f'%t[n])
    plt.xlim((x[0],x[-1]))
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend()
    plt.title(title)
    plt.savefig(path+"."+ext)
# save N arbitrary plots to an image file
def saveNgraphs(N,x,y,lbl,xlbl,ylbl,title,path,ext="png",xmin=None,xmax=None,legloc=0) :
    plt.figure()
    for i in range(N) :
        plt.plot(x[i,:],y[i,:],label=lbl[i])
    if xmin == None:
        xmin = x[0]
    if xmax == None:
        xmax = x[-1]
    plt.xlim((xmin,xmax))
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend(loc=legloc)
    plt.title(title)
    plt.savefig(path+"."+ext)
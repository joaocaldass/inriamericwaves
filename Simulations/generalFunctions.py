
import matplotlib.pyplot as plt
from matplotlib import animation
from JSAnimation import IPython_display

def plotAnimation(x,u,t,xmin,xmax,ymin,ymax,ylabel) :
    
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
    
    return anim
import matplotlib.pyplot as plt
from matplotlib import animation
from JSAnimation import IPython_display

def plotAnimationTwoSolutions(x,u1,u2,t1,t2,xmin,xmax,ymin,ymax,lb1,lb2,ylabel) :
    
    print("*** Plotting animation ...")
    
    fig = plt.figure()
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
    line1, = ax.plot([], [], lw=2, label=lb1)
    line2, = ax.plot([], [], lw=2, label=lb2)    
    ax.set_ylabel(ylabel)
    title = ax.set_title(r'$t=0.0 s$')
    plt.legend()

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1,line2,

    def animate(i):
        line1.set_data(x, u1[:,i])
        line2.set_data(x, u2[:,i])
        title.set_text('t=%.3f'%(t1[i]))
        return line1,line2,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                        frames=u1.shape[-1], interval=300)
    
    return anim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from JSAnimation import IPython_display

def plotAnimationNSolutions(N,x,u,t,xmin,xmax,ymin,ymax,lb,ylabel,location=0) :
    
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
            line[j].set_data(x, u[j,:,i])
        title.set_text('t=%.3f'%(t[i]))
        return line

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                        frames=u.shape[-1], interval=300)
    
    return anim
# save plot of one solution in instants t_n, n in ns, to an image file
def saveSnapshots(ns,x,u,t,xlbl,ylbl,title,path,ext="png",legloc=0) :
    plt.figure()
    for n in ns :
        plt.plot(x,u[:,n],label=r't = %.3f'%t[n])
    plt.xlim((x[0],x[-1]))
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend(loc=legloc)
    plt.title(title)
    plt.savefig(path+"."+ext)
# save plot of N solutions in the instant t_n to an image file
def saveSnapshotNsolutions(N,n,x,u,t,lbl,xlbl,ylbl,path,ext="png",xmin=None,xmax=None,ymin=None,ymax=None,legloc=0) :
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
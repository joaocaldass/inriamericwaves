
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

import matplotlib.pyplot as plt
from matplotlib import animation
from JSAnimation import IPython_display

def plotAnimation(x,u,t,xmin,xmax,ymin,ymax) :
    
    print("*** Plotting animation ...")
    
    fig = plt.figure()
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
    line, = ax.plot([], [], lw=2)
    ax.set_ylabel(r'$u$')
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
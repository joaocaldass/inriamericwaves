import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib import animation
from JSAnimation import IPython_display
g = 9.81
def cmEquation(x,hl,hr,g) :
    return -8.*g*hr*x*x*np.power(np.sqrt(g*hl)-x,2) + np.power(x*x-g*hr,2)*(x*x+g*hr)
def SWAnalyticalSolutionWet(x,t,x0,hl,hr,g) :
    c = np.sqrt(g*hl)
    cm = fsolve(cmEquation,4/3*hl,args=(hl,hr,g))[0]
    xa = x0 - t*c
    xb = x0 + t*(2.*c - 3.*cm)
    xc = x0 + t*(2*cm*cm*(c-cm)/(cm*cm-g*hr))

    h = np.where(x<=xa, hl, np.where(x<xb,4./(9.*g)*np.power(c-(x[:]-x0)/(2.*t),2),np.where(x<xc,cm*cm/g,hr)))
    u = np.where(x<=xa, 0,np.where(x<xb, 2./3.*(c+(x[:]-x0)/t),np.where(x<xc, 2*(c-cm), 0)))
    
    return h,u
def SWAnalyticalSolutionDry(x,t,x0,hl,g) :
    c = np.sqrt(g*hl)
    xa = x0 - t*c
    xb = x0 + 2.*t*c
    g = 9.81

    h = np.where(x[:]<=xa, hl, np.where(x<xb,4./(9.*g)*np.power(c-(x[:]-x0)/(2.*t),2),0))
    u = np.where(x[:]<=xa, 0,np.where(x<xb, 2./3.*(c+(x[:]-x0)/t),0))

    
    return h,u
def plot_dry(hl):
    fig = plt.figure()
    ax = plt.axes(xlim=(-10,10), ylim=(-0.1, 10.1))    
    x = np.linspace(-10,10,100)    
    line, = ax.plot([],[])
    def animate(i):
        h1,u1 = SWAnalyticalSolutionDry(x,i/10.0,0.0,hl,9.81)
        line.set_data(x,h1)
        return line,

    anim = animation.FuncAnimation(fig, animate, frames=20, interval=45)
    return anim
def plot_wet(hl,hr):
    fig = plt.figure()
    ax = plt.axes(xlim=(-10,10), ylim=(-0.1, 10.1))
    line, = ax.plot([],[])
    x = np.linspace(-10,10,100)
    g = 9.81
    def animate(i):    
        t = i/10.
        x0 = 0.0    
        h1,u1 = SWAnalyticalSolutionWet(x,t,x0,hl,hr,g)
        line.set_data(x,h1)
        return line,

    return animation.FuncAnimation(fig, animate, frames=20, interval=45)
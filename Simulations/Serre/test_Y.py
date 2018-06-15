import Zt_tools as Z

ub = 0.5
# ub = 0.02445528381075566
h0 = 0.32749170021928925

nx = 100
xmin = -15.413344025217135
xmax =  15.413344025217135

dx = 0.3082668805043429
dt = 0.05

Nf = 10000

ps = Z.Parameters(h0, dx, nx, xmin, xmax, dt, Nf)

Y = Z.compute_Y2(ps, ub)
Z.plot_Y(Y, Nf)


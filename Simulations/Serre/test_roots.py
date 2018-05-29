################################################################################
#
# Small python programm for testing roots properties of the polynomial from the
# discretsation of Serre linearised equations
#
################################################################################

# Dependencie
import numpy as np
import matplotlib.pyplot as plt

# Defining constants of the problem
g = 9.81
dx = 0.01
dt = 0.05
h0 = 1.
ht = - h0**2/3

# Dimensional and adimensional constants for the reformulated polynomial
a = 4*ht/(g*h0)
b = 4*(dx**2)/(g*h0)
s = lambda z: (2./dt) * (z-1.)/(z+1.)

# Testing sign of s^2
#  theta = np.linspace(0, 2*np.pi, 301)
#  plt.plot(2*b*(1 - np.cos(theta)) - c, label="den")
#  plt.plot(2*a*(1 - np.cos(theta)) - 2*np.cos(2*theta) + 8*np.cos(theta) - 6, label="num")
#  plt.legend()
#  plt.show()

# Studying the polynomial
X = np.linspace(-3, 3, 51)
Y = np.linspace(-3, 3, 51)
theta = np.linspace(0, 2*np.pi, 1001)
P = [0 + 0j for i in range(5)]

root = [[] for i in range(4)]

plt.plot(np.cos(theta), np.sin(theta))

for x in X:
  for y in Y:

    z = x + y*1j
    sc = s(z)
    if abs(sc) < 1:
      continue

    P[0] = 1. + 0j
    P[1] = -a*(sc**2)
    P[2] = (sc**2)*(2*a - b) - 2.
    P[3] = -a*(sc**2)
    P[4] = 1. + 0j
    R = np.roots(P)
    R = np.sort(R)[::-1]
    mR = [abs(r) for r in R]
    i = 0
    while len(mR) > 0:
      for k in range(4):
        r = R[k]
        if (r not in root[i]) and (abs(r) == max(mR)):
          root[i].append(r)
          mR.remove(abs(r))
          i += 1
          break

xi = np.array([np.zeros(len(root[i])) for i in range(4)])
yi = np.array([np.zeros(len(root[i])) for i in range(4)])
for i in range(4):
  for k in range(len(root[i])):
    xi[i][k] = root[i][k].real
    yi[i][k] = root[i][k].imag

for i in range(4):
  plt.plot(xi[i], yi[i], '.', label = "r{}".format(i))

plt.legend()
plt.show()

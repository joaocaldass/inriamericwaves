################################################################################
#
# A Python environment for testing Transparent Boundary Conditions (TBC) on the
# BOSZ linearized equation.
#
# WARNING : Written for Python 2.7
#
# This package contains tools for the Z-transform performed in linearBOSZ.py.
#
################################################################################

# dependencies
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft
from scipy.interpolate import pade

class Parameters:
  """
  A small parameter class to adapt the code to Zt_tools.py
  """

  def __init__(self, h0, dx, nx, xmin, xmax, dt, Nf):
    ## pb parameters
    self.h0 = h0
    self.dx = dx
    self.nx = nx
    self.xmin = xmin
    self.xmax = xmax
    self.dt = dt
    self.Nf = Nf

    ## boussinesq
    alpha = -0.53753*self.h0
    self.hb = alpha*(alpha/2. + self.h0)
    self.ht = self.h0*(alpha*alpha/2. + alpha*self.h0 + self.h0**2/3.)
    self.xi_pow = 0

    ## personnal parameters
    self.ht = - (h0**2)/3
    self.g = 9.81

def compute_Y3(ps):
  """
  Computes the exact convolution variables before apprxomating them.
  """

  print "*** Starting computations of Ys"

  N = ps.Nf

  ## creating the polynomial basis
  # dimensional and adimensional constants for the reformulated polynomial
  a = ps.h0*(ps.dx**2) / ps.ht + 0j
  b = (ps.hb*(ps.dx**2)) / (ps.g*ps.ht) + 0j
  c = (ps.dx**4) / (ps.g*ps.ht) + 0j
  s = lambda z: (2./ps.dt) * (z-1.)/(z+1.)
  P = [0 + 0j for i in range(5)]

  ## initialization
  Y = np.zeros((9, N), dtype=complex)
  K = np.zeros((9, N), dtype=complex)
  omegaN = np.exp(2*np.pi*1j/N) # unity root
  rrc = 1.001  # radius of the circle for computing Z^{-1}

  for n in range(N):

    z = rrc*(omegaN**n)

    sc = s(z)
    P[0] = 1. + 0j
    P[1] = a - b*(sc**2) - 4.
    P[2] = (sc**2)*(2*b-c) - 2.*a + 6.
    P[3] = a - b*(sc**2) - 4.
    P[4] = 1. + 0j
    R = np.roots(P)

    # sorting the roots from the largest to the smallest (in terms of modulus)
    mR = [abs(r) for r in R]
    root = []
    while len(mR) > 0:
      for r in R:
        if (r not in root) and (abs(r) == max(mR)):
          root.append(r)
          mR.remove(abs(r))
          break

    #  ar = abs(np.array(root))
    #  assert((ar[0] > ar[1]) and (ar[1] > 1) and (1 > ar[2]) and (ar[2] > ar[3]))

    # re-ordering
    root = root[::-1]

    # computing the kernels
    ## roots smaller than 1
    K[0,n] = root[0] + root[1]
    K[1,n] = K[0,n]**2
    K[2,n] = root[0] * root[1]
    assert(abs(K[2,n]) < 1)
    K[3,n] = K[2,n]**2
    ## roots greater than 1
    K[4,n] = (root[2] + root[3]) / (root[2] * root[3])
    K[5,n] = K[4,n]**2
    K[6,n] = 1. / (root[2] * root[3])
    assert(abs(K[6,n]) < 1)
    K[7,n] = K[6,n]**2
    K[8,n] = (root[2] + root[3]) / ((root[2] * root[3])**2)

    for i in range(9):
      if i < 4:
        K[i,n] = ((1+1/z)**ps.xi_pow)*K[i,n]
      else:
        K[i,n] = ((1+1/z)**ps.xi_pow)*K[i,n]

  print "*** Computation of Ys (in Fourier space) --> done\n"

  print "*** Applying iFFT on Ys"

  for i in range(9):
    Kn = ifft(K[i,:])
    for n in range(N):
      Y[i,n] = (rrc**n)*Kn[n]

  print " *  max(Y1) = {}".format(np.amax(Y.real[:4,]))
  print " *  max(Y2) = {}".format(np.amax(Y.real[4:,]))

  print "*** iFFT --> done\n"

  # keeping the real part
  return Y.real

def compute_Y2(ps, ub):
  """
  Computes the exact convolution variables before apprxomating them.
  This is the version for the linearized Serre equation with ub != 0.
  """

  print "*** Starting computations of Ys"

  ## creating the polynomial basis
  # dimensional and adimensional constants for the reformulated polynomial
  s = lambda z: (2./ps.dt) * (z-1.)/(z+1.)
  P = [0 + 0j for i in range(5)]
  N = ps.Nf

  alpha = (ps.h0**2*ub)/(2*ps.dx**3)
  beta = ps.h0**2/ps.dx**2

  ## initialization
  Y = np.zeros((9, N), dtype=complex)
  K = np.zeros((9, N), dtype=complex)
  omegaN = np.exp(2*np.pi*1j/N) # unity root
  rrc = 1.001  # radius of the circle for computing Z^{-1}

  for n in range(N):

    z = rrc*(omegaN**n)

    sc = s(z)
    P[0] = - (1./3.) * alpha * (1./s(z))
    P[1] =   (2./3.) * alpha * (1./s(z)) - (1./3.) * beta
    P[2] =   (2./3.) * beta + 1
    P[3] = - (2./3.) * alpha * (1./s(z)) - (1./3.) * beta
    P[4] =   (1./3.) * alpha * (1./s(z))
    R = np.roots(P)

    # sorting the roots from the largest to the smallest (in terms of modulus)
    mR = [abs(r) for r in R]
    root = []
    while len(mR) > 0:
      for r in R:
        if (r not in root) and (abs(r) == max(mR)):
          root.append(r)
          mR.remove(abs(r))
          break

    #  ar = abs(np.array(root))
    #  assert((ar[0] > ar[1]) and (ar[1] > 1) and (1 > ar[2]) and (ar[2] > ar[3]))

    # re-ordering
    root = root[::-1]

    # computing the kernels
    ## roots smaller than 1
    K[0,n] = root[0] + root[1]
    K[1,n] = K[0,n]**2
    K[2,n] = root[0] * root[1]
    assert(abs(K[2,n]) < 1)
    K[3,n] = K[2,n]**2
    ## roots greater than 1
    K[4,n] = (root[2] + root[3]) / (root[2] * root[3])
    K[5,n] = K[4,n]**2
    K[6,n] = 1. / (root[2] * root[3])
    assert(abs(K[6,n]) < 1)
    K[7,n] = K[6,n]**2
    K[8,n] = (root[2] + root[3]) / ((root[2] * root[3])**2)

    #  for i in range(9):
    #    if i < 4:
    #      K[i,n] = ((1+1/z)**ps.xi_pow)*K[i,n]
    #    else:
    #      K[i,n] = ((1+1/z)**ps.xi_pow)*K[i,n]

  print "*** Computation of Ys (in Fourier space) --> done\n"

  print "*** Applying iFFT on Ys"

  for i in range(9):
    Kn = ifft(K[i,:])
    for n in range(N):
      Y[i,n] = (rrc**n)*Kn[n]

  print " *  max(Y1) = {}".format(np.amax(Y.real[:4,]))
  print " *  max(Y2) = {}".format(np.amax(Y.real[4:,]))

  print "*** iFFT --> done\n"

  # keeping the real part
  return Y.real


def compute_K(dx, h0):
  """
  Computes only the kernels when there is no time dependency to solve in z for
  the Serre splitting equations.
  """

  print "*** Starting computations of Ks"

  alpha = 36.*(dx**2)/(h0**2) + 30
  P = [1, 16, -alpha, -16, 1]
  R = np.roots(P)
  R = sorted(R, key=abs)[::-1]
  root = [abs(r) for r in R]
  root = root[::-1]

  print " *  roots of P :", root

  K = np.zeros((9,1))

  K[0,0] = root[0] + root[1]
  K[1,0] = K[0,0]**2
  K[2,0] = root[0] * root[1]
  assert(abs(K[2,0]) < 1)
  K[3,0] = K[2,0]**2
  ## roots greater than 1
  K[4,0] = (root[2] + root[3]) / (root[2] * root[3])
  K[5,0] = K[4,0]**2
  K[6,0] = 1. / (root[2] * root[3])
  assert(abs(K[6,0]) < 1)
  K[7,0] = K[6,0]**2
  K[8,0] = (root[2] + root[3]) / ((root[2] * root[3])**2)

  print " *  kernels :\n", K

  print "*** done"

  return K

def compute_Y(ps):
  """
  Computes the exact convolution variables before apprxomating them.
  """

  print "*** Starting computations of Ys"

  ## creating the polynomial basis
  # dimensional and adimensional constants for the reformulated polynomial
  a = 4*ps.ht/(ps.g*ps.h0)
  b = 4*(ps.dx**2)/(ps.g*ps.h0)
  s = lambda z: (2./ps.dt) * (z-1.)/(z+1.)
  P = [0 + 0j for i in range(5)]
  N = ps.Nf

  ## initialization
  Y = np.zeros((9, N), dtype=complex)
  K = np.zeros((9, N), dtype=complex)
  omegaN = np.exp(2*np.pi*1j/N) # unity root
  rrc = 1.001  # radius of the circle for computing Z^{-1}

  for n in range(N):

    z = rrc*(omegaN**n)

    sc = s(z)
    P[0] = 1. + 0j
    P[1] = -a*(sc**2)
    P[2] = (sc**2)*(2*a - b) - 2.
    P[3] = -a*(sc**2)
    P[4] = 1. + 0j
    R = np.roots(P)

    # sorting the roots from the largest to the smallest (in terms of modulus)
    mR = [abs(r) for r in R]
    root = []
    while len(mR) > 0:
      for r in R:
        if (r not in root) and (abs(r) == max(mR)):
          root.append(r)
          mR.remove(abs(r))
          break

    #  ar = abs(np.array(root))
    #  assert((ar[0] > ar[1]) and (ar[1] > 1) and (1 > ar[2]) and (ar[2] > ar[3]))

    # re-ordering
    root = root[::-1]

    # computing the kernels
    ## roots smaller than 1
    K[0,n] = root[0] + root[1]
    K[1,n] = K[0,n]**2
    K[2,n] = root[0] * root[1]
    assert(abs(K[2,n]) < 1)
    K[3,n] = K[2,n]**2
    ## roots greater than 1
    K[4,n] = (root[2] + root[3]) / (root[2] * root[3])
    K[5,n] = K[4,n]**2
    K[6,n] = 1. / (root[2] * root[3])
    assert(abs(K[6,n]) < 1)
    K[7,n] = K[6,n]**2
    K[8,n] = (root[2] + root[3]) / ((root[2] * root[3])**2)

    #  for i in range(9):
    #    if i < 4:
    #      K[i,n] = ((1+1/z)**ps.xi_pow)*K[i,n]
    #    else:
    #      K[i,n] = ((1+1/z)**ps.xi_pow)*K[i,n]

  print "*** Computation of Ys (in Fourier space) --> done\n"

  print "*** Applying iFFT on Ys"

  for i in range(9):
    Kn = ifft(K[i,:])
    for n in range(N):
      Y[i,n] = (rrc**n)*Kn[n]

  print " *  max(Y1) = {}".format(np.amax(Y.real[:4,]))
  print " *  max(Y2) = {}".format(np.amax(Y.real[4:,]))

  print "*** iFFT --> done\n"

  # keeping the real part
  return Y.real

def plot_Y(Y, N, log_scale = False, export = False, save = False):
  """
  Plots the Ys, either in normal scale or log scale. If export is set to True,
  the Ys are exported to a csv file, which path is set with path.
  """

  if not log_scale:
    for i in range(4):
      plt.plot(Y[i,:N], label="i = {}".format(i))
    plt.legend()
    if save:
      plt.savefig("Y_right.pdf")
      plt.clf()
    else:
      plt.show()
    for i in range(4,9):
      plt.plot(Y[i,:N], label="i = {}".format(i))
    plt.legend()
    if save:
      plt.savefig("Y_left.pdf")
      plt.clf()
    else:
      plt.show()

  elif log_scale:
    for i in range(4):
      plt.plot(Y[i,:], label="i = {}".format(i))
    plt.plot(10000*np.linspace(1, N, N)**(-3./2.))
    plt.yscale('log')
    plt.xscale('log')
    plt.axis((0, 5*N, 10**(-20), 1000))
    plt.legend()
    if save:
      plt.savefig("Y_log_right.pdf")
      plt.clf()
    else:
      plt.show()
    for i in range(4, 8):
      plt.plot(Y[i,:], label="i = {}".format(i))
    plt.plot(10000*np.linspace(1, N, N)**(-3./2.))
    plt.yscale('log')
    plt.xscale('log')
    plt.axis((0, 5*N, 10**(-20), 1000))
    plt.legend()
    if save:
      plt.savefig("Y_log_left.pdf")
      plt.clf()
    else:
      plt.show()

def exp_coeff(Y, L, nu, n_stop):
  """
  Computes the coefficient for the exponential approximation of the convolution
  coefficients.
  """

  print "*** Starting Pade approximation"

  # defining the formal serie
  # with the real part of the coefficients
  f = [[Y[i,n].real for n in range(nu, n_stop)] for i in range(8)]

  # matrix to store q_{i,l} and b_{i,l}
  q = np.zeros((8, L), dtype=complex)
  b = np.zeros((8, L), dtype=complex)

  for i in range(8):
    P, Q = pade(f[i], L)
    assert(P.o == L-1 and Q.o == L)
    root = Q.roots
    for l in range(L):
      q[i,l] = root[l]
      #  assert(abs(q[i,l]) > 1)
      b[i,l] = -(P(q[i,l])/Q.deriv()(q[i,l]))*(q[i,l]**(nu-1))


  print " *  max(b) = {}, max(q) = {}".format(np.amax(abs(b)), np.amax(abs(q)))

  print "*** Pade approximation --> done\n"

  print "*** Checking Pade approximation"

  Y_appr = np.zeros((8, n_stop), dtype=complex)
  for i in range(8):
      for n in range(nu):
        Y_appr[i,n] = Y[i,n]
      for n in range(nu, n_stop):
        Y_appr[i,n] = np.sum([b[i,l]*(q[i,l]**(-n)) for l in range(L)])

  # testing the approximation
  test = True
  for i in range(8):
    error = np.linalg.norm(abs(Y[i,:] - Y_appr[i,:]))
    print " *  i = {} --> |Y-Y_appr| = {}".format(i, error)
    if error > 10**(-3):
      test = False
  assert(test == True)
  print "*** Pade approximation --> checked\n"

  return q, b

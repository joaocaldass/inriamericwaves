################################################################################
#
# A Python environment for testing Transparent Boundary Conditions (TBC) on the
# BOSZ linearized equation.
#
# WARNING : Written for Python 2.7
#
# This package contains tools to compute convolutions.
#
################################################################################

# dependencies
import numpy as np

def convolution_exact(nit, Y, U):
  """
  Computes convolution of Y (convolution coefficients) and U between nu and nit.
  If interface is set to a string, then other indexes are filled so that the
  convolution can be used in an interface at idx_int.
  """


  Ct = np.zeros((9,15))
  for k in range(nit):
    Ct[0,-2] += Y[0,k]*U[-2, nit-k]
    Ct[1,-3] += Y[1,k]*U[-3, nit-k]
    Ct[2,-3] += Y[2,k]*U[-3, nit-k]
    Ct[3,-5] += Y[3,k]*U[-5, nit-k]
    Ct[4,1]   += Y[4,k]*U[1, nit-k]
    Ct[5,2]   += Y[5,k]*U[2, nit-k]
    Ct[6,2]   += Y[6,k]*U[2, nit-k]
    Ct[7,4]   += Y[7,k]*U[4, nit-k]
    Ct[8,3]   += Y[8,k]*U[3, nit-k]

  return Ct



import numpy as np
from scipy import special

def WaveLengthDepth(k,a0,a1):
    """
        Returns the wavelength and mean depth
        of a cnoidal wave with parameters k,a0,a1
    """
    kappa = np.sqrt(3*a1)/(2*np.sqrt(a0*(a0+a1)*(a0+(1-k*k)*a1)))
    h0 = a0+ a1*special.ellipe(k)/special.ellipk(k)    
    lam = 2.*special.ellipk(k)/kappa
    return lam,h0

def analyticalSolution(x,t,k,a0,a1):
    """
        Returns the cnoidal solution with parameters k,a0,a1
        at (x,t) (possibly arrays)        
    """
    g = 9.81
    kappa = np.sqrt(3*a1)/(2*np.sqrt(a0*(a0+a1)*(a0+(1-k*k)*a1)))
    h0 = a0+ a1*special.ellipe(k)/special.ellipk(k)
    c = np.sqrt(g*a0*(a0+a1)*(a0+(1.-k*k)*a1))/h0
    
    sn,cn,dn,ph = special.ellipj(kappa*(x-c*t),k)
    h = a0+a1*dn**2
    u = c*(1-h0/h)
    
    return h,u
    
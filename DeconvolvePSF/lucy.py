import numpy as np
import scipy
import numpy.lib.index_tricks as itricks
import pdb
from WavefrontPSF.psf_evaluator import Moment_Evaluator
#from scipy.signal import convolve2d as convolve

def convolve(A, B):
    """ Performs a convolution of two 2D arrays """
    C = np.fft.ifft2(np.fft.fft2(A) * np.fft.fft2(B))
    C = np.fft.fftshift(C)
    C = C / np.sum(C)
    return np.real(C)

def convolveStar(A, B):
    """ Performs a convolution of two 2D arrays, but take the complex conjugate of B """
    C = np.fft.ifft2(np.fft.fft2(A) * np.conjugate(np.fft.fft2(B)))
    C = np.fft.fftshift(C)
    C = C / np.sum(C)
    return np.real(C)

def calcChi2(PSF,psi_r,phi_tilde,beta,mu0):
    """ Calculate chi2 between PSF convolved with restored image and measured image
    """
    imageC = beta * convolve(PSF,psi_r)
    diffImage = phi_tilde-imageC
    varianceImage = phi_tilde + mu0
    chi2 = np.sum(diffImage*diffImage/varianceImage)
    return chi2

def makeGaussian(shape,Mxx,Myy,Mxy):
    """ Return a 2-d Gaussian function, centered at 0, with desired 2-nd order moments
    """
    ny = shape[0]
    nx = shape[1]
    ylo = -ny/2. + 0.5
    yhi = ny/2. - 0.5
    xlo = -nx/2. + 0.5 
    xhi = nx/2. - 0.5
    yArr,xArr = itricks.mgrid[ylo:yhi:1j*ny,xlo:xhi:1j*nx]
    rho = Mxy/np.sqrt(Mxx*Myy)

    gaussian = np.exp( -((yArr*yArr)/Myy + (xArr*xArr)/Mxx - 2.*rho*xArr*yArr/np.sqrt(Mxx*Myy))/(2.*(1-rho*rho))  )
    return gaussian

def makeMask(image,sigma,nsigma=3.):
    """ build a mask from the noisy Image
    """
    mask = np.where(image>nsigma*sigma,1.,0.)

    # use working copy
    maskcopy = mask.copy()

    # mask edge
    maskcopy[0,:] = 0.
    maskcopy[-1,:] = 0.
    maskcopy[:,0] = 0.
    maskcopy[:,-1] = 0.

    # demand that pixels have 3 neighbors also above 3sigma
    shape = mask.shape
    for j in range(1,shape[0]-1):
        for i in range(1,shape[1]-1):
            if mask[j,i]==1:
                # check 8 neighbors
                nNeigh = mask[j+1,i-1] + mask[j+1,i] + mask[j+1,i+1] + mask[j,i-1] + mask[j,i+1] + mask[j-1,i-1] + mask[j-1,i] + mask[j-1,i+1]
                if nNeigh<3:
                    maskcopy[j,i] = 0.

    # fill return array
    mask = maskcopy.copy()
    return mask

def deconvolve(PSF,phi_tilde,psi_0=None,mask=None,mu0=0.0,niterations=10,convergence=-1,chi2Level=0.0,extra=False):
    """ Implementation of the Richardson-Lucy deconvolution algorithm.
    Notation follows Lucy 1974, Eqn 15 and 14.  Add  noise term following
    Snyder et al 1993.

    Arguments
    ---------
    PSF          known Point Spread Function
    phi_tilde    measured object
    psi_0        starting guess for deconvolution
    mask         =0 for bins where we know that recovered image has no flux 
    mu0          background noise estimate
    """

    # normalize PSF
    PSF = PSF / np.sum(PSF)

    # if no initial guess, make one from 2nd moments of input image - PSF
    if psi_0 is None:
        #Turns out Gaussians are a bad initial guess, still unclear as to why
        #Can use the image itself as the initial guess, also works fine.
        psi_r = np.ones(PSF.shape)
        
    else:
        # initial guess
        psi_r = np.abs(psi_0)

    # mask starting guess
    if mask is not None:
        psi_r = psi_r * mask

    # normalize starting guess
    psi_r = psi_r / np.sum(psi_r)
    
    #TODO Maybe this should be an error instead of a warning.
    if np.any(np.isnan(psi_r)):
        raise RuntimeWarning("NaN in initial guess, skip this value. ")
                

    # mask image too
    if mask is not None:
        phi_tilde = phi_tilde * mask
        
    # find normalization for measured image
    beta = np.sum(phi_tilde)
        
        
    # now iterate, either until convergence reached or fixed number of iterations are done
    psiByIter = []
    diffByIter = []
    chi2ByIter = []
    iteration = 0
    continueTheLoop = True
    while continueTheLoop: 

        # calculate next approximation to psi
        phi_r = beta*convolve(psi_r,PSF) + mu0
        #fixing a possible bug in noisy deocnv
        psi_rplus1 = psi_r * convolveStar(beta*(phi_tilde)/phi_r,PSF)

        # mask the next iteration
        if mask != None:
            psi_rplus1 = psi_rplus1 * mask
        
        # normalize it
        psi_rplus1 = psi_rplus1 / np.sum(psi_rplus1)

        # check for convergence if desired
        #Why are the psiByIter appends inside the convergence test?
        if convergence>0:
            # compare psi_r and psi_rplus1
            psiByIter.append(psi_rplus1)
            diff = np.sum(np.abs(psi_rplus1 - psi_r))
            diffByIter.append(diff)
            if diff<convergence:
                continueTheLoop = False

        # also calculate how close to a solution we are
        chi2 = calcChi2(PSF,psi_rplus1,phi_tilde,beta,mu0)
        chi2ByIter.append(chi2)
        if chi2<chi2Level:
            continueTheLoop = False
        
        # check for Chi2 level
                

        # always check number of iterations
        if iteration==niterations-1:
            continueTheLoop = False

        # save for next iteration
        iteration+=1     
        psi_r = np.array(psi_rplus1)  # does a deepcopy


    #TODO rescale deconv by flux

    # we are done!

    #check to see if the deconv failed
    evaluator = Moment_Evaluator()

    resid_moments = evaluator(psi_rplus1)

    #TODO what to do if makeGaussian throws an error?
    # subtract 2nd order moments in quadrature, use an object with the difference
    Mxx = resid_moments['Mxx'][0]
    Myy = resid_moments['Myy'][0]
    Mxy = resid_moments['Mxy'][0]

    #print Mxx, Myy, Mxy
    if any(np.isnan(x) for x in [Mxx, Myy, Mxy]):
        raise RuntimeWarning("Deconvolution Failed.")


    if extra:
        return psi_rplus1,diffByIter,psiByIter,chi2ByIter
    else:
        return psi_rplus1        
 

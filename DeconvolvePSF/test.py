#
# test Richardon-Lucy code
#

from donutlib.makedonut import makedonut
from DeconvolvePSF.lucy import deconvolve, convolve, makeMask
from WavefrontPSF.psf_evaluator import Moment_Evaluator

import numpy as np
import numpy.random
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.04f}'.format(x, y, z)


def realImage(image,flux,background):
    imageBN = flux * image / np.sum(image)  + background
    shape = image.shape
    nranval = numpy.random.normal(0.0,1.0,shape)
    noisyImage = imageBN + nranval*numpy.sqrt(imageBN) - background
    return noisyImage


# calculate moments
evaluator = Moment_Evaluator()

# size of stamps
nPixels = 32
nbin = 8*nPixels

inputDict = {'writeToFits':False,'iTelescope':0,'nZernikeTerms':37,'nbin':nbin,'nPixels':nPixels,'pixelOverSample':8,'scaleFactor':1.,'nEle':1.0e6, 'background':0., 'randomFlag':False}
makeAd = makedonut(**inputDict)

# PSF - w/ aberrations, and an r0 equiv to 0.566" FWHM
inputDictPSF = {'rzero':0.247, 'nEle':1.0e6, 'background':0., 'randomFlag':False, 'ZernikeArray':[0.,0.,0.,0.5,-0.12,.2,.18,-0.09,0.6,-0.15,0.2]}
PSF = makeAd.make(**inputDictPSF)

# star - one aberrations, just a little atmosphere, with an r0 equiv to 0.7" seeing
inputDictStar = {'rzero':0.20, 'nEle':1.0e6, 'background':0., 'randomFlag':False,'ZernikeArray':[0.,0.,0.,0.5,1.0]}
star = makeAd.make(**inputDictStar)

# now convolve these two to make an image - no noise
image = convolve(star,PSF)
image = image/np.sum(image)


# normalize and then add background and noise and then subtract background
flux = 1.e6      # e- total flux
background = 6000.  # e- per pixel
numpy.random.seed(238411)
noisyImage = realImage(image,flux,background)

# try a better starting guess - based on our knowledge of the PSF
image_moments = evaluator(image)
PSF_moments = evaluator(PSF)

# subtract FWHM in quadrature, use an object with this size
fwhm_diff = np.sqrt(np.power(image_moments['fwhm'][0],2)-np.power(PSF_moments['fwhm'][0],2))
r0estimate = 0.14/fwhm_diff
inputDictPsiT = {'rzero':r0estimate, 'nEle':1.0e6, 'background':0., 'randomFlag':False, 'ZernikeArray':[0.,0.,0.,0.0]}
psi0 = makeAd.make(**inputDictPsiT)

# build a mask from the noisy Image
mask = makeMask(noisyImage,np.sqrt(background),2.)

# now try R-L
starEstimate,diffs,psiByIter,chi2ByIter = deconvolve(PSF,noisyImage,psi_0=psi0,mask=None,mu0=background,convergence=1.0e-3,chi2Level=0.,niterations=50,extra=True)

print "Diff", diffs
print "Chi2", chi2ByIter

f1,axArr = plt.subplots(2,2)
im00 = axArr[0,0].imshow(PSF,interpolation='none',origin='lower',cmap='gray',norm=LogNorm(vmin=1.0e6/1.0e4, vmax=1.0e6))
axArr[0,0].format_coord = Formatter(im00)
im01 = axArr[0,1].imshow(star/np.sum(star),interpolation='none',origin='lower',cmap='gray',norm=LogNorm(vmin=1.0e-4, vmax=1.0))
axArr[0,1].format_coord = Formatter(im01)
im10 = axArr[1,0].imshow(noisyImage,interpolation='none',origin='lower',cmap='gray',norm=LogNorm(vmin=flux/1.0e4, vmax=flux))
axArr[1,0].format_coord = Formatter(im10)
im11 = axArr[1,1].imshow(starEstimate,interpolation='none',origin='lower',cmap='gray',norm=LogNorm(vmin=1.0e-4, vmax=1.0))
axArr[1,1].format_coord = Formatter(im11)    

f1.show()


f2,ax2Arr = plt.subplots(1,10)
for i in range(10):
    ax2Arr[i].imshow(psiByIter[i],interpolation='none',origin='lower',cmap='gray',norm=LogNorm(vmin=1.0e-4, vmax=1.0))
f2.show()
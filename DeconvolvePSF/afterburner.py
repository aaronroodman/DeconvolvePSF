#TODO Add intepreter here
#@Author Sean McLaughlin
desc ='''
This module is the first I'm writing for my project for the winter quarter of 2016 in Aaron Roodman's group.
This is built on top of previous work conducted by Aaron and his graduate student Chris Davis. They have
developed WavefrontPSF, which estimates the optical contribution of the PSF with a relatively simple model.
It was found that the optical portion was not as signifcant a fraction of the total PSF as hoped,
so some sort of afterburner is going to need to be added. After the optical portion has been deconvolved
from the observed stars (using Richardson-Lucy deconvolution), the remainder will be treated as the "atmospheric"
portion of the psf. This module load in preprocessed observed stars, run WavefrontPSF on them, deconvolve
the optical PSF, then run PSFEX (a packaged PSF modeler) on the residual.

TODO Details on actually running the module.
'''

from argparse import ArgumentParser
parser = ArgumentParser(description = desc)

parser.add_argument('expid', metavar = 'expid', type = int, help =\
                    'ID of the exposure to analyze')

args = vars(parser.parse_args())

import numpy as np
from itertools import izip
from optical_model import getOpticalPSF
from glob import glob
from astropy.io import fits #TODO check if I should support pyfits
from lucy import deconvolve

#get optical PSF
optPSFStamps, starStamps = getOpticalPSF(args['expid'])

print starStamps[0].type

print optPSFStamps.shape, type(optPSFStamps)
print starStamps.shape, type(starStamps)

vignettes = []
full_cats = []
ext = 2

for file in files:
    hdulist = fits.open(file)
    full_cats.append(hdulist[ext].data)
    vignettes.append(hdulist[ext].data['VIGNET'])

#vignettes = np.array(vignettes)

print len(vignettes)
for i in xrange(10):
    print vignettes[i].shape
print optPSFStamps.shape

from matplotlib import pyplot as plt
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(vignettes[0][0,:,:])
plt.subplot(1,2, 2)
plt.title('Optical')
plt.imshow(optPSFStamps[0,:,:])
plt.show()

for optPSFStamp, vignette in izip(optPSFStamps, vignettes):
    aptPSFEst = deconvolve(optPSFStamp,vignette,psi_0=None,mask=None,mu0=0,convergence=1.0e-3,chi2Level=0.,niterations=50)
    break

print aptPSFEst.shape

from matplotlib import pyplot as plt
plt.subplot(1,3,1)
plt.title('Original')
plt.imshow(vignette)
plt.subplot(1,3, 2)
plt.title('Optical')
plt.imshow(optPSFStamp)
plt.subplot(1,3,3)
plt.title('Remainder')
plt.imshow(aptPSFEst)

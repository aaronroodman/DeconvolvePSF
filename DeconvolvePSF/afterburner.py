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
optPSFStamps, full_cat = getOpticalPSF(args['expid'])

vignettes = np.zeros((optPSFStamps.shape[0], 32,32))
i=0
#TODO See if this is slow and optomize
for rec_arr in full_cat:
    for v in rec_arr['VIGNET']:
        #TODO Check for off by one errors and centering.
        #Slice 63x63 down to 32x32 so deconv will work.
        #TODO Turn sliced off pixels into background estimate
        vignettes[i] = v[15:47, 15:47]
        i+=1

for optPSFStamp, vignette in izip(optPSFStamps, vignettes):
    aptPSFEst,diffs,psiByIter,chi2ByIter = deconvolve(optPSFStamp,vignette,psi_0=None,mask=None,mu0=6e3,convergence=1e-3,chi2Level=0.,niterations=50, extra= True)
    break


from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

plt.subplot(1,3,1)
plt.title('Original')
plt.imshow(vignette,interpolation='none',origin='lower',cmap='gray',norm=LogNorm(vmin=1.0e-4, vmax=1.0))
plt.subplot(1,3, 2)
plt.title('Optical')
plt.imshow(optPSFStamp,interpolation='none',origin='lower',cmap='gray',norm=LogNorm(vmin=1.0e-4, vmax=1.0))
plt.subplot(1,3,3)
plt.title('Remainder')
plt.imshow(aptPSFEst,interpolation='none',origin='lower',cmap='gray',norm=LogNorm(vmin=1.0e-4, vmax=1.0))

plt.show()

f2,ax2Arr = plt.subplots(1,10)
print len(ax2Arr), len(psiByIter)
for i in xrange(10):
    ax2Arr[i].imshow(psiByIter[i],interpolation='none',origin='lower',cmap='gray',norm=LogNorm(vmin=1.0e-4, vmax=1.0))
f2.show()

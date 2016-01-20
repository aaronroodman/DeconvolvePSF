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

print 'Opts Calculated.'

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
    break

aptPSFEst_list = []
for optPSFStamp, vignette in izip(optPSFStamps, vignettes):
    aptPSFEst_small,diffs,psiByIter,chi2ByIter = deconvolve(optPSFStamp,vignette,psi_0=None,mask=None,mu0=6e3,convergence=1e-3,chi2Level=0.,niterations=50, extra= True)
    aptPSFEst = np.zeros((63,63))
    aptPSFEst[15:47, 15:47] = aptPSFEst_small
    aptPSFEst_list.append(aptPSFEst.flatten())
    i-=1
    if i ==0:
        break

print 'Deconv done.'

for rec_arr in full_cat:
    for j in xrange(i): 
        rec_arr.VIGNET[j] = aptPSFEst_list[j] 
    break

print 'Copy done.'

from astropy.io import fits
#NOTE Not sure if I need to do a more involved write. 
#Could save myself the trouble by having the hdulist objects before I modify them

#fits.writeto('test.fits', full_cat[0])

#NOTE Depreceated. Use BinTableHDU.from_columns
tbhdu = fits.BinTableHDU.from_columns(rec_arr[0])
tbhdu.header.set('EXTNAME', 'LDAC_OBJECTS', 'a name')

prihdr = fits.Header()
prihdu = fits.PrimaryHDU(header=prihdr)
thdulist = fits.HDUList([prihdu,tbhdu])
thdulist.writeto('test.fits',clobber=True)

'''
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
for i in xrange(10):

    plt.subplot(1,3,1)
    plt.title('Original')
    plt.imshow(vignette,interpolation='none',origin='lower',cmap='gray')
    plt.subplot(1,3, 2)
    plt.title('Optical')
    plt.imshow(optPSFStamp,interpolation='none',origin='lower',cmap='gray')
    plt.subplot(1,3,3)
    plt.title('Remainder')
    plt.imshow(aptPSFEst,interpolation='none',origin='lower',cmap='gray')

    plt.show()

nPlots = len(psiByIter)
nPlots = 10 if 10< nPlots else nPlots

f2,ax2Arr = plt.subplots(1,nPlots)
for i in xrange(nPlots):
    ax2Arr[i].imshow(psiByIter[i],interpolation='none',origin='lower',cmap='gray',norm=LogNorm(vmin=1.0e-4, vmax=1.0))
f2.show()
'''

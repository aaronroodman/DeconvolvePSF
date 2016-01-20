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

#TODO Details on actually running the module.
'''
#TODO (General) I switch between camelcase and underscores. Python style guide calls for underscores, so fix.
from argparse import ArgumentParser
parser = ArgumentParser(description = desc)

parser.add_argument('expid', metavar = 'expid', type = int, help =\
                    'ID of the exposure to analyze')

parser.add_argument('outputDir', metavar = 'outputDir', type = str, help =\
                    'Directory to store outputs.')

args = vars(parser.parse_args())

from os import path
if not path.isdir(args['outputDir']):
    raise IOError("The directory %s does not exist."%args['outputDir'])

if args['outputDir'][-1]  != '/':
    args['outputDir']+='/'

import numpy as np
from itertools import izip
from optical_model import getOpticalPSF
from lucy import deconvolve

#get optical PSF
optPSFStamps, metaHDUList = getOpticalPSF(args['expid'])

print 'Opts Calculated.' ,

vignettes = np.zeros((optPSFStamps.shape[0], 32,32))

vigIdx=0
#TODO See if this is slow and optomize
for hdulist in metaHDUList:
    for v in hdulist[2].data['VIGNET']:
        #TODO Check for off by one errors and centering.
        #Slice 63x63 down to 32x32 so deconv will work.
        #TODO Turn sliced off pixels into background estimate
        vignettes[vigIdx] = v[15:47, 15:47]
        vigIdx+=1

aptPSFEst_list = []
for optPSFStamp, vignette in izip(optPSFStamps, vignettes):
    aptPSFEst_small,diffs,psiByIter,chi2ByIter = deconvolve(optPSFStamp,vignette,psi_0=None,mask=None,mu0=6e3,convergence=1e-3,chi2Level=0.,niterations=50, extra= True)
    aptPSFEst = np.zeros((63,63))
    aptPSFEst[15:47, 15:47] = aptPSFEst_small
    aptPSFEst_list.append(aptPSFEst)

    print aptPSFEst_small.mean(), aptPSFEst_small.std()
    print '*-_-'*10

#TODO np.array(aptPSFst_list?)

print 'Deconv done.'
i =0
for hdulist in metaHDUList:
    #for j in xrange(len(rec_arr)):
    for j in xrange(hdulist[2].data.shape[0]):
        hdulist[2].data['VIGNET'][j] = aptPSFEst_list[i+j]
    i+=hdulist[2].data.shape[0]

    #Make new filename from old one.
    originalFname = hdulist.filename().split('/')[-1]#just get the filename, not the path
    originalFnameSplit = originalFname.split('_')
    originalFnameSplit[-1] = '_seldeconv.fits'
    hdulist.writeto(args['outputDir']+''.join(originalFnameSplit), clobber = True)

print 'Copy and write done.'


#TODO Clear temporary files?

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

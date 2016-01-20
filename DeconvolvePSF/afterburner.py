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
from optical_model import get_optical_psf
from lucy import deconvolve

#get optical PSF
optpsf_stamps, meta_hdulist = get_optical_psf(args['expid'])

print 'Opts Calculated.' ,

vignettes = np.zeros((optpsf_stamps.shape[0], 32,32))

vig_idx=0
#TODO See if this is slow and optomize.
for hdulist in meta_hdulist:
    #TODO Check for off by one errors and centering.
    #TODO Turn sliced off pixels into background estimate
    '''
    for v in hdulist[2].data['VIGNET']:
        #Slice 63x63 down to 32x32 so deconv will work.
        vignettes[vig_idx] = v[15:47, 15:47]
        vig_idx+=1
    '''
    list_len = hdulist[2].data.shape[0]
    vignettes[vig_idx:vig_idx+list_len] = hdulist[2].data['VIGNET'][:, 15:47, 15:47]
    vig_idx+=list_len

atmpsf_list = []
for optpsf, vignette in izip(optpsf_stamps, vignettes):
    atmpsf_small,diffs,psiByIter,chi2ByIter = deconvolve(optpsf,vignette,psi_0=None,mask=None,mu0=6e3,convergence=1e-3,chi2Level=0.,niterations=50, extra= True)
    atmpsf = np.zeros((63,63))
    atmpsf[15:47, 15:47] = atmpsf_small
    atmpsf_list.append(atmpsf)

#TODO np.array(aptPSFst_list?)

print 'Deconv done.'
atmpsf_idx =0
for hdulist in meta_hdulist:
    #for j in xrange(len(rec_arr)):
    '''
    for j in xrange(hdulist[2].data.shape[0]):
        hdulist[2].data['VIGNET'][j] = atmpsf_list[i+j]
    '''
    list_len = hdulist[2].data.shape[0]
    hdulist[2].data['VIGNET'] = atmpsf_list[atmpsf_idx:atmpsf_idx+list_len]
    atmpsf_idx+=list_len

    #Make new filename from old one.
    original_fname = hdulist.filename().split('/')[-1]#just get the filename, not the path
    original_fname_split = original_fname.split('_')
    original_fname_split[-1] = '_seldeconv.fits'
    hdulist.writeto(args['outputDir']+''.join(original_fname_split), clobber = True)

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
    plt.imshow(optpsf,interpolation='none',origin='lower',cmap='gray')
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

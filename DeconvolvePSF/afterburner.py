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
#May want to rename to tmp, since that's what the files really are.
parser.add_argument('outputDir', metavar = 'outputDir', type = str, help =\
                    'Directory to store outputs.')

args = vars(parser.parse_args())

#Ensure provided dir exists
from os import path
if not path.isdir(args['outputDir']):
    raise IOError("The directory %s does not exist."%args['outputDir'])

if args['outputDir'][-1]  != '/':
    args['outputDir']+='/'

import numpy as np
from itertools import izip
from optical_model import get_optical_psf
from lucy import deconvolve, convolve
from subprocess import call
from psfex import PSFEx
from glob import glob
import cPickle as pickle

#For identifying "bad stars"
import warnings
warnings.filterwarnings('error')

print 'Starting.'

#get optical PSF
optpsf_stamps, meta_hdulist = get_optical_psf(args['expid'])

print 'Opts Calculated.' 

vignettes = np.zeros((optpsf_stamps.shape[0], 32,32))

#extract star vignettes from the hdulists
vig_idx=0
hdu_lengths = np.zeros((62,))
for ccd_num, hdulist in enumerate(meta_hdulist):
    #TODO Check for off by one errors and centering.
    #TODO Turn sliced off pixels into background estimate

    list_len = hdulist[2].data.shape[0]
    sliced_vig  = hdulist[2].data['VIGNET'][:, 15:47, 15:47] #slice to same size as stamps
    sliced_vig[sliced_vig<-1000] = 0 #set really negative values to 0
    sliced_vig = sliced_vig/sliced_vig.sum((1,2))[:, None, None] #normalize
    vignettes[vig_idx:vig_idx+list_len] = sliced_vig 
    vig_idx+=list_len
    vig_shape = hdulist[2].data['VIGNET'][0].shape
    #print 'CCD: %d\tVignette Shape:(%d, %d)'%(ccd_num+1, vig_shape[0], vig_shape[1] )
    hdu_lengths[ccd_num] = list_len

#Calculate the atmospheric portion of the psf
atmpsf_list = []
for idx, (optpsf, vignette) in enumerate(izip(optpsf_stamps, vignettes)):
    #atmpsf_small,diffs,psiByIter,chi2ByIter = deconvolve(optpsf,vignette,psi_0=None,mask=None,mu0=6e3,convergence=1e-3,chi2Level=0.,niterations=50, extra= True)
    atmpsf = np.zeros((63,63))
    try:
        atmpsf_small = deconvolve(optpsf,vignette,psi_0=None,mask=None,mu0=6e3,convergence=1e-3,chi2Level=0.,niterations=50, extra= False)
        atmpsf[15:47, 15:47] = atmpsf_small
    except RuntimeWarning:
       #TODO What should I do on a failure?
        print 'Failed on %d'%idx
        pass
        

    atmpsf_list.append(atmpsf)

atmpsf_list =  np.array(atmpsf_list)

print 'Deconv done.'

#now, insert the atmospheric portion back into the hdulists, and write them to disk
#PSFEx needs the information in those lists to run correctly.
atmpsf_idx =0
for hdulist in meta_hdulist:
    list_len = hdulist[2].data.shape[0]
    hdulist[2].data['VIGNET'] = atmpsf_list[atmpsf_idx:atmpsf_idx+list_len]
    atmpsf_idx+=list_len

    #Make new filename from old one.
    original_fname = hdulist.filename().split('/')[-1]#just get the filename, not the path
    original_fname_split = original_fname.split('_')
    original_fname_split[-1] = 'seldeconv.fits'
    hdulist.writeto(args['outputDir']+'_'.join(original_fname_split), clobber = True)

print 'Copy and write done.'



#call psfex
psfex_path = '/nfs/slac/g/ki/ki22/roodman/EUPS_DESDM/eups/packages/Linux64/psfex/3.17.3+0/bin/psfex'
psfex_config = '/afs/slac.stanford.edu/u/ec/roodman/Astrophysics/PSF/desdm-plus.psfex'
#TODO This is gonna run on all *.fits in the outputdir. If the user doesn't want that uh... then what?
#TODO Actaully want to call on just the ones with this expid
command_list = [psfex_path, args['outputDir']+'*.fits', "-c", psfex_config]

#If shell != True, the wildcard won't work
psfex_return= call(' '.join(command_list), shell = True)
psfex_success = True if psfex_return==0 else False
print 'PSFEx Call Successful: %s'%psfex_success

#TODO Clear temporary files?

#no use continuing if the psfex call failed.
if not psfex_success:
    from sys import exit
    exit(1)

#Now, load in psfex's work, and reconolve with the optics portion. 
psf_files = glob(args['outputDir']+'*.psf')
atmpsf_list = []
#TODO Check that files are in the same order as the hdulist
for file, hdulist in  izip(psf_files, meta_hdulist):
    pex = PSFEx(file)
    for yimage, ximage in izip(hdulist[2].data['Y_IMAGE'], hdulist[2].data['X_IMAGE']):
        atmpsf = np.zeros((32,32))
        #psfex has a tendency to return images of weird and varying sizes
        #This scheme ensures that they will all be the same 32x32 by zero padding
        #assumes the images are square and smaller than 32x32
        #Proof god is real and hates observational astronomers.
        atmpsf_small = pex.get_rec(ximage, yimage)
        atm_shape = atmpsf_small.shape[0] #assumed to be square
        pad_amount = (32-atmpsf_small.shape[0])/2
        atmpsf[pad_amount:32-(pad_amount+atm_shape%2),pad_amount:32-(pad_amount+atm_shape%2) ] = atmpsf_small
        atmpsf_list.append(atmpsf)


atmpsf_list = np.array(atmpsf_list)

stars = []#TODO what to do with these
for idx, (optpsf, atmpsf) in enumerate(izip(optpsf_stamps, atmpsf_list)):
    try:
        stars.append(convolve(optpsf, atmpsf))
    except ValueError:
        for ccd_num, hdu_len in enumerate( hdu_lengths) :
            if hdu_len > idx:
                print 'Failed on CCD %d Image %d'%(ccd_num+1, idx)
                break
            else:
                idx-=hdu_len
        raise

np.savetxt(args['outputDir']+'%s_stars.pkl'%args['expid'], np.array(stars), delimiter=',')
np.savetxt(args['outputDir']+'%s_opt.pkl'%args['expid'], optpsf_stamps, delimiter=',')
np.savetxt(args['outputDir']+'%s_atm.pkl'%args['expid'], atmpsf_list, delimiter=',')

'''
from matplotlib import pyplot as plt
for star in stars:
    im = plt.imshow(star, cmap = plt.get_cmap('afmhot'), interpolation = 'none')
    plt.colorbar(im)
    plt.savefig(args['outputDir']+'%s_star.png'%args['expid'])
'''
print 'Done'

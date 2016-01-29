#!/nfs/slac/g/ki/ki06/roodman/Software/anaconda/bin/python
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
from os import path, mkdir
if not path.isdir(args['outputDir']):
    raise IOError("The directory %s does not exist."%args['outputDir'])

if args['outputDir'][-1]  != '/':
    args['outputDir']+='/'

#Make new dir to store files from this run
if not path.isdir(args['outputDir']+'00%d/'%args['expid']):
    try:
        mkdir(args['outputDir']+'00%d/'%args['expid'])
    except OSError:
        print 'Failed making directory; using original output directory.'
    else:
        args['outputDir']+='00%d/'%args['expid']
else:
    args['outputDir']+='00%d/'%args['expid']



import numpy as np
from itertools import izip
from optical_model import get_optical_psf
from lucy import deconvolve, convolve
from subprocess import call
from psfex import PSFEx
from glob import glob

#For identifying "bad stars"
import warnings
warnings.filterwarnings('error')

print 'Starting.'

#get optical PSF
optpsf_stamps, meta_hdulist = get_optical_psf(args['expid'])

#np.save(args['outputDir']+'%s_opt_test.npy'%args['expid'], optpsf_stamps)

print 'Opts Calculated.' 

vignettes = np.zeros((optpsf_stamps.shape[0], 32,32))

#extract star vignettes from the hdulists
vig_idx=0
hdu_lengths = np.zeros((62,))
for ccd_num, hdulist in enumerate(meta_hdulist):
    #TODO Turn sliced off pixels into background estimate

    list_len = hdulist[2].data.shape[0]

    sliced_vig  = hdulist[2].data['VIGNET'][:, 15:47, 15:47] #slice to same size as stamps
    sliced_vig[sliced_vig<-1000] = 0 #set really negative values to 0; it's a mask
    sliced_vig = sliced_vig/sliced_vig.sum((1,2))[:, None, None] #normalize
    vignettes[vig_idx:vig_idx+list_len] = sliced_vig 
    vig_idx+=list_len
    vig_shape = hdulist[2].data['VIGNET'][0].shape
    #print 'CCD: %d\tVignette Shape:(%d, %d)'%(ccd_num+1, vig_shape[0], vig_shape[1] )

    hdu_lengths[ccd_num] = list_len

#Calculate the atmospheric portion of the psf
resid_list = []
bad_stars = set() #keep idx's of bad stars
for idx, (optpsf, vignette) in enumerate(izip(optpsf_stamps, vignettes)):
    #resid_small,diffs,psiByIter,chi2ByIter = deconvolve(optpsf,vignette,psi_0=None,mask=None,mu0=6e3,convergence=1e-3,chi2Level=0.,niterations=50, extra= True)
    resid = np.zeros((63,63))
    try:
        resid_small = deconvolve(optpsf,vignette,psi_0=None,mask=None,mu0=6e3,convergence=1e-3,chi2Level=0.,niterations=50, extra= False)
        resid[15:47, 15:47] = resid_small
    except RuntimeWarning: #Some will fail
        bad_stars.add(idx)
        #TODO check how this mask works
        resid = np.ones((63,63))*-9999 #forcing a mask

        cp_idx = idx#still need this, since we're going to keep iterating.
        for ccd_num, hdu_len in enumerate( hdu_lengths) :
            if hdu_len > idx:
                print 'Deconvolve failed on CCD %d Image %d'%(ccd_num+1, cp_idx)
                break
            else:
                cp_idx-=hdu_len
        pass

    resid_list.append(resid)

resid_list =  np.array(resid_list)

print 'Deconv done.'

#now, insert the atmospheric portion back into the hdulists, and write them to disk
#PSFEx needs the information in those lists to run correctly.

resid_idx =0
for hdulist in meta_hdulist:
    list_len = hdulist[2].data.shape[0]
    hdulist[2].data['VIGNET'] = resid_list[resid_idx:resid_idx+list_len]
    resid_idx+=list_len

    #Make new filename from old one.
    original_fname = hdulist.filename().split('/')[-1]#just get the filename, not the path
    original_fname_split = original_fname.split('_')
    original_fname_split[-1] = 'seldeconv.fits'
    hdulist.writeto(args['outputDir']+'_'.join(original_fname_split), clobber = True)

print 'Copy and write done.'

#call psfex
psfex_path = '/nfs/slac/g/ki/ki22/roodman/EUPS_DESDM/eups/packages/Linux64/psfex/3.17.3+0/bin/psfex'
psfex_config = '/afs/slac.stanford.edu/u/ec/roodman/Astrophysics/PSF/desdm-plus.psfex'

command_list = [psfex_path, args['outputDir']+'*.fits', "-c", psfex_config]

#If shell != True, the wildcard won't work
psfex_return= call(' '.join(command_list), shell = True)
psfex_success = True if psfex_return==0 else False
print 'PSFEx Call Successful: %s'%psfex_success

#no use continuing if the psfex call failed.
if not psfex_success:
    from sys import exit
    exit(1)

#Now, load in psfex's work, and reconolve with the optics portion. 
psf_files = sorted(glob(args['outputDir']+'*.psf'))
atmpsf_list = []
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

stars = []
for idx, (optpsf, atmpsf) in enumerate(izip(optpsf_stamps, atmpsf_list)):
    #TODO if idx in bad_stars: continue
    #Not sure if I should skip the bads
    try:
        stars.append(convolve(optpsf, atmpsf))
    except ValueError:
        for ccd_num, hdu_len in enumerate( hdu_lengths) :
            if hdu_len > idx:
                print 'Convolve failed on CCD %d Image %d'%(ccd_num+1, idx)
                break
            else:
                idx-=hdu_len
        raise

np.save(args['outputDir']+'%s_stars.npy'%args['expid'], np.array(stars))
np.save(args['outputDir']+'%s_opt.npy'%args['expid'], optpsf_stamps)
np.save(args['outputDir']+'%s_atm.npy'%args['expid'], atmpsf_list)

'''
from matplotlib import pyplot as plt
for star in stars:
    im = plt.imshow(star, cmap = plt.get_cmap('afmhot'), interpolation = 'none')
    plt.colorbar(im)
    plt.savefig(args['outputDir']+'%s_star.png'%args['expid'])
'''
print 'Done'

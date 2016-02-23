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

#TODO I like to do imports after argparse, chris put these before. Probably not a big difference
#The difference is fairly significant. -h is slow with all these imports before it.
from WavefrontPSF.psf_interpolator import Mesh_Interpolator
from WavefrontPSF.digestor import Digestor
from WavefrontPSF.psf_evaluator import Moment_Evaluator
from WavefrontPSF.donutengine import DECAM_Model_Wavefront
from glob import glob
from itertools import izip
from psfex import PSFEx
import pandas as pd
from astropy.io import fits
import numpy as np
from optical_model import get_optical_psf
from lucy import deconvolve, convolve
from subprocess import call


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

#TODO same thing with outputDir?
expid = args['expid']

#Make new dir to store files from this run
if not path.isdir(args['outputDir']+'00%d/'%expid):
    try:
        mkdir(args['outputDir']+'00%d/'%expid)
    except OSError:
        print 'Failed making directory; using original output directory.'
    else:
        args['outputDir']+='00%d/'%expid
else:
    args['outputDir']+='00%d/'%expid

# import numpy as np
# from itertools import izip
# from optical_model import get_optical_psf
# from lucy import deconvolve, convolve
# from subprocess import call
# from psfex import PSFEx
# from glob import glob

#For identifying "bad stars"
import warnings
warnings.filterwarnings('error')

print 'Starting.'

#get optical PSF
optpsf_stamps, meta_hdulist = get_optical_psf(expid)

#np.save(args['outputDir']+'%s_opt_test.npy'%expid, optpsf_stamps)

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
    #vig_shape = hdulist[2].data['VIGNET'][0].shape
    #print 'CCD: %d\tVignette Shape:(%d, %d)'%(ccd_num+1, vig_shape[0], vig_shape[1] )

    hdu_lengths[ccd_num] = list_len

#Calculate the atmospheric portion of the psf
resid_list = []
bad_stars = set() #keep idx's of bad stars
for idx, (optpsf, vignette) in enumerate(izip(optpsf_stamps, vignettes)):
    #resid_small,diffs,psiByIter,chi2ByIter = deconvolve(optpsf,vignette,psi_0=None,mask=None,mu0=6e3,convergence=1e-3,chi2Level=0.,niterations=50, extra= True)
    resid = np.zeros((63,63))
    try:
        #this makes initial guess be all ones
        #TODO consider vignette as original guess, result sorta depends on noise
        background = vignette[vignette< vignette.mean()+vignette.std()]
        resid_small = deconvolve(optpsf,vignette,mask=None,mu0=background.mean(),convergence=1e-4,niterations=50, extra= False)

        resid[15:47, 15:47] = resid_small
    except RuntimeWarning: #Some will fail
        bad_stars.add(idx)
        #TODO check how this mask works
        resid = np.ones((63,63))*-9999 #forcing a mask

        cp_idx = idx#still need this, since we're going to keep iterating.
        for ccd_num, hdu_len in enumerate( hdu_lengths) :
            if hdu_len > cp_idx:
                print 'Deconvolve failed on CCD %d Image %d'%(ccd_num+1, cp_idx)
                break
            else:
                cp_idx-=hdu_len

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
outcat_name = args['outputDir'] + '%d_outcat.cat'%expid

command_list = [psfex_path, args['outputDir']+'*.fits', "-c", psfex_config, "-OUTCAT_NAME",outcat_name ]

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
        atmpsf_loaded = pex.get_rec(yimage, ximage)
        atm_shape = atmpsf_loaded.shape[0] #assumed to be square
        if atm_shape < atmpsf.shape[0]:
           pad_amount = int((atmpsf.shape[0]-atmpsf_loaded.shape[0])/2)
           pad_amount_upper = pad_amount + atmpsf_loaded.shape[0]

           atmpsf[pad_amount:pad_amount_upper,pad_amount:pad_amount_upper] = atmpsf_loaded
        elif atm_shape > atmpsf.shape[0]:
            # now we have to cut psf for... reasons
            # TODO: I am 95% certain we don't care if the psf is centered, but let us worry anyways
            center = int(atm_shape / 2)
            lower = center - int(atmpsf.shape[0] / 2)
            upper = lower + atmpsf.shape[0]
            atmpsf = atmpsf_loaded[lower:upper, lower:upper]

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

np.save(args['outputDir']+'%s_stars.npy'%expid, np.array(stars))
np.save(args['outputDir']+'%s_opt.npy'%expid, optpsf_stamps)
np.save(args['outputDir']+'%s_atm.npy'%expid, atmpsf_list)
np.save(args['outputDir']+'%d_stars_minus_opt.npy'%expid, resid_list)

np.save(args['outputDir'] + '%s_bad_star_idxs.npy', np.array(sorted(list(bad_stars))) )

print 'Done'

# from matplotlib import pyplot as plt
# for star in stars:
#     im = plt.imshow(star, cmap = plt.get_cmap('afmhot'), interpolation = 'none')
#     plt.colorbar(im)
#     plt.savefig(args['outputDir']+'%s_star.png'%expid)

#Chris wrote all this below; it's an elaborate save procedure
#If it sucks blame him

sample_num = 0
kils = True
if kils:
    # these give the deconvolved stars
    #TODO change this to the passed in output?
    out_base = '/nfs/slac/g/ki/ki18/des/swmclau2/DeconvOutput/'
    #out_base = '/nfs/slac/g/ki/ki18/des/cpd/DeconvOutput/'
    deconv_dir = out_base + '{0:08d}'.format(expid)
    # not sure what stars these really are? the combined psfex + deconv?
    deconvmodel_loc = out_base + '{0:08d}/{0}_stars.npy'.format(expid)
    deconvopt_loc = out_base + '{0:08d}/{0}_opt.npy'.format(expid)
    deconvatm_loc = out_base + '{0:08d}/{0}_atm.npy'.format(expid)
    deconvstarsminusopt_loc = out_base + '{0:08d}/{0}_stars_minus_opt.npy'.format(expid)

    deconvopt_immediate_loc = '/nfs/slac/g/ki/ki18/des/swmclau2/DeconvOutput/{0:08d}/{0}_opt_test.npy'.format(expid)

    jamierod_results_path = '/nfs/slac/g/ki/ki18/des/cpd/jamierod_results.csv'
    mesh_directory = '/nfs/slac/g/ki/ki22/roodman/ComboMeshesv20'
    # directory containing the input data files
    base_directory = '/nfs/slac/g/ki/ki18/des/cpd/psfex_catalogs/SVA1_FINALCUT/psfcat/'


def evaluate_stamps_and_combine_with_data(stamps, data):
    eval_data = WF.evaluate_psf(stamps)
    eval_data.index = data.index
    combined_df = eval_data.combine_first(data)
    return combined_df

jamierod_results = pd.read_csv(jamierod_results_path)
jamierod_results = jamierod_results.set_index('expid')

# set up objects. make sure I get the right mesh
digestor = Digestor()
PSF_Evaluator = Moment_Evaluator()
mesh_name = 'Science-20121120s1-v20i2_All'
PSF_Interpolator = Mesh_Interpolator(mesh_name=mesh_name, directory=mesh_directory)

# This will be our main wavefront
WF = DECAM_Model_Wavefront(PSF_Interpolator=PSF_Interpolator)

# load up data
expid_path = '/{0:08d}/{1:08d}'.format(expid - expid % 1000, expid)
data_directory = base_directory + expid_path
files = sorted(glob(data_directory + '/*{0}'.format('_selpsfcat.fits')))

data_df = digestor.digest_fits(files[0], do_exclude=False)
meta_hdulist = [fits.open(files[0])] #list of HDULists #META

for file in files[1:]:
    tmpData = digestor.digest_fits(file,do_exclude=False )
    data_df = data_df.append(tmpData)
    meta_hdulist.append(fits.open(file))

if sample_num > 0:
    full_size = len(data_df)
    sample_indx = np.random.choice(full_size, sample_num)
    data_df = data_df.iloc[sample_indx]
    print(full_size, sample_num)

# make the psfex models for both portions
psf_files = sorted(glob(data_directory + '/*{0}'.format('psfcat_validation_subtracted.psf')))

psfex_list = []
psfex_flipped_list = []

#TODO Inconcisitent definition of stars!
stars = []
#TODO Check that files are in the same order as the hdulist
for psfex_file, hdulist in izip(psf_files, meta_hdulist):
    pex_orig = PSFEx(psfex_file)
    for yimage, ximage in izip(hdulist[2].data['YWIN_IMAGE'], hdulist[2].data['XWIN_IMAGE']):
        atmpsf_tmp = np.zeros((32,32))
        #psfex has a tendency to return images of weird and varying sizes
        #This scheme ensures that they will all be the same 32x32 by zero padding
        #assumes the images are square and smaller than 32x32
        #Proof god is real and hates observational astronomers.
        atmpsf_small = pex_orig.get_rec(yimage, ximage)
        atm_shape = atmpsf_small.shape[0] #assumed to be square
        if atm_shape < atmpsf_tmp.shape[0]:
            pad_amount = int((atmpsf_tmp.shape[0]-atmpsf_small.shape[0])/2)
            pad_amount_upper = pad_amount + atmpsf_small.shape[0]

            atmpsf_tmp[pad_amount:pad_amount_upper,pad_amount:pad_amount_upper] = atmpsf_small
        elif atm_shape > atmpsf_tmp.shape[0]:
            # now we have to cut psf for... reasons
            # TODO: I am 95% certain we don't care if the psf is centered, but let us worry anyways
            center = int(atm_shape / 2)
            lower = center - int(atmpsf_tmp.shape[0] / 2)
            upper = lower + atmpsf_tmp.shape[0]
            atmpsf_tmp = atmpsf_small[lower:upper, lower:upper]
        psfex_list.append(atmpsf_tmp)

        atmpsf_tmp = np.zeros((32,32))
        #psfex has a tendency to return images of weird and varying sizes
        #This scheme ensures that they will all be the same 32x32 by zero padding
        #assumes the images are square and smaller than 32x32
        #Proof god is real and hates observational astronomers.
        atmpsf_small = pex_orig.get_rec(ximage, yimage)
        atm_shape = atmpsf_small.shape[0] #assumed to be square
        if atm_shape < atmpsf_tmp.shape[0]:
            pad_amount = int((atmpsf_tmp.shape[0]-atmpsf_small.shape[0])/2)
            pad_amount_upper = pad_amount + atmpsf_small.shape[0]

            atmpsf_tmp[pad_amount:pad_amount_upper,pad_amount:pad_amount_upper] = atmpsf_small
        elif atm_shape > atmpsf_tmp.shape[0]:
            # now we have to cut psf for... reasons
            # TODO: I am 95% certain we don't care if the psf is centered, but let us worry anyways
            center = int(atm_shape / 2)
            lower = center - int(atmpsf_tmp.shape[0] / 2)
            upper = lower + atmpsf_tmp.shape[0]
            atmpsf_tmp = atmpsf_small[lower:upper, lower:upper]
        psfex_flipped_list.append(atmpsf_tmp)

    # try to cut out stars
#     stars.append(hdulist[2].data['VIGNET'][:, 15:47, 15:47])
    stars.append(hdulist[2].data['VIGNET'][:, 16:48, 16:48])

psfexpsf = np.array(psfex_list)
psfexflippsf = np.array(psfex_flipped_list)
stars = np.array(stars)
stars = np.vstack(stars).astype(np.float64)

if sample_num > 0:
    psfexpsf = psfexpsf[sample_indx]
    psfexflippsf = psfexflippsf[sample_indx]
    stars = stars[sample_indx]

stars_df = evaluate_stamps_and_combine_with_data(stars, data_df)
psfexpsf_df = evaluate_stamps_and_combine_with_data(psfexpsf, data_df)
psfexflippsf_df = evaluate_stamps_and_combine_with_data(psfexflippsf, data_df)

atmpsf = np.load(deconvatm_loc)
optpsf = np.load(deconvopt_loc)
# set the shape to be right
starminusopt = np.load(deconvstarsminusopt_loc)[:, 15:47, 15:47]
model = np.load(deconvmodel_loc)

if sample_num > 0:
    atmpsf = atmpsf[sample_indx]
    optpsf = optpsf[sample_indx]
    starminusopt = starminusopt[sample_indx]
    model = model[sample_indx]

atmpsf_df = evaluate_stamps_and_combine_with_data(atmpsf, data_df)
optpsf_df = evaluate_stamps_and_combine_with_data(optpsf, data_df)
starminusopt_df = evaluate_stamps_and_combine_with_data(starminusopt, data_df)
model_df = evaluate_stamps_and_combine_with_data(model, data_df)

combinekeys = ['e0', 'e1', 'e2', 'E1norm', 'E2norm', 'delta1', 'delta2', 'zeta1', 'zeta2']
# make a big df with all the above columns combined
df = stars_df.copy()
names = ['model', 'psfex', 'starminusopt', 'opt', 'atm', 'psfex_flip']
diff_names = ['model', 'psfex']
df_list = [model_df, psfexpsf_df, starminusopt_df, optpsf_df, atmpsf_df, psfexflippsf_df]

# names += ['opt_load']
# df_list += [optpsf_load_df]

# names += ['atm_make']
# df_list += [atmpsf_make_df]

for key in combinekeys:
    # add the other medsub
    if key == 'E1norm':
        df[key] = df['e1'] / df['e0']
    elif key == 'E2norm':
        df[key] = df['e2'] / df['e0']
    df['{0}_medsub'.format(key)] = df[key] - df[key].median()
    for name, psf in zip(names, df_list):
        if key == 'E1norm':
            psf[key] = psf['e1'] / psf['e0']
        elif key == 'E2norm':
            psf[key] = psf['e2'] / psf['e0']
        df['{0}_{1}'.format(name, key)] = psf[key]
        # add medsub
        df['{0}_{1}_medsub'.format(name, key)] = df['{0}_{1}'.format(name, key)] - df['{0}_{1}'.format(name, key)].median()
        df['{0}_{1}_diff'.format(name, key)] = df['{0}_{1}'.format(name, key)] - df[key]
        df['{0}_{1}_medsub_diff'.format(name, key)] = df['{0}_{1}_medsub'.format(name, key)] - df['{0}_medsub'.format(key)]

np.save(out_base + '{0:08d}/{0}_psfexalone.npy'.format(expid), psfexpsf)
np.save(out_base + '{0:08d}/{0}_data.npy'.format(expid), stars)

df.to_hdf(out_base + '{0:08d}/results.h5'.format(expid),
          key='table_{0:08d}'.format(expid),
          mode='a', format='table', append=False)

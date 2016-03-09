#!/nfs/slac/g/ki/ki06/roodman/Software/anaconda/bin/python
#@Author Sean McLaughlin
desc ='''
Arguments:
    - expid: the exposure ID of the exposure to run against
    - output_dir: the directory in which to create a subdirectory for temporary files and final outputs.
Requirements:
    -WavefrontPSF
    -numpy, pandas, astropy or pyfits
    -a psfex installation and python binding

This module is the main module for my project for the winter quarter of 2016 in Aaron Roodman's group.
This is built on top of previous work conducted by Aaron and his graduate student Chris Davis. They have
developed WavefrontPSF, which estimates the optical contribution of the PSF with a relatively simple model.
It was found that the optical portion was not as signifcant a fraction of the total PSF as hoped,
so some sort of afterburner is going to need to be added. This module deconvolves the optical portion of the 
psf from the observed stars. After the optical portion has been deconvolved
from the stars (using Richardson-Lucy deconvolution), the remainder is be treated as the "atmospheric"
portion of the psf. This module load in preprocessed observed stars, run WavefrontPSF on them, deconvolve
the optical PSF, then run PSFEX (a packaged PSF modeler) on the residual.
'''

#Do imports here instead of before the argparse because users will be able to acces
#the help script without these packages instalelled, and more quickly
#TODO now that argparse is in a __main__, not sure where to put these that'd be easiest.
from WavefrontPSF.psf_interpolator import Mesh_Interpolator
from WavefrontPSF.digestor import Digestor
from WavefrontPSF.donutengine import DECAM_Model_Wavefront
try:
    from astropy.io import fits #TODO change to fitsio?
except ImportError:
    import pyfits as fits #should have the same API
import numpy as np
from psfex import PSFEx #Move into one function that uses it?
from glob import glob
from itertools import izip
from subprocess import call
from optical_model import get_optical_psf
from lucy import deconvolve, convolve

#Value with which to mask failed deconvolutions
MASK_VAL = -9999

def get_hdu_idxs(meta_hdulist):
    """
    helper function to calculate the start/end idxs of each HDUlist in the 1D flattened case
    :param meta_hdulist:
    :return: hdu_idxs
    """
    hdu_lengths = np.zeros((62,))
    for ccd, hdulist in enumerate(meta_hdulist):
        hdu_lengths[ccd] = hdulist[2].data.shape[0]

    hdu_idxs = hdu_lengths.cumsum()
    np.insert(hdu_idxs, 0, 0)#insert 0 as first elem
    return hdu_idxs

def get_ccd_idx(idx_1d, hdu_idxs):
    """
    return the ccd and ccd idx given a 1d idx of a star.
    :param idx_1d: the idx in the flattened case
    :param hdu_idxs: output of getHDULength, the start/end idxs of each hdu in the 1D flattened case
    :return: ccd_num, ccd_idx
    """
    last_idx = 0
    for ccd_num, hdu_idx in enumerate( hdu_idxs) :
        if idx_1d > hdu_idx:
            last_idx = hdu_idx
            continue
        break
    return ccd_num, idx_1d-last_idx

def get_vignettes(NObj, meta_hdulist,hdu_idxs = None):
    """
    Get the vignettes from the hdulists as a numpy array datacube
    :param NObj: number of stars
    :param meta_hdulist: list of hdulists with the snippets in ['VIGNET']
    :param hdu_idxs: (Optional) Defines the idxs where the hdus start and end in the 1D flattened case
    :return: vignettes (nObj, 32,32) datacube of star vignettes
    """

    if hdu_idxs is None:
        hdu_idxs = get_hdu_idxs(meta_hdulist)
    vignettes = np.zeros((NObj, 32,32))

    for ccd_num, hdulist in enumerate(meta_hdulist):
        sliced_vig  = hdulist[2].data['VIGNET'][:, 15:47, 15:47] #slice to same size as stamps
        #TODO more clever interpolations?
        sliced_vig[sliced_vig<-1000] = 0 #set really negative values to 0; it's a mask
        sliced_vig = sliced_vig/sliced_vig.sum((1,2))[:, None, None] #normalize
        vignettes[hdu_idxs[ccd_num]:hdu_idxs[ccd_num+1]] = sliced_vig

    return vignettes

def deconv_optpsf(NObj, optpsf_arr, vignettes ):
    """
    deconvolves the optical model from the given vignettes. Returns the residuals and a boolean array of which
    deconvolutions were successful according the the LR deconv algorithm
    :param NObj: number of stars
    :param optpsf_arr: (nObj, 32,32) datacube of the optical model of the psf
    :param vignettes: (nObj, 32,32) datacube of the star vignettes
    :return: resid_arr and deconv_successful, (nObj, 63,63) array of residuals and deconv_successful, a boolean array if a deconvolution was successful
    """
    resid_arr = np.zeros((NObj, 63,63))
    deconv_successful = np.ones((NObj,), dtype=bool)
    for idx, (optpsf, vignette) in enumerate(izip(optpsf_arr, vignettes)):
        #background is all pixels below 1 std. Could vary but won't make much difference.
        background = vignette[vignette< vignette.mean()+vignette.std()]
        try:
            #this makes initial guess be all ones; could guess vignette, result isn't all that different
            resid_small = deconvolve(optpsf,vignette,mask=None,mu0=background.mean(),convergence=1e-2,niterations=50, extra= False)

            resid_arr[idx, 15:47, 15:47] = resid_small
        except RuntimeError: #Some will fail
            resid_arr[idx]+= MASK_VAL #forcing a mask
            deconv_successful[idx] = False
            #If i wanted to store bad stars by ccd and ccd_idx, I'd call get_ccd_idx here

    return resid_arr, deconv_successful

def write_resid(output_dir, meta_hdulist, resid_arr,hdu_idxs = None):
    """
    Take the calculated residuals, insert them into the existing hdulists, and write them to file.
    Returns the filenames written to.
    :param output_dir: the directory to store the output and temp files
    :param meta_hdulist: list of hdulists to insert residuals into
    :param resid_arr: residuals from deconvolution.
    :param hdu_idxs: (Optional) Defines the idxs where the hdus start and end in the 1D flattened case
    :return: fnames, the filenames the hdulists were written to
    """

    if hdu_idxs is None:
        hdu_idxs = get_hdu_idxs(meta_hdulist)

    fnames = []
    for ccd_num, hdulist in enumerate(meta_hdulist):
        hdulist[2].data['VIGNET'] = resid_arr[hdu_idxs[ccd_num]:hdu_idxs[ccd_num+1]]

        #Make new filename from old one.
        original_fname = hdulist.filename().split('/')[-1]#just get the filename, not the path
        original_fname_split = original_fname.split('_')
        original_fname_split[-1] = 'seldeconv.fits'
        fname = output_dir+'_'.join(original_fname_split)
        hdulist.writeto(fname, clobber = True)

        fnames.append(fname)

    return fnames

#TODO include as option in write resid, or separate function?
#Can't decide between balance of copied code and different purposes.
def write_resid_new_file(output_dir, meta_hdulist, resid_arr, deconv_successful, hdu_idxs = None):
    """
    Similar to write_resid, but removes stars where deconvolution failed. Creates new HDUlists to do this.
    NOTE currently not compatible with PSFEx
    param meta_hdulist: list of hdulists to insert residuals into
    :param output_dir: the directory to store the output and temp files
    :param resid_arr: residuals from deconvolution.
    :param deconv_successful: a boolean array defining which deconvolutions were successful
    :param hdu_idxs: (Optional) Defines the idxs where the hdus start and end in the 1D flattened case
    :return: fnames, the filenames the hdulists were written to. Also new_meta_hdulist, a list of the new hdulists
    """

    if hdu_idxs is None:
        hdu_idxs = get_hdu_idxs(meta_hdulist)

    new_meta_hdulist = []
    fnames = []
    for ccd_num, hdulist in enumerate(meta_hdulist):
        hdulist[2].data['VIGNET'] = resid_arr[hdu_idxs[ccd_num]:hdu_idxs[ccd_num+1]]


        #make a new hdulist, removing the stars we've masked.
        #NOTE currently not working with PSFEx
        primary_table = hdulist[0].copy() #will shallow copy work?
        imhead = hdulist[1].copy()
        objects = fits.BinTableHDU(data = hdulist[2].data[deconv_successful[hdu_idxs[ccd_num]:hdu_idxs[ccd_num+1]]], header = hdulist[2].header,\
                                   name = hdulist[2].name)
        #Not sure these do anything, but trying
        objects.header.set('EXTNAME', 'LDAC_OBJECTS', 'a name')
        objects.header.set('NAXIS2', str(deconv_successful[hdu_idxs[ccd_num]:hdu_idxs[ccd_num+1]].sum()), 'Trying this...')

        new_hdulist = fits.HDUList(hdus = [primary_table, imhead, objects])
        new_meta_hdulist.append(new_hdulist)

        #Make new filename from old one.
        original_fname = hdulist.filename().split('/')[-1]#just get the filename, not the path
        original_fname_split = original_fname.split('_')
        original_fname_split[-1] = 'seldeconv.fits'
        fname = output_dir+'_'.join(original_fname_split)
        new_hdulist.writeto(fname, clobber = True)
        fnames.append(fname)

    return fnames, new_meta_hdulist

def call_psfex(expid,output_dir, fnames = None):
    """
    calls psfex on ki-ls on the files. returns True if the call executed without error.
    :param expid: The id of the exposure being studied
    :param output_dir: the directory to store the output and temp files
    :param fnames: (Optional) filenames to call psfex on. If omitted, will be called on all fits files in output_dir.
    :return: psfex_success, True if the call didn't return an error
    """
    psfex_path = '/nfs/slac/g/ki/ki22/roodman/EUPS_DESDM/eups/packages/Linux64/psfex/3.17.3+0/bin/psfex'
    psfex_config = '/afs/slac.stanford.edu/u/ec/roodman/Astrophysics/PSF/desdm-plus.psfex'
    outcat_name = output_dir + '%d_outcat.cat'%expid

    if fnames is None:
        file_string = output_dir+'*.fits'
    else:
        file_string = " ".join(fnames)

    command_list = [psfex_path,file_string, "-c", psfex_config, "-OUTCAT_NAME",outcat_name ]

    #If shell != True, the wildcard won't work
    psfex_return= call(' '.join(command_list), shell = True)
    psfex_success = True if psfex_return==0 else False
    print 'PSFEx Call Successful: %s'%psfex_success

    return psfex_success

def load_psfex(psf_files, NObj, meta_hdulist):
    """
    Loads output files from PSFEx for given stars
    :param psf_files: the output files from psfex
    :param NObj: the number of objects that will be loaded
    :param meta_hdulist: the list of HDULists
    :return: psfex_arr: a (NObj, 32,32) datacube
    """
    psfex_arr = np.zeros(NObj, 32,32)
    for idx, (file, hdulist) in enumerate(izip(psf_files, meta_hdulist)):
        pex = PSFEx(file)
        for yimage, ximage in izip(hdulist[2].data['Y_IMAGE'], hdulist[2].data['X_IMAGE']):
            #psfex has a tendency to return images of weird and varying sizes
            #This scheme ensures that they will all be the same 32x32 by zero padding
            #assumes the images are square and smaller than 32x32
            #Proof god is real and hates observational astronomers.
            psfex_loaded = pex.get_rec(yimage, ximage)
            atm_shape = psfex_loaded.shape[0] #assumed to be square
            if atm_shape < psfex_arr.shape[1]:
               pad_amount = int((psfex_arr.shape[1]-psfex_loaded.shape[0])/2)
               pad_amount_upper = pad_amount + psfex_loaded.shape[0]

               psfex_arr[idx, pad_amount:pad_amount_upper,pad_amount:pad_amount_upper] = psfex_loaded
            elif atm_shape > psfex_arr.shape[1]:
                # now we have to cut psf for... reasons
                # TODO: I am 95% certain we don't care if the psf is centered, but let us worry anyways
                center = int(atm_shape / 2)
                lower = center - int(psfex_arr.shape[1] / 2)
                upper = lower + psfex_arr.shape[1]
                psfex_arr[idx] = psfex_loaded[lower:upper, lower:upper]

    return psfex_arr

def make_stars(NObj, optpsf_arr, atmpsf_arr, deconv_successful = None):
    """
    convolve the optical and psf models to make a full model for the psf of the stars
    :param NObj: number of stars
    :param optpsf_arr: array of optical psf models
    :param atmpsf_arr: array of atmospheric psf models
    :param deconv_successful: boolean array denoting if the deconvolution converged. If passed in, will be used to
    slice bad indexs from optpsf_arr
    :return: stars, (nObj, 32,32) array of star psf estimates.
    """
    stars = np.array(NObj, 32,32)

    #Note that atmpsf_arr will already have the bad stars removed if the user is using that scheme.
    if deconv_successful is not None:
        #TODO make sure this isn't modifying the outer object
        optpsf_arr = optpsf_arr[deconv_successful] #slice off failed ones.

    for idx, (optpsf, atmpsf) in enumerate(izip(optpsf_arr, atmpsf_arr)):

        try:
            stars[idx] = convolve(optpsf, atmpsf)
        except ValueError:

            print 'Convolve failed on object (1D Index) #%d'%(idx)
            raise

    return stars

def evaluate_stamps_and_combine_with_data(WF, stamps, data):
    eval_data = WF.evaluate_psf(stamps)
    eval_data.index = data.index
    combined_df = eval_data.combine_first(data)
    return combined_df

def make_wavefront(expid, output_dir, optpsf = None, atmpsf = None, starminusopt = None, model = None):
    """
    Make a wavefront, useful for diagnostic plots
    :param expid: the id of the exposure being studied
    :param output_dir: the directory to store the output and temp files
    :param optpsf: (Optional) the optical psf in a datacube
    :param atmpsf: (Optional) the atmopsheric psf in a datacube
    :param starminusopt: (Optional) the residual when the optpsf is deconvolved
    :param model: (model) the convolution of optpsf and atmpsf
    :return: None
    """
    # these give the deconvolved stars
    #Wish I knew how to loop this
    if optpsf is None:
        deconvopt_loc = output_dir + '{0:08d}/{0}_opt.npy'.format(expid)
        optpsf = np.load(deconvopt_loc)

    if atmpsf is None:
        deconvatm_loc = output_dir + '{0:08d}/{0}_atm.npy'.format(expid)
        atmpsf = np.load(deconvatm_loc)

    if starminusopt is None:
        deconvstarsminusopt_loc = output_dir + '{0:08d}/{0}_stars_minus_opt.npy'.format(expid)
        #set the shape to be right
        starminusopt = np.load(deconvstarsminusopt_loc)[:, 15:47, 15:47]

    if model is None:
        deconvmodel_loc = output_dir + '{0:08d}/{0}_stars.npy'.format(expid)
        model = np.load(deconvmodel_loc)

    mesh_directory = '/nfs/slac/g/ki/ki22/roodman/ComboMeshesv20'
    # directory containing the input data files
    base_directory = '/nfs/slac/g/ki/ki18/des/cpd/psfex_catalogs/SVA1_FINALCUT/psfcat/'

    # set up objects. make sure I get the right mesh
    digestor = Digestor()
    mesh_name = 'Science-20121120s1-v20i2_All'
    PSF_Interpolator = Mesh_Interpolator(mesh_name=mesh_name, directory=mesh_directory)

    # This will be our main wavefront
    WF = DECAM_Model_Wavefront(PSF_Interpolator=PSF_Interpolator)

    # load up data
    expid_path = '/{0:08d}/{1:08d}'.format(expid - expid % 1000, expid)
    data_directory = base_directory + expid_path
    files = sorted(glob(data_directory + '/*{0}'.format('_selpsfcat.fits')))

    data_df = digestor.digest_fits(files[0], do_exclude=False)
    #Can't use the new one above, because we're calling on different data.
    meta_hdulist = [fits.open(files[0])]

    for file in files[1:]:
        tmpData = digestor.digest_fits(file,do_exclude=False )
        data_df = data_df.append(tmpData)
        meta_hdulist.append(fits.open(file))

    hdu_idxs = get_hdu_idxs(meta_hdulist)
    NObj = hdu_idxs[-1]

    # make the psfex models for both portions
    psf_files = sorted(glob(data_directory + '/*{0}'.format('psfcat_validation_subtracted.psf')))

    psfexpsf = load_psfex(psf_files, NObj, meta_hdulist)

    stars = get_vignettes(NObj, meta_hdulist, hdu_idxs)

    stars_df = evaluate_stamps_and_combine_with_data(WF, stars, data_df)
    psfexpsf_df = evaluate_stamps_and_combine_with_data(WF, psfexpsf, data_df)

    atmpsf_df = evaluate_stamps_and_combine_with_data(WF, atmpsf, data_df)
    optpsf_df = evaluate_stamps_and_combine_with_data(WF, optpsf, data_df)
    starminusopt_df = evaluate_stamps_and_combine_with_data(WF, starminusopt, data_df)
    model_df = evaluate_stamps_and_combine_with_data(WF, model, data_df)

    combinekeys = ['e0', 'e1', 'e2', 'E1norm', 'E2norm', 'delta1', 'delta2', 'zeta1', 'zeta2']
    # make a big df with all the above columns combined
    df = stars_df.copy()
    names = ['model', 'psfex', 'starminusopt', 'opt', 'atm', 'psfex_flip']
    df_list = [model_df, psfexpsf_df, starminusopt_df, optpsf_df, atmpsf_df]

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

    np.save(output_dir + '{0:08d}/{0}_psfexalone.npy'.format(expid), psfexpsf)
    np.save(output_dir + '{0:08d}/{0}_data.npy'.format(expid), stars)

    df.to_hdf(output_dir + '{0:08d}/results.h5'.format(expid),
              key='table_{0:08d}'.format(expid),
              mode='a', format='table', append=False)

if __name__ == '__main__':

    #TODO verbose tag
    #TODO delete temp files, use temp files, other options?
    from argparse import ArgumentParser
    parser = ArgumentParser(description = desc)

    parser.add_argument('expid', metavar = 'expid', type = int, help =\
                        'ID of the exposure to analyze')
    #May want to rename to tmp
    parser.add_argument('output_dir', metavar = 'output_dir', type = str, help =\
                        'Directory to store outputs.')

    args = vars(parser.parse_args())

    expid = args['expid']
    output_dir = args['output_dir']

    #Ensure provided dir exists
    from os import path, mkdir
    if not path.isdir(output_dir):
        raise IOError("The directory %s does not exist."%output_dir)

    if output_dir[-1]  != '/':
        output_dir+='/'

    #Make new dir to store files from this run
    if not path.isdir(output_dir+'00%d/'%expid):
        try:
            mkdir(output_dir+'00%d/'%expid)
        except OSError:
            print 'Failed making directory; using original output directory.'
        else:
            output_dir+='00%d/'%expid
    else:
        output_dir+='00%d/'%expid

    print 'Starting.'

    #get optical PSF
    optpsf_stamps, meta_hdulist = get_optical_psf(expid)

    NObj = optpsf_stamps.shape[0]#I'm undecided about hte use of this carrier
    #For the most part the number of stars is contained implicitly in the other objects I pass around
    #still, being explicit costs next to nothing and is clear to the user.

    hdu_idxs = get_hdu_idxs(meta_hdulist)

    #np.save(output_dir+'%s_opt_test.npy'%expid, optpsf_stamps)

    print 'Opts Calculated.'

    #extract star vignettes from the hdulists
    vignettes = get_vignettes(NObj, meta_hdulist, hdu_idxs)

    #deconvolve the optical model from the observed stars
    resid_arr, deconv_successful = deconv_optpsf(NObj, optpsf_stamps, vignettes)

    print 'Deconv done.'

    #now, insert the atmospheric portion back into the hdulists, and write them to disk
    #PSFEx needs the information in those lists to run correctly.

    resid_fnames = write_resid(output_dir, meta_hdulist, resid_arr, hdu_idxs)
    #resid_fnames, new_meta_hdulist = write_resid_new_file(meta_hdulist, resid_arr, deconv_successful, hdu_idxs)
    #NObj = deconv_successful.sum() #if making the new HDUlist, the number of objects has changed. Make sure to account.

    print 'Copy and write done.'

    psfex_success = call_psfex(expid, output_dir, resid_fnames)

    #no use continuing if the psfex call failed.
    #TODO if I write a verbose flag this should print if that is turned off. Cuz the user always needs to know why it exited.
    if not psfex_success:
        from sys import exit
        exit(1)

    psf_files = sorted(glob(output_dir+'*.psf'))
    atmpsf_arr = load_psfex(psf_files, NObj, meta_hdulist)
    #atmpsf_arr = load_atmpsf(psf_files, NObj, new_meta_hdulist)

    stars = make_stars(NObj, optpsf_stamps, atmpsf_arr)
    #stars = make_stars(NObj, optpsf_stamps, atmpsf_arr, deconv_successful)

    #TODO what to save?
    #TODO saved deconv_succsseful sliced arrays?
    #Note that these won't al have the same dimensions without a slice by deconv_successful
    np.save(output_dir+'%s_stars.npy'%expid, stars)
    np.save(output_dir+'%s_opt.npy'%expid, optpsf_stamps)
    np.save(output_dir+'%s_atm.npy'%expid, atmpsf_arr)
    np.save(output_dir+'%d_stars_minus_opt.npy'%expid, resid_arr)

    np.save(output_dir+'%s_deconv_successful.npy', deconv_successful)

    print 'Done'

    optpsf_stamps = optpsf_stamps[deconv_successful]
    resid_arr = resid_arr[deconv_successful]

    #TODO optional make_wavefront call

#!/usr/bin/env python
#NOTE I took this function from ki-ls, and am modifying it to return the models I need.

"""
Two sets of things:
    1. Plots for aaron
    2. Plots for poster

New dates to ingest:
20150731, 20150828
note: take out z11 in interpolator
"""
from __future__ import print_function
import pandas as pd
import numpy as np
from os import path, makedirs
from glob import glob
from astropy.io import fits

KILS = True
if KILS:
    jamierod_results = pd.read_csv('/nfs/slac/g/ki/ki18/des/cpd/jamierod_results.csv')
    jamierod_results = jamierod_results.set_index('expid')

    # output directory
    out_dir = '/nfs/slac/g/ki/ki18/des/cpd/jamierod_results'
    if not path.exists(out_dir):
        makedirs(out_dir)
    log_dir = out_dir + '/logs'
    if not path.exists(log_dir):
        makedirs(log_dir)
    plot_dir = out_dir + '/plots'
    if not path.exists(plot_dir):
        makedirs(plot_dir)
    pkl_dir = out_dir + '/pickles'
    if not path.exists(pkl_dir):
        makedirs(pkl_dir)

    # directory containing the input data files
    base_directory = '/nfs/slac/g/ki/ki18/des/cpd/psfex_catalogs/SVA1_FINALCUT/psfcat/'
    new_mesh_directory = '/u/ec/roodman/Astrophysics/Donuts/Meshesv20'
    mesh_directory = '/nfs/slac/g/ki/ki22/roodman/ComboMeshesv20'
    # bsub runs
    code_path = '/nfs/slac/g/ki/ki18/cpd/Projects/WavefrontPSF/code/WavefrontPSF_analysis/Jamie_Pipeline/plot_jamie_results.py'
    req_time = 20 # min

else:
    # output directory
    out_dir = '/Users/cpd/Projects/WavefrontPSF/meshes/tests'
    if not path.exists(out_dir):
        makedirs(out_dir)
    log_dir = out_dir + '/logs'
    if not path.exists(log_dir):
        makedirs(log_dir)
    plot_dir = out_dir + '/plots'
    if not path.exists(plot_dir):
        makedirs(plot_dir)
    pkl_dir = out_dir + '/pickles'
    if not path.exists(pkl_dir):
        makedirs(pkl_dir)
    # my laptop
    new_mesh_directory = '/Users/cpd/Projects/WavefrontPSF/meshes/Science-20140212s2-v20i2'
    mesh_directory = '/Users/cpd/Projects/WavefrontPSF/meshes/Science-20140212s2-v20i2'



###############################################################################
# run
###############################################################################
def get_optical_psf(expid, aos=False):

    #TODO move up top?
    from WavefrontPSF.psf_interpolator import Mesh_Interpolator
    from WavefrontPSF.wavefront import Wavefront
    from WavefrontPSF.digestor import Digestor
    from WavefrontPSF.psf_evaluator import Moment_Evaluator
    from WavefrontPSF.donutengine import DECAM_Model_Wavefront



    medsubkeys = ['e0', 'e1', 'e2', 'E1norm', 'E2norm', 'delta1', 'delta2', 'zeta1', 'zeta2']

    rows = ['e0', 'e0_medsub',
            'e1', 'e1_medsub',
            'e2', 'e2_medsub',
            'E1norm', 'E1norm_medsub',
            'E2norm', 'E2norm_medsub',
            'delta1', 'delta1_medsub',
            'delta2', 'delta2_medsub',
            'zeta1', 'zeta1_medsub',
            'zeta2', 'zeta2_medsub']

    # set up objects. make sure I get the right mesh
    digestor = Digestor()
    PSF_Evaluator = Moment_Evaluator()
    mesh_name = 'Science-20121120s1-v20i2_All'
    PSF_Interpolator = Mesh_Interpolator(mesh_name=mesh_name, directory=mesh_directory)

    # This will be our main wavefront
    WF = DECAM_Model_Wavefront(PSF_Interpolator=PSF_Interpolator)
    # let's create a Wavefront object for the data
    WF_data = Wavefront(PSF_Interpolator=None, PSF_Evaluator=PSF_Evaluator)

    # premake coordinate list
    coords = []
    for num_bins in xrange(6):
        # create coordinates
        x = []
        y = []
        if num_bins >= 2:
            num_bins_make = num_bins + (num_bins-1)
        else:
            num_bins_make = num_bins
        for key in WF.decaminfo.infoDict.keys():
            if 'F' in key:
                continue
            xi, yi = WF.decaminfo.getBounds(key, num_bins_make)
            xi = np.array(xi)
            xi = 0.5 * (xi[1:] + xi[:-1])
            yi = np.array(yi)
            yi = 0.5 * (yi[1:] + yi[:-1])
            xi, yi = np.meshgrid(xi, yi)
            xi = xi.flatten()
            yi = yi.flatten()
            x += list(xi)
            y += list(yi)
        x = np.array(x)
        y = np.array(y)
        coords_i = pd.DataFrame({'x': x, 'y': y})
        coords.append(coords_i)


    # load up data
    expid_path = '{0:08d}/{1:08d}'.format(expid - expid % 1000, expid)
    data_directory = base_directory + expid_path
    files = glob(data_directory + '/*{0}'.format('_selpsfcat.fits'))

    # load up all the data from an exposure. Unfortunately, pandas is stupid and
    # can't handle the vignet format, so we don't load those up
    # note that you CAN load them up by passing "do_exclude=True", which then
    # returns a second variable containing the vignets and aperture fluxes and
    # errors

    #data has certain columns removed, needed for processing.
    #unfortunately I need full_data's vignettes and other info for later steps
    #TODO optimize this cuz this is clearly wasteful
    #I'm loading an HDUlist in 2 places, but overhauling the digestor to load it once would be a challenge
    data = digestor.digest_fits(files[0], do_exclude=False)
    metaHDUList = [fits.open(files[0])] #list of HDULists #META

    for file in files[1:]:
        tmpData = digestor.digest_fits(file,do_exclude=False )
        data = data.append(tmpData)
        metaHDUList.append(fits.open(file))

    fit_i = jamierod_results.loc[expid]

    #making rzero larger, among other things
    misalignment = {'z04d': fit_i['z04d'], 'z04x': fit_i['z04x'], 'z04y': fit_i['z04y'],
                    'z05d': fit_i['z05d'], 'z05x': fit_i['z05x'], 'z05y': fit_i['z05y'],
                    'z06d': fit_i['z06d'], 'z06x': fit_i['z06x'], 'z06y': fit_i['z06y'],
                    'z07d': fit_i['z07d'], 'z07x': fit_i['z07x'], 'z07y': fit_i['z07y'],
                    'z08d': fit_i['z08d'], 'z08x': fit_i['z08x'], 'z08y': fit_i['z08y'],
                    'z09d': fit_i['z09d'], 'z09x': fit_i['z09x'], 'z09y': fit_i['z09y'],
                    'z10d': fit_i['z10d'], 'z10x': fit_i['z10x'], 'z10y': fit_i['z10y'],
                    'rzero': 1.3*fit_i['rzero']}

    
    data['rzero'] = misalignment['rzero']
    optPSFStamps, model= WF.draw_psf(data, misalignment=misalignment)

    #optPSFStamps is a numpy data cube
    #full_data is data frame including vignettes of the stars
    return optPSFStamps, metaHDUList

if __name__ == '__main__':
#admittedly lazy test.
    from sys import argv
    expid = int(argv[1])
    psf, metaHDUList = get_optical_psf(expid)
    print(psf.shape)



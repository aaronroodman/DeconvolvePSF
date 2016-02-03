#!/usr/bin/env python
#@Author: Chris Davis
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

from WavefrontPSF.psf_interpolator import Mesh_Interpolator
from WavefrontPSF.digestor import Digestor
from WavefrontPSF.psf_evaluator import Moment_Evaluator
from WavefrontPSF.donutengine import DECAM_Model_Wavefront


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
def get_optical_psf_2(expid, aos=False):
    # set up objects. make sure I get the right mesh
    digestor = Digestor()
    PSF_Evaluator = Moment_Evaluator()
    mesh_name = 'Science-20121120s1-v20i2_All'
    PSF_Interpolator = Mesh_Interpolator(mesh_name=mesh_name, directory=mesh_directory)

    # This will be our main wavefront
    WF = DECAM_Model_Wavefront(PSF_Interpolator=PSF_Interpolator)

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
    files = sorted(glob(data_directory + '/*{0}'.format('_selpsfcat.fits')))

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

    misalignment = {'z04d': fit_i['z04d'], 'z04x': fit_i['z04x'], 'z04y': fit_i['z04y'],
                    'z05d': fit_i['z05d'], 'z05x': fit_i['z05x'], 'z05y': fit_i['z05y'],
                    'z06d': fit_i['z06d'], 'z06x': fit_i['z06x'], 'z06y': fit_i['z06y'],
                    'z07d': fit_i['z07d'], 'z07x': fit_i['z07x'], 'z07y': fit_i['z07y'],
                    'z08d': fit_i['z08d'], 'z08x': fit_i['z08x'], 'z08y': fit_i['z08y'],
                    'z09d': fit_i['z09d'], 'z09x': fit_i['z09x'], 'z09y': fit_i['z09y'],
                    'z10d': fit_i['z10d'], 'z10x': fit_i['z10x'], 'z10y': fit_i['z10y'],
                    'rzero': fit_i['rzero']}

    print(misalignment['z04d'])#focus
    misalignment['z04d'] = .5 
    print(misalignment['z05d'])
    misalignment['z05d'] = .5

    #print(misalignment['rzero'])
    #rzero needs to be adjusted to be smaller than the stars!
    x = .3/.14#4 
    misalignment['rzero'] = 1/(1/misalignment['rzero'] - x)
    #print(misalignment['rzero'])

    #print(.14*x )
    
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



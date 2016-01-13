#!/usr/bin/env python
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
def do_run(expid, aos=False):
    if aos:
        plot_dir = out_dir + '/plots_aos/{0:08d}'.format(expid)
    else:
        plot_dir = out_dir + '/plots/{0:08d}'.format(expid)
    if not path.exists(plot_dir):
        makedirs(plot_dir)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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

    # load up all the data from an exposure. Unfortunately, pandas is stupid and
    # can't handle the vignet format, so we don't load those up
    # note that you CAN load them up by passing "do_exclude=True", which then
    # returns a second variable containing the vignets and aperture fluxes and
    # errors
    model = digestor.digest_directory(
                data_directory,
                file_type='_selpsfcat.fits')
    # cut the old data appropriately
    model = model[(model['SNR_WIN'] > 90) &
                  (model['SNR_WIN'] < 400)]

    # create normalized moments
    model['E1norm'] = model['e1'] / model['e0']
    model['E2norm'] = model['e2'] / model['e0']

    # do med sub and add to WF_data
    for key in medsubkeys:
        model['{0}_medsub'.format(key)] = model[key] - np.median(model[key])
    WF_data.data = model

    # set the number of bins from total number of stars
    if len(model) < 200:
        num_bins = 0
        num_bins_mis = 0
        num_bins_whisker = 0
    elif len(model) < 1000:
        num_bins = 1
        # num_bins_mis = 0
        num_bins_mis = 1
        num_bins_whisker = 1
    elif len(model) < 10000:
        num_bins = 2
        # num_bins_mis = 1
        num_bins_mis = 2
        num_bins_whisker = 2
    else:
        num_bins = 3
        num_bins_mis = 2
        num_bins_whisker = 2


    # generate optics model from fit data
    fit_i = jamierod_results.loc[expid]
    # TODO: Add ALL fit_i params that were used?
    # TODO: get rzero?

    if aos:
        misalignment = {'z04d': fit_i['aos_z04d'],
                        'z05d': fit_i['aos_z05d'], 'z05x': fit_i['aos_z05x'], 'z05y': fit_i['aos_z05y'],
                        'z06d': fit_i['aos_z06d'], 'z06x': fit_i['aos_z06x'], 'z06y': fit_i['aos_z06y'],
                        'z07d': fit_i['aos_z07d'], 'z07x': fit_i['aos_z07x'], 'z07y': fit_i['aos_z07y'],
                        'z08d': fit_i['aos_z08d'], 'z08x': fit_i['aos_z08x'], 'z08y': fit_i['aos_z08y'],
                        'z09d': fit_i['aos_z09d'],
                        'z10d': fit_i['aos_z10d'],
                        'rzero': fit_i['aos_rzero']}
    else:
        misalignment = {'z04d': fit_i['z04d'], 'z04x': fit_i['z04x'], 'z04y': fit_i['z04y'],
                        'z05d': fit_i['z05d'], 'z05x': fit_i['z05x'], 'z05y': fit_i['z05y'],
                        'z06d': fit_i['z06d'], 'z06x': fit_i['z06x'], 'z06y': fit_i['z06y'],
                        'z07d': fit_i['z07d'], 'z07x': fit_i['z07x'], 'z07y': fit_i['z07y'],
                        'z08d': fit_i['z08d'], 'z08x': fit_i['z08x'], 'z08y': fit_i['z08y'],
                        'z09d': fit_i['z09d'], 'z09x': fit_i['z09x'], 'z09y': fit_i['z09y'],
                        'z10d': fit_i['z10d'], 'z10x': fit_i['z10x'], 'z10y': fit_i['z10y'],
                        'rzero': fit_i['rzero']}

    # create model fit from donuts
    WF.data = coords[num_bins].copy()
    WF.data['rzero'] = misalignment['rzero']
    WF.data = WF(WF.data, misalignment=misalignment)
    # add dc factors
    WF.data['e0'] += fit_i['e0']
    WF.data['e1'] += fit_i['e1']
    WF.data['e2'] += fit_i['e2']
    WF.data['delta1'] += fit_i['delta1']
    WF.data['delta2'] += fit_i['delta2']
    WF.data['zeta1'] += fit_i['zeta1']
    WF.data['zeta2'] += fit_i['zeta2']

    # create normalized moments
    WF.data['E1norm'] = WF.data['e1'] / WF.data['e0']
    WF.data['E2norm'] = WF.data['e2'] / WF.data['e0']

    # update WF medsubs appropriately
    for key in medsubkeys:
        WF.data['{0}_medsub'.format(key)] = WF.data[key] - np.median(WF.data[key])

    # add a couple diagnostic things
    WF.data['num_bins'] = num_bins
    WF.data['expid'] = expid

    # put in data into field
    WF.reduce(num_bins=num_bins)
    # create another reduced field for setting the color levels
    field_model, _, _ = WF.reduce_data_to_field(
        WF.data, xkey='x', ykey='y', reducer=np.median,
        num_bins=num_bins_mis)

    # update WF_data fields
    WF_data.reduce(num_bins=num_bins)
    # create another reduced field for setting the color levels
    field_data, _, _ = WF_data.reduce_data_to_field(
        WF_data.data, xkey='x', ykey='y', reducer=np.median,
        num_bins=num_bins_mis)

    # put in residual of data minus model for field
    for row_i, row in enumerate(rows):
        WF.field[row + '_data'] = WF_data.field[row]
        WF.field[row + '_residual'] = WF_data.field[row] - WF.field[row]
        field_model[row + '_residual'] = field_data[row] - field_model[row]

    # create plots
    for row in rows:
        ncols = 3
        nrows = 1
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows))
        fig.suptitle('Expid: {0}, {1}'.format(expid, row))
        vmin = np.nanmin((field_model[row].min(),
                          field_data[row].min()))
        vmax = np.nanmax((field_data[row].max(),
                          field_model[row].max()))
        vmindiff = field_model[row + '_residual'].min()
        vmaxdiff = field_model[row + '_residual'].max()
        ax = axs[0]
        ax.set_title('Data')
        WF_data.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
        ax = axs[1]
        ax.set_title('Model')
        WF.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
        ax = axs[2]
        ax.set_title('Residual')
        WF.plot_field(row + '_residual', fig=fig, ax=ax, a=vmindiff, b=vmaxdiff)
        fig.savefig(plot_dir + '/{0}_{1}.png'.format(row, expid))
        fig.savefig(plot_dir + '/{0}_{1}.pdf'.format(row, expid))

        # TODO: Save each axis individually as well:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
        ax.set_title('Expid: {0}, {1}'.format(expid, row))
        vmin = np.nanmin((field_model[row].min(),
                          field_data[row].min()))
        vmax = np.nanmax((field_data[row].max(),
                          field_model[row].max()))
        vmindiff = field_model[row + '_residual'].min()
        vmaxdiff = field_model[row + '_residual'].max()
        WF_data.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
        fig.savefig(plot_dir + '/{0}_{1}_data.png'.format(row, expid))
        fig.savefig(plot_dir + '/{0}_{1}_data.pdf'.format(row, expid))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
        ax.set_title('Expid: {0}, {1}'.format(expid, row))
        WF.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
        fig.savefig(plot_dir + '/{0}_{1}_model.png'.format(row, expid))
        fig.savefig(plot_dir + '/{0}_{1}_model.pdf'.format(row, expid))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
        ax.set_title('Expid: {0}, {1}'.format(expid, row))
        WF.plot_field(row + '_residual', fig=fig, ax=ax, a=vmindiff, b=vmaxdiff)
        fig.savefig(plot_dir + '/{0}_{1}_residual.png'.format(row, expid))
        fig.savefig(plot_dir + '/{0}_{1}_residual.pdf'.format(row, expid))
        plt.close('all')

    # save whisker plots too
    # do w, e, and normalized e.
    # for each: data and model separate, data plus model, residual

    ###########################################################################
    # w
    ###########################################################################
    num_spokes = 2
    scalefactor = 0.2
    scalefactor_residual = 0.5
    quiverdict = {'width': 2}
    # w
    # data
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=False,
                              fig=None, ax=None,
                              scalefactor=scalefactor,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='e1_data', e2key='e2_data',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_w_{0}_data.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_w_{0}_data.pdf'.format(expid))

    # w
    # model
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=False,
                              fig=None, ax=None,
                              scalefactor=scalefactor,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='e1', e2key='e2',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_w_{0}_model.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_w_{0}_model.pdf'.format(expid))

    # w
    # data + model
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=False,
                              fig=None, ax=None,
                              scalefactor=scalefactor,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='e1_data', e2key='e2_data',
                              do_var=False, legend=False, quiverdict=quiverdict)
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=False,
                              fig=fig, ax=ax,
                              scalefactor=scalefactor, num_spokes=num_spokes,
                              color='red',
                              e1key='e1', e2key='e2',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_w_{0}_blackdata_redmodel.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_w_{0}_blackdata_redmodel.pdf'.format(expid))

    # w
    # residual
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=False,
                              fig=None, ax=None,
                              scalefactor=scalefactor_residual,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='e1_residual', e2key='e2_residual',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_w_{0}_residual.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_w_{0}_residual.pdf'.format(expid))
    ###########################################################################


    ###########################################################################
    # e
    ###########################################################################
    num_spokes = 1
    scalefactor = 1
    scalefactor_residual = 5
    quiverdict = {'width': 2}
    # e
    # data
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=False,
                              fig=None, ax=None,
                              scalefactor=scalefactor,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='e1_data', e2key='e2_data',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_e_{0}_data.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_e_{0}_data.pdf'.format(expid))

    # e
    # model
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=False,
                              fig=None, ax=None,
                              scalefactor=scalefactor,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='e1', e2key='e2',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_e_{0}_model.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_e_{0}_model.pdf'.format(expid))

    # e
    # data + model
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=False,
                              fig=None, ax=None,
                              scalefactor=scalefactor,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='e1_data', e2key='e2_data',
                              do_var=False, legend=False, quiverdict=quiverdict)
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=False,
                              fig=fig, ax=ax,
                              scalefactor=scalefactor, num_spokes=num_spokes,
                              color='red',
                              e1key='e1', e2key='e2',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_e_{0}_blackdata_redmodel.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_e_{0}_blackdata_redmodel.pdf'.format(expid))

    # e
    # residual
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=False,
                              fig=None, ax=None,
                              scalefactor=scalefactor_residual,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='e1_residual', e2key='e2_residual',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_e_{0}_residual.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_e_{0}_residual.pdf'.format(expid))
    ###########################################################################

    ###########################################################################
    # E
    ###########################################################################
    num_spokes = 1
    scalefactor = 1
    scalefactor_residual = 5
    quiverdict = {'width': 2}
    # E
    # data
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=True,
                              fig=None, ax=None,
                              scalefactor=scalefactor,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='E1norm_data', e2key='E2norm_data',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_Enorm_{0}_data.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_Enorm_{0}_data.pdf'.format(expid))

    # E
    # model
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=True,
                              fig=None, ax=None,
                              scalefactor=scalefactor,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='E1norm', e2key='E2norm',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_Enorm_{0}_model.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_Enorm_{0}_model.pdf'.format(expid))

    # E
    # data + model
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=True,
                              fig=None, ax=None,
                              scalefactor=scalefactor,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='E1norm_data', e2key='E2norm_data',
                              do_var=False, legend=False, quiverdict=quiverdict)
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=True,
                              fig=fig, ax=ax,
                              scalefactor=scalefactor, num_spokes=num_spokes,
                              color='red',
                              e1key='E1norm', e2key='E2norm',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_Enorm_{0}_blackdata_redmodel.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_Enorm_{0}_blackdata_redmodel.pdf'.format(expid))

    # E
    # residual
    fig, ax = WF.plot_whisker(WF.field, num_bins=num_bins_whisker,
                              normalized_ellipticity=True,
                              fig=None, ax=None,
                              scalefactor=scalefactor_residual,
                              num_spokes=num_spokes,
                              color='black',
                              e1key='E1norm_residual', e2key='E2norm_residual',
                              do_var=False, legend=True, quiverdict=quiverdict)
    fig.savefig(plot_dir + '/whisker_Enorm_{0}_residual.png'.format(expid))
    fig.savefig(plot_dir + '/whisker_Enorm_{0}_residual.pdf'.format(expid))
    ###########################################################################


    # make plot of N
    fig, ax = plt.subplots(figsize=(6,5))
    WF_data.plot_field('N', fig=fig, ax=ax)
    fig.savefig(plot_dir + '/N_{0}.png'.format(expid))
    plt.close('all')

    # save summary statistics of stars to npy file for later collection
    if aos:
        field_jamie = pd.read_pickle(pkl_dir + '/{0:08d}.pkl'.format(expid))
        for row in rows:
            WF.field[row + '_jamie'] = field_jamie[row]
            WF.field[row + '_jamie_residual'] = WF.field[row + '_data'] - \
                                                WF.field[row + '_jamie']
        WF.field.to_pickle(pkl_dir + '/aos_{0:08d}.pkl'.format(expid))
    else:
        WF.field.to_pickle(pkl_dir + '/{0:08d}.pkl'.format(expid))

###############################################################################
# collect
###############################################################################
def collect():
    # take the results of all the runs and collect them into a big hdf5
    from glob import glob
    pickles = glob(pkl_dir + '/*.pkl')
    out_h5 = pkl_dir + '/results.h5'

    for pickle_i, pickle in enumerate(pickles):
        print(pickle_i, len(pickles))
        field = pd.read_pickle(pickle)
        # gotta rename the indices
        field.index.rename(['x_int', 'y_int'], inplace=True)
        field.to_hdf(out_h5, 'data',
                     format='table', mode='a', append=True)

###############################################################################
# meshes
###############################################################################
def meshes(mesh_i=-1):
    # make wavefront plots from the by_date meshes and show how they affect
    # params. also show difference from the nominal wavefront of all of them
    # combined
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from WavefrontPSF.psf_interpolator import Mesh_Interpolator, kNN_Interpolator
    from WavefrontPSF.wavefront import generate_one_coordinate_per_bin
    from WavefrontPSF.donutengine import DECAM_Model_Wavefront

    from glob import glob
    mesh_names = glob(new_mesh_directory + '/z4*Science-20140212s2-v20i2*.dat')
    mesh_names = [mesh_name.split('z4Mesh_')[-1].split('.dat')[0] for mesh_name in mesh_names]
    if mesh_i != -1:
        mesh_names_enumerate = [mesh_names[mesh_i]]
    else:
        mesh_names_enumerate = mesh_names
    rzero = 0.15

    medsubkeys = ['e0', 'e1', 'e2', 'E1norm', 'E2norm',
                  'delta1', 'delta2', 'zeta1', 'zeta2']
    medsubkeys += ['z{0}'.format(zi) for zi in xrange(4, 12)]

    rows = medsubkeys + [key + '_medsub' for key in medsubkeys]

    ###########################################################################
    # generate just from donuts used
    ###########################################################################
    num_bins = 2

    for indx, mesh_name in enumerate(mesh_names_enumerate):
        plot_dir = out_dir + '/plots_meshes/{0}'.format(mesh_name)
        if not path.exists(plot_dir):
            makedirs(plot_dir)
        print(indx, mesh_name, len(mesh_names))
        PSF_Interpolator = Mesh_Interpolator(mesh_name=mesh_name, directory=new_mesh_directory)
        # subtract the median z4 seeing to put this nominally in focus
        PSF_Interpolator.data['z4'] = PSF_Interpolator.data['z4'] - \
                                      PSF_Interpolator.data['z4'].median()

        # generate mesh that contains all EXCEPT this mesh
        mesh_names_ex = [mesh_name_ex for mesh_name_ex in mesh_names
                         if mesh_name_ex != mesh_name]
        for indx_ex, mesh_name_ex in enumerate(mesh_names_ex):
            PSF_Interpolator_ex = Mesh_Interpolator(mesh_name=mesh_name_ex,
                                                    directory=new_mesh_directory)
            # subtract the median z4 seeing to put this nominally in focus
            PSF_Interpolator_ex.data['z4'] = PSF_Interpolator_ex.data['z4'] - \
                                             PSF_Interpolator_ex.data['z4'].median()
            if indx == 0:
                data = PSF_Interpolator_ex.data.copy()
                data['mesh_name'] = mesh_name_ex
            else:
                # append PSF_Interpolator data to All_PSF_Interpolator
                data_i = PSF_Interpolator_ex.data.copy()
                data_i['mesh_name'] = mesh_name_ex
                data = data.append(data_i, ignore_index=True)
        Rest_PSF_Interpolator = kNN_Interpolator(data)

        print('Generating WF')
        # generate the two WFs
        WF = DECAM_Model_Wavefront(PSF_Interpolator=PSF_Interpolator,
                                   num_bins=num_bins)

        # generate coordinates
        x, y = generate_one_coordinate_per_bin(num_bins)
        model = pd.DataFrame({'x': x, 'y': y})
        model = Rest_PSF_Interpolator(model, force_interpolation=True)

        WF_rest = DECAM_Model_Wavefront(PSF_Interpolator=Rest_PSF_Interpolator,
                                        num_bins=num_bins)

        misalignment = {'rzero': rzero}
        # create model fit from donuts
        PSF_Interpolator.data['rzero'] = misalignment['rzero']
        WF.data = WF(PSF_Interpolator.data, misalignment=misalignment)
        WF.reduce(num_bins=num_bins)
        model['rzero'] = misalignment['rzero']
        WF_rest.data = WF_rest(model, misalignment=misalignment)
        WF_rest.reduce(num_bins=num_bins)

        # add corrections to norms in field
        WF.field['E1norm'] = WF.field['e1'] / WF.field['e0']
        WF.field['E2norm'] = WF.field['e2'] / WF.field['e0']
        WF_rest.field['E1norm'] = WF_rest.field['e1'] / WF_rest.field['e0']
        WF_rest.field['E2norm'] = WF_rest.field['e2'] / WF_rest.field['e0']

        # collate the medsubs
        for key in medsubkeys:
            WF.field[key + '_medsub'] = WF.field[key] - WF.field[key].median()
            WF_rest.field[key + '_medsub'] = WF_rest.field[key] - WF_rest.field[key].median()

        # now add residual to WF
        for row in rows:
            WF.field[row + '_residual'] = WF_rest.field[row] - WF.field[row]

        print('Making plots')
        # plots!
        for row in rows:
            ncols = 3
            nrows = 1
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                    figsize=(6*ncols, 5*nrows))
            fig.suptitle('{0}, {1}'.format(mesh_name, row))
            vmin = np.nanmin((WF.field[row].min(),
                              WF_rest.field[row].min()))
            vmax = np.nanmax((WF.field[row].max(),
                              WF_rest.field[row].max()))
            vmindiff = WF.field[row + '_residual'].min()
            vmaxdiff = WF.field[row + '_residual'].max()
            ax = axs[0]
            ax.set_title('Baseline Minus Exposure')
            WF_rest.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
            ax = axs[1]
            ax.set_title('Single Exposure')
            WF.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
            ax = axs[2]
            ax.set_title('Residual')
            WF.plot_field(row + '_residual', fig=fig, ax=ax, a=vmindiff, b=vmaxdiff)
            fig.savefig(plot_dir + '/{0}_{1}.png'.format(row, mesh_name))
            fig.savefig(plot_dir + '/{0}_{1}.pdf'.format(row, mesh_name))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
            fig.suptitle('{0}, {1}'.format(mesh_name, row))
            ax.set_title('Baseline Minus Exposure')
            WF_rest.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
            fig.savefig(plot_dir + '/{0}_{1}_baseline.png'.format(row, mesh_name))
            fig.savefig(plot_dir + '/{0}_{1}_baseline.pdf'.format(row, mesh_name))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
            fig.suptitle('{0}, {1}'.format(mesh_name, row))
            ax.set_title('Single Exposure')
            WF.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
            fig.savefig(plot_dir + '/{0}_{1}_exposure.png'.format(row, mesh_name))
            fig.savefig(plot_dir + '/{0}_{1}_exposure.pdf'.format(row, mesh_name))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
            fig.suptitle('{0}, {1}'.format(mesh_name, row))
            ax.set_title('Residual')
            WF.plot_field(row + '_residual', fig=fig, ax=ax, a=vmindiff, b=vmaxdiff)
            fig.savefig(plot_dir + '/{0}_{1}_residual.png'.format(row, mesh_name))
            fig.savefig(plot_dir + '/{0}_{1}_residual.pdf'.format(row, mesh_name))
            plt.close('all')


        #######################################################################
        # generate interpolated model
        #######################################################################
        print('interpolated model')
        num_bins = 5
        # generate coordinates
        x, y = generate_one_coordinate_per_bin(num_bins)
        model = pd.DataFrame({'x': x, 'y': y})
        model = PSF_Interpolator(model, force_interpolation=True)

        # generate the two WFs
        WF = DECAM_Model_Wavefront(PSF_Interpolator=PSF_Interpolator,
                                   num_bins=num_bins)

        # generate coordinates
        model_rest = pd.DataFrame({'x': x, 'y': y})
        model_rest = Rest_PSF_Interpolator(model_rest, force_interpolation=True)

        WF_rest = DECAM_Model_Wavefront(PSF_Interpolator=Rest_PSF_Interpolator,
                                        num_bins=num_bins)

        misalignment = {'rzero': rzero}
        # create model fit from donuts
        model['rzero'] = misalignment['rzero']
        WF.data = WF(model, misalignment=misalignment)
        WF.reduce(num_bins=num_bins)
        model_rest['rzero'] = misalignment['rzero']
        WF_rest.data = WF_rest(model_rest, misalignment=misalignment)
        WF_rest.reduce(num_bins=num_bins)

        # add corrections to norms in field
        WF.field['E1norm'] = WF.field['e1'] / WF.field['e0']
        WF.field['E2norm'] = WF.sdffield['e2'] / WF.field['e0']
        WF_rest.field['E1norm'] = WF_rest.field['e1'] / WF_rest.field['e0']
        WF_rest.field['E2norm'] = WF_rest.field['e2'] / WF_rest.field['e0']

        # collate the medsubs
        for key in medsubkeys:
            WF.field[key + '_medsub'] = WF.field[key] - WF.field[key].median()
            WF_rest.field[key + '_medsub'] = WF_rest.field[key] - WF_rest.field[key].median()

        # now add residual to WF
        for row in rows:
            WF.field[row + '_residual'] = WF_rest.field[row] - WF.field[row]

        print('plots')
        # plots!
        for row in rows:
            ncols = 3
            nrows = 1
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                    figsize=(6*ncols, 5*nrows))
            fig.suptitle('{0}, {1}'.format(mesh_name, row))
            vmin = np.nanmin((WF.field[row].min(),
                              WF_rest.field[row].min()))
            vmax = np.nanmax((WF.field[row].max(),
                              WF_rest.field[row].max()))
            vmindiff = WF.field[row + '_residual'].min()
            vmaxdiff = WF.field[row + '_residual'].max()
            ax = axs[0]
            ax.set_title('Baseline Minus Exposure')
            WF_rest.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
            ax = axs[1]
            ax.set_title('Single Exposure')
            WF.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
            ax = axs[2]
            ax.set_title('Residual')
            WF.plot_field(row + '_residual', fig=fig, ax=ax, a=vmindiff, b=vmaxdiff)
            fig.savefig(plot_dir + '/interp_{0}_{1}.png'.format(row, mesh_name))
            fig.savefig(plot_dir + '/interp_{0}_{1}.pdf'.format(row, mesh_name))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
            fig.suptitle('{0}, {1}'.format(mesh_name, row))
            ax.set_title('Baseline Minus Exposure')
            WF_rest.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
            fig.savefig(plot_dir + '/interp_{0}_{1}_baseline.png'.format(row, mesh_name))
            fig.savefig(plot_dir + '/interp_{0}_{1}_baseline.pdf'.format(row, mesh_name))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
            fig.suptitle('{0}, {1}'.format(mesh_name, row))
            ax.set_title('Single Exposure')
            WF.plot_field(row, fig=fig, ax=ax, a=vmin, b=vmax)
            fig.savefig(plot_dir + '/interp_{0}_{1}_exposure.png'.format(row, mesh_name))
            fig.savefig(plot_dir + '/interp_{0}_{1}_exposure.pdf'.format(row, mesh_name))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
            fig.suptitle('{0}, {1}'.format(mesh_name, row))
            ax.set_title('Residual')
            WF.plot_field(row + '_residual', fig=fig, ax=ax, a=vmindiff, b=vmaxdiff)
            fig.savefig(plot_dir + '/interp_{0}_{1}_residual.png'.format(row, mesh_name))
            fig.savefig(plot_dir + '/interp_{0}_{1}_residual.pdf'.format(row, mesh_name))
            plt.close('all')


###############################################################################
# submit
###############################################################################
def submit(aos=False):
    from subprocess import call, check_output, STDOUT
    for expid_i, expid in enumerate(jamierod_results.index):
        if expid_i % 100 == 0:
            print(expid_i, len(jamierod_results.index))
        if aos:
            jobname = 'aos_plots_{0:08d}'.format(expid)
            if path.exists(pkl_dir + '/aos_{0:08d}.pkl'.format(expid)):
                continue
        else:
            jobname = 'plots_{0:08d}'.format(expid)
            if path.exists(pkl_dir + '/{0:08d}.pkl'.format(expid)):
                continue
        logfile = log_dir + '/{0}.log'.format(jobname)
        # check jobname
        jobcheck = check_output(['bjobs', '-J', jobname], stderr=STDOUT)
        if 'not found' not in jobcheck:
            continue
        print(jobname, logfile)


        command = ['bsub',
                   '-J', jobname,
                   '-o', logfile,
                   '-W', str(req_time),
                   'python', code_path,
                   '--job', 'run',
                   '--expid', str(expid)]
        if aos:
            command.append('--aos')
        call(command)

###############################################################################
# submit_meshes
###############################################################################
def submit_meshes():
    from subprocess import call, check_output, STDOUT
    from glob import glob
    mesh_names = glob(new_mesh_directory + '/z4*Science-20140212s2-v20i2*.dat')
    mesh_names = [mesh_name.split('z4Mesh_')[-1].split('.dat')[0] for mesh_name in mesh_names]

    for mesh_i, mesh_name in enumerate(mesh_names):
        if mesh_i % 100 == 0:
            print(mesh_i, len(mesh_names))
        jobname = 'meshes_{0}'.format(mesh_name)
        logfile = log_dir + '/{0}.log'.format(jobname)
        # check jobname
        jobcheck = check_output(['bjobs', '-J', jobname], stderr=STDOUT)
        if 'not found' not in jobcheck:
            continue
        print(jobname, logfile)

        command = ['bsub',
                   '-J', jobname,
                   '-o', logfile,
                   '-W', str(60),
                   'python', code_path,
                   '--job', 'meshes',
                   '--expid', str(mesh_i)]
        call(command)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='lastkpc_plots')
    parser.add_argument("--job",
                        dest="job",
                        default="submit",
                        help="submit, run")
    parser.add_argument("--expid",
                        dest="expid",
                        type=int,
                        default=-1,
                        help="expid")
    parser.add_argument("--aos",
                        dest="aos",
                        action="store_true",
                        help='do aos version')
    parser.set_defaults(aos=False)
    options = parser.parse_args()
    args = vars(options)

    if args['job'] == 'submit':
        submit(args['aos'])
    elif args['job'] == 'collect':
        collect()
    elif args['job'] == 'submit_meshes':
        submit_meshes()
    elif args['job'] == 'meshes':
        meshes(args['expid'])
    elif args['job'] == 'run':
        do_run(args['expid'], args['aos'])

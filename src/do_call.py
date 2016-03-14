#@Author Chris Davis
#Send a random batch of exposures to ki-ls
#optionally, specify the expids from the command line
from __future__ import print_function, division
import numpy as np
import pandas as pd
from subprocess import call
from os import path
from sys import argv

jamierod_results_path = '/nfs/slac/g/ki/ki18/des/cpd/jamierod_results.csv'
jamierod_results = pd.read_csv(jamierod_results_path)

#out_dir = '/nfs/slac/g/ki/ki18/des/cpd/DeconvOutput'
out_dir = '/nfs/slac/g/ki/ki18/des/swmclau2/DeconvOutput'
#code_path = '/nfs/slac/g/ki/ki18/cpd/Projects/WavefrontPSF/code/DeconvolvePSF/afterburner.py'
code_path = '/u/ki/swmclau2/Git/DeconvolvePSF/DeconvolvePSF/afterburner.py'

if len(argv) == 1: #do a random collection of 40
    # choose a random 40 expids from jamierod results
    indx_choice = np.random.choice(len(jamierod_results), 40)
    expids = jamierod_results.iloc[indx_choice]['expid']

    expids = list(expids)
else:
    expids = [int(expid) for expid in sys[1:] ]

req_time = 240 # minutes

for expid in expids:
    print(expid)
    # check that the expid doesn't exist in the output
    if not path.exists(out_dir + '/{0:08d}'.format(expid)):
        jobname = '{0}PSF'.format(expid)
        logfile = out_dir + '/{0}.log'.format(expid)
        command = ['bsub',
                   '-J', jobname,
                   '-o', logfile,
                   '-W', str(req_time),
                   'python', code_path,
                   str(expid), out_dir]
        call(command)
    else:
        print('{0} exists!'.format(expid))

from __future__ import print_function, division
import numpy as np
import pandas as pd
from subprocess import call
from os import path

jamierod_results_path = '/nfs/slac/g/ki/ki18/des/cpd/jamierod_results.csv'
jamierod_results = pd.read_csv(jamierod_results_path)

# choose a random 40 expids from jamierod results
indx_choice = np.random.choice(len(jamierod_results), 40)
expids = jamierod_results.iloc[indx_choice]['expid']

expids = list(expids)
expids.append(149440)

#out_dir = '/nfs/slac/g/ki/ki18/des/cpd/DeconvOutput'
out_dir = '/nfs/slac/g/ki/ki18/des/swmclau2/DeconvOutput'
#code_path = '/nfs/slac/g/ki/ki18/cpd/Projects/WavefrontPSF/code/DeconvolvePSF/afterburner.py'
code_path = '/u/ki/swmclau2/Git/DeconvolvePSF/DeconvolvePSF/afterburner.py'

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

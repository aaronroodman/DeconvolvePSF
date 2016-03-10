#@Author Sean McLaughlin
# Send one exposure to the ki-ls cluster for analysis.
# TODO Merge this with do_call; it's copied code!
from __future__ import print_function, division
import pandas as pd
from subprocess import call
from os import path
from sys import argv

expid = int(argv[1])

jamierod_results_path = '/nfs/slac/g/ki/ki18/des/cpd/jamierod_results.csv'
jamierod_results = pd.read_csv(jamierod_results_path)

# choose a random 40 expids from jamierod results

# out_dir = '/nfs/slac/g/ki/ki18/des/cpd/DeconvOutput'
out_dir = '/nfs/slac/g/ki/ki18/des/swmclau2/DeconvOutput'
# code_path = '/nfs/slac/g/ki/ki18/cpd/Projects/WavefrontPSF/code/DeconvolvePSF/afterburner.py'
code_path = '/u/ki/swmclau2/Git/DeconvolvePSF/DeconvolvePSF/afterburner.py'

req_time = 240 # minutes

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

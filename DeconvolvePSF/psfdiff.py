
#
# deconvolution, etc...
#
import numpy as np
import numpy.ma as ma
from WavefrontPSF.stamp_collector import stamp_collector
from WavefrontPSF.psf_evaluator import Moment_Evaluator
from matplotlib import pyplot as plt
plt.interactive(True)


expid = 181243

fit_rzero = 0.168
fit_fwhm = 0.14/0.168
use_fwhm = np.sqrt(fit_fwhm*fit_fwhm - 0.3*0.3)
use_rzero = 0.14/use_fwhm

psf_df, model_stamps, decam_stamps = stamp_collector(expid, Nmax=0, rzero=use_rzero, snr_max=400, snr_min=90)

evaluator = Moment_Evaluator()


# model_stamps is the PSF model
# decam_stamps is the data...

# mask bad pixel data
# and crop it to 32x32
# this puts the center pixel in the 63x63 just below the center line in the 32x32 array
decam_stamps_clean = np.where(decam_stamps<1e6,decam_stamps,0.)
decam_stamps_clean = np.where(decam_stamps_clean>-1e5,decam_stamps_clean,0.)
decam_stamps_crop = decam_stamps_clean[:,16:48,16:48]

moments_decam = evaluator(decam_stamps_crop.astype(np.float64))
moments_model = evaluator(model_stamps)

Mxx = moments_decam['Mxx'] - moments_model['Mxx']
Myy = moments_decam['Myy'] - moments_model['Myy']
Mxy = moments_decam['Mxy'] - moments_model['Mxy']

# now try deconvolving...
# move RL code into GitHub
# setup on ki-ls

# list of expid analyzed
from glob import glob
import numpy as np
params_list = glob('/nfs/slac/g/ki/ki18/des/jamierod/SVA1/params/*.npy')
expids = np.sort([int(param.split('.npy')[-2].split('_')[-1]) for param in params_list])
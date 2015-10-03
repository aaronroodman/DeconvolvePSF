
#
# deconvolution, etc...
#
import numpy as np
import numpy.ma as ma
from scipy.stats import tstd
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
plt.interactive(True)

from WavefrontPSF.stamp_collector import stamp_collector
from WavefrontPSF.psf_evaluator import Moment_Evaluator
from DeconvolvePSF.lucy import deconvolve, convolve, makeMask, makeGaussian



class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.04f}'.format(x, y, z)


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

# find background level from RMS of decam_stamps data

# now try deconvolving...
stdback = tstd(decam_stamps_clean.flatten(),limits=(-200,200))
background = stdback*stdback

#for i in range(decam_stamps_crop.shape[0]):

for i in range(10):

    psi0 = makeGaussian((32,32),Mxx[i],Myy[i],Mxy[i])
    
    # now try R-L
    deconImage,diffs,psiByIter,chi2ByIter = deconvolve(model_stamps[i],decam_stamps_crop[i],psi_0=psi0,mask=None,mu0=background,convergence=0.5e-3,chi2Level=1024.,niterations=1000,extra=True)

    # convolve back
    predictImage = convolve(model_stamps[i],deconImage)

    f1,axArr = plt.subplots(3,2)
    im00 = axArr[0,0].imshow(decam_stamps_crop[i]/np.sum(decam_stamps_crop[i]),interpolation='none',origin='lower',cmap='hot')
    ##,norm=LogNorm(vmin=1.0e-4, vmax=1.0))
    #axArr[0,0].colorbar(im00)
    axArr[0,0].format_coord = Formatter(im00)
    im01 = axArr[0,1].imshow(model_stamps[i]/np.sum(model_stamps[i]),interpolation='none',origin='lower',cmap='hot')
    ##norm=LogNorm(vmin=1.0e-4, vmax=1.0))
    #axArr[0,1].colorbar(im01)
    axArr[0,1].format_coord = Formatter(im01)
    
    im10 = axArr[1,0].imshow(psi0,interpolation='none',origin='lower',cmap='hot',norm=LogNorm(vmin=1.0e-4, vmax=1.0))
    #axArr[1,0].colorbar(im10)
    axArr[1,0].format_coord = Formatter(im10)
    im11 = axArr[1,1].imshow(deconImage,interpolation='none',origin='lower',cmap='hot',norm=LogNorm(vmin=1.0e-4, vmax=1.0))
    #axArr[1,1].colorbar(im11)
    axArr[1,1].format_coord = Formatter(im11)    

    im20 = axArr[2,0].imshow(predictImage,interpolation='none',origin='lower',cmap='hot')
    ##,norm=LogNorm(vmin=1.0e-4, vmax=1.0))
    #axArr[2,0].colorbar(im20)
    axArr[2,0].format_coord = Formatter(im20)
    im21 = axArr[2,1].imshow(decam_stamps_crop[i]/np.sum(decam_stamps_crop[i])-predictImage,interpolation='none',origin='lower',cmap='hot',vmin=-0.01, vmax=0.01)
    #axArr[2,1].colorbar(im21)
    axArr[2,1].format_coord = Formatter(im21)    

    f1.show()

    print "flux_adaptive,snr from sextractor",psf_df['FLUX_ADAPTIVE'].ilog[i],psf_df['SNR_WIN'].iloc[i]
    print "n iterations",len(diffs)
    print "starting Diff,Chi2", diffs[0],chi2ByIter[0]
    print "ending Diff,Chi2", diffs[-1],chi2ByIter[-1]


    

# list of expid analyzed
from glob import glob
import numpy as np
params_list = glob('/nfs/slac/g/ki/ki18/des/jamierod/SVA1/params/*.npy')
expids = np.sort([int(param.split('.npy')[-2].split('_')[-1]) for param in params_list])